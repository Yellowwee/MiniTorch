"""Transactional inference cache interface and contiguous implementation.

All batch members have the same live length. This module's KVCache is contiguous;
PagedKVCache implements the same transaction interface in paged_kv_cache.py.
Padding masks, beam reordering, concurrent use and training writes are unsupported.
"""
import ctypes as ct
from functools import lru_cache
from pathlib import Path
import weakref
from .autodiff import is_grad_enabled
from .amp import autocast_dtype
from .dtypes import dtype_code, itemsize
from .cuda_resident_ops import CudaResidentOps as Ops, call, pointer, P, I
from .cuda_storage import runtime


@lru_cache(maxsize=1)
def _library():
    try:
        return ct.CDLL(str(Path(__file__).parent/'cuda_kernels/inference_kernels.so'))
    except OSError as error:
        raise RuntimeError('Build inference_kernels.so with compile_cuda.sh') from error


def last_position(x):
    """Copy only the last sequence position, without recording an autograd op."""
    if is_grad_enabled() or not x.backend.device_storage or x.dims != 3:
        raise ValueError('last_position requires no_grad and device [B,T,C] input')
    b,n,w = x.shape
    if min(b,n,w) < 1:
        raise ValueError('Empty last-position input')
    x = x.contiguous()
    out = Ops.empty((b,1,w),x.backend,x.dtype)
    call(_library(),'mt_last_row',[P,P,I,I,I,I],pointer(out,True),pointer(x,True),b,n,w,dtype_code(x.dtype))
    return out


def greedy_token(logits):
    """GPU argmax, returning FP32 token IDs [B,1] suitable for embeddings.

    Only call .item()/to_numpy() on this small result for EOS/output handling.
    Ties and NaNs choose the first index, matching numpy.argmax.
    """
    if is_grad_enabled() or not logits.backend.device_storage or logits.dims != 3 or logits.shape[1] != 1:
        raise ValueError('greedy_token requires no_grad and device logits [B,1,V]')
    b,_,v = logits.shape
    if b<1 or not 1 <= v <= 2**24:
        raise ValueError('Vocabulary must fit exact FP32 token IDs')
    logits = logits.contiguous()
    out = Ops.empty((b,1),logits.backend)
    call(_library(),'mt_row_argmax',[P,P,I,I,I],pointer(out),pointer(logits,True),b,v,dtype_code(logits.dtype))
    return out


class CacheBase:
    def __init__(self, model, batch_size=1, capacity=None):
        if not model.backend.device_storage:
            raise ValueError('KV cache requires storage_mode=device')
        if model.training:
            raise ValueError('Call model.eval() before creating an inference cache')
        capacity = model.n_positions if capacity is None else capacity
        if type(batch_size) is not int or batch_size < 1 or type(capacity) is not int or not 1 <= capacity <= model.n_positions:
            raise ValueError('Invalid cache batch size or capacity')
        layers = model.transformer_layers()
        if any(l.attention.attention_impl != 'flash' or not l.attention._flash_supported() for l in layers):
            raise ValueError('Cached inference requires FlashAttention and head_dim<=128')
        self.model = weakref.ref(model)
        self.backend = model.backend
        self.batch_size, self.capacity = batch_size, capacity
        self.dtype = autocast_dtype() or 'float32'
        self.length, self.valid, self.closed = 0, True, False
        self.layout = tuple((l.attention.n_head,l.attention.attn_hidden_dim) for l in layers)
        self._pending_tokens = None
        self._written_layers = set()
        self.weights = self._weights()

    def _weights(self):
        model = self.model()
        if model is None:
            raise ValueError('Cache model no longer exists')
        return tuple((id(p.value),p.value._tensor._storage.owner.version) for p in model.parameters())

    def prefill_attention(self,layer,q,k,v):
        if self.length!=0: raise ValueError('prefill attention requires empty cache')
        return self._attention(layer,q,k,v,prefill=True)

    def decode_attention(self,layer,q,k,v):
        if self.length==0 or self._pending_tokens!=1:
            raise ValueError('decode attention requires populated cache and one token')
        return self._attention(layer,q,k,v,prefill=False)

    def attention(self,layer,q,k,v):
        """Compatibility entry; model execution explicitly selects the phase."""
        return (self.prefill_attention if self.length==0 else self.decode_attention)(layer,q,k,v)

    def _validate_model(self, model):
        if self.closed or not self.valid:
            raise ValueError('Cache is closed or invalid; reset/create a cache')
        if self.model() is not model:
            raise ValueError('Cache/model/backend mismatch')
        if model.training or is_grad_enabled():
            raise ValueError('Cached inference requires eval and no_grad')
        if self.dtype != (autocast_dtype() or 'float32') or self.weights != self._weights():
            raise ValueError('Cache precision or model weights changed; reset/create a cache')
        if any(l.attention.attention_impl != 'flash' for l in model.transformer_layers()):
            raise ValueError('Cache requires FlashAttention')

    def validate(self, model, idx, prefill):
        self._validate_model(model)
        if idx.backend is not self.backend:
            raise ValueError('Cache/model/backend mismatch')
        if idx.dims != 2 or idx.shape[0] != self.batch_size or idx.dtype != 'float32':
            raise ValueError('Expected FP32 token IDs [cache_batch, sequence]')
        if prefill and self.length != 0:
            raise ValueError('prefill requires an empty cache')
        if not prefill and (self.length == 0 or idx.shape[1] != 1):
            raise ValueError('decode_step requires a populated cache and one token per batch')
        if idx.shape[1] < 1 or self.length+idx.shape[1] > self.capacity:
            raise ValueError('KV cache capacity exceeded or empty prompt')

    def prepare_append(self, tokens):
        if self.closed or not self.valid or self._pending_tokens is not None:
            raise ValueError('Cache closed, invalid, or append already pending')
        if is_grad_enabled() or self.dtype != (autocast_dtype() or 'float32'):
            raise ValueError('Cache writes require no_grad and matching precision')
        if type(tokens) is not int or tokens < 1 or self.length+tokens > self.capacity:
            raise ValueError('Invalid append size or cache capacity exceeded')
        self._pending_tokens = tokens
        self._written_layers.clear()

    def _check_attention(self, layer, q, k, v):
        if (is_grad_enabled() or self.closed or not self.valid or self._pending_tokens is None
            or type(layer) is not int or not 0 <= layer < len(self.layout)
            or layer in self._written_layers):
            raise ValueError('Invalid cache append transaction or layer')
        h,d = self.layout[layer]
        expected = (self.batch_size,h,self._pending_tokens,d)
        if any(t.shape != expected or t.backend is not self.backend or t.dtype != self.dtype for t in (q,k,v)):
            raise ValueError('Cache append shape/dtype/backend mismatch')

    def commit_append(self):
        if self._pending_tokens is None or len(self._written_layers) != len(self.layout):
            raise ValueError('Cannot commit an incomplete cache append')
        self.length += self._pending_tokens
        self._pending_tokens = None
        self._written_layers.clear()

    def abort_append(self):
        self._pending_tokens = None
        self._written_layers.clear()
        self.valid = False

    def reset(self):
        if self.closed:
            raise ValueError('Cannot reset a closed cache')
        self.weights = self._weights()
        self.length, self.valid = 0, True
        self._pending_tokens = None
        self._written_layers.clear()


class KVCache(CacheBase):
    def __init__(self, model, batch_size=1, capacity=None):
        super().__init__(model,batch_size,capacity)
        self.buffers = [(Ops.empty((batch_size,h,self.capacity,d),self.backend,self.dtype),
                         Ops.empty((batch_size,h,self.capacity,d),self.backend,self.dtype)) for h,d in self.layout]

    @property
    def nbytes(self):
        return sum(t.size*t._tensor._storage.itemsize for pair in self.buffers for t in pair)

    def memory_stats(self):
        capacity = self.nbytes
        actual = sum(t._tensor._storage.owner.nbytes for pair in self.buffers for t in pair)
        live = 0 if self.closed else 2*self.batch_size*self.length*sum(h*d for h,d in self.layout)*itemsize(self.dtype)
        return dict(kv_live_bytes=live,kv_capacity_bytes=capacity,kv_unused_bytes=capacity-live,
            kv_slot_utilization=live/capacity if capacity else None,kv_metadata_bytes=0,
            kv_allocator_slack_bytes=actual-capacity,kv_owned_allocation_bytes=actual)

    def _attention(self, layer, q, k, v, *, prefill):
        self._check_attention(layer,q,k,v)
        kc,vc = self.buffers[layer]
        b,h,n,d = q.shape
        if (k.shape != q.shape or v.shape != q.shape or kc.shape != (b,h,self.capacity,d)
            or any(t.backend is not self.backend or t.dtype != self.dtype for t in (q,k,v))
            or self.closed or not self.valid or self.length+n > self.capacity):
            raise ValueError('Cache append shape/dtype/lifetime mismatch')
        q,k,v = (t.contiguous() for t in (q,k,v))
        self._append_layer_kv(layer,k,v)
        out=Ops.flash_attention_inference(q,k if prefill else kc,v if prefill else vc,
            length=self.length+n,offset=self.length)
        self.last_attention_path='flash_prefill' if prefill else 'contiguous_decode'
        self._written_layers.add(layer)
        return out

    def _append_layer_kv(self,layer,k,v):
        kc,vc=self.buffers[layer]
        b,h,n,d=k.shape
        for src,dst in ((k,kc),(v,vc)):
            call(_library(),'mt_cache_append',[P,P]+[I]*6,pointer(dst,True),pointer(src,True),
                 b*h,n,d,self.capacity,self.length,dtype_code(self.dtype))
            dst._tensor._storage.owner.version += 1

    def close(self):
        self.buffers.clear()
        self.abort_append()
        self.closed, self.valid, self.length = True, False, 0
