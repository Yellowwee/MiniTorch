"""Transactional transient view of independent paged caches for ragged decode."""
import numpy as np
from .cuda_storage import RawDeviceBuffer
from .cuda_resident_ops import CudaResidentOps as Ops, call, pointer, P, I
from .dtypes import dtype_code
from .paged_kv_cache import PagedKVCache, _library
from .tensor_functions import tensor_from_numpy


class PagedDecodeBatch:
    """Batched projections/MLP plus per-row paged attention, on the default stream.

    Owns only temporary metadata. Child caches retain all KV pages and committed
    lengths. A failed execution invalidates/aborts every participating cache.
    Callers must not mutate caches concurrently.
    """
    def __init__(self, model, idx, caches):
        self.caches = tuple(caches)
        if not self.caches or len({id(c) for c in self.caches}) != len(self.caches):
            raise ValueError('Decode batch needs nonempty, unique caches')
        if any(not isinstance(c,PagedKVCache) for c in self.caches):
            raise ValueError('Batched decode requires paged caches')
        self.pool = self.caches[0].pool
        self.backend, self.dtype, self.layout = model.backend,self.pool.dtype,self.pool.layout
        self.batch_size = len(self.caches)
        if idx.shape != (self.batch_size,1) or idx.backend is not self.backend or idx.dtype != 'float32':
            raise ValueError('Expected FP32 token IDs [request_count,1] on model backend')
        for cache in self.caches:
            cache._validate_model(model)
            if (cache.pool is not self.pool or self.pool.closed or cache.batch_size!=1 or
                cache._pending_tokens is not None or not 0<cache.length<cache.capacity):
                raise ValueError('Decode caches must share a live pool, have batch=1 and remaining capacity')
        self.offsets = np.array([c.length for c in self.caches],np.int32)
        self.columns = max((c.length+1+self.pool.page_size-1)//self.pool.page_size for c in self.caches)
        self.table = self.device_offsets = self.position_ids = None
        self._prepared, self._written_layers = [],set()
        self._committed = False
        self.last_attention_path = 'paged_batch_decode'

    def prepare(self):
        if self._prepared or self._committed:
            raise ValueError('Decode batch already executed')
        # Context manager aborts every cache, including the one whose append fails.
        for cache in self.caches:
            self._prepared.append(cache)
            cache.prepare_append(1)
        table = np.full((self.batch_size,self.columns),-1,np.int32)
        for i,cache in enumerate(self.caches):
            table[i,:len(cache._pages[0])] = cache._pages[0]
        self.table = RawDeviceBuffer(table.nbytes)
        self.table.upload(table)
        self.device_offsets = RawDeviceBuffer(self.offsets.nbytes)
        self.device_offsets.upload(self.offsets)
        self.position_ids = tensor_from_numpy(self.offsets.astype(np.float32).reshape(-1,1),self.backend)

    def decode_attention(self, layer, q, k, v):
        if self.table is None or layer in self._written_layers or not 0<=layer<len(self.layout):
            raise ValueError('Invalid decode batch layer/transaction')
        h,d = self.layout[layer]
        if any(t.shape!=(self.batch_size,h,1,d) or t.backend is not self.backend or t.dtype!=self.dtype for t in (q,k,v)):
            raise ValueError('Invalid batched decode Q/K/V')
        q,k,v = (t.contiguous() for t in (q,k,v))
        out = Ops.empty(q.shape,self.backend,self.dtype)
        pool = self.pool
        call(_library(),'mt_paged_decode_batch',[P]*7+[I]*10,
             pointer(out,True),pointer(q,True),pointer(k,True),pointer(v,True),
             pool.slab_table.ptr,self.table.ptr,self.device_offsets.ptr,
             self.batch_size,h,d,pool.page_size,pool.pages_per_slab,len(pool._slabs),self.columns,
             layer,len(self.layout),dtype_code(self.dtype))
        self._written_layers.add(layer)
        for cache in self.caches:
            cache._written_layers.add(layer)
            cache.last_attention_path = self.last_attention_path
        return out

    def commit(self):
        if len(self._written_layers)!=len(self.layout):
            raise ValueError('Cannot commit incomplete decode batch')
        # Check every transaction before changing any committed length.
        if any(c._pending_tokens!=1 or len(c._written_layers)!=len(c.layout) for c in self.caches):
            raise ValueError('Decode transaction modified during execution')
        for cache in self.caches: cache.commit_append()
        self._committed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self._committed:
            for cache in self._prepared: cache.abort_append()
        for buffer in (self.table,self.device_offsets):
            if buffer is not None: buffer.close()
        self.position_ids = None
