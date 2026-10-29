"""Resident CUDA backend with typed basic ops and FP32 fused kernels."""
import ctypes as ct
import math
from contextlib import nullcontext
import numpy as np
from . import operators
from .tensor import Tensor
from .tensor_ops import TensorOps
from .tensor_data import shape_broadcast
from .cuda_storage import DeviceStorage, runtime
from .dtypes import normalize_dtype, dtype_code, itemsize, promote_types


_FUNCTIONS = (operators.add, operators.mul, operators.id, operators.neg,
              operators.lt, operators.eq, operators.sigmoid, operators.relu,
              operators.relu_back, operators.log, operators.log_back,
              operators.exp, operators.inv, operators.inv_back, operators.is_close,
              operators.max, operators.pow, operators.tanh)
_IDS = {fn: i+1 for i, fn in enumerate(_FUNCTIONS)}


class Layout(ct.Structure):
    _fields_ = [('dims', ct.c_int), ('shape', ct.c_int*32), ('strides', ct.c_int*32)]

    @classmethod
    def of(cls, tensor):
        if tensor.dims > 32 or tensor.size > 2**31-1:
            raise ValueError("Resident ABI supports <=32 dimensions and int32 indexing")
        return cls(tensor.dims, (ct.c_int*32)(*tensor.shape),
                   (ct.c_int*32)(*tensor._tensor.strides))


def pointer(tensor, typed=False):
    if not isinstance(tensor._tensor._storage, DeviceStorage):
        raise TypeError("Resident CUDA operation requires device storage")
    if not typed and tensor.dtype != 'float32':
        raise TypeError('This native kernel requires float32; cast explicitly with .to(dtype="float32")')
    return tensor._tensor._storage.ptr


def call(library, name, signature, *args):
    f = getattr(library, name)
    f.argtypes, f.restype = signature, ct.c_int
    runtime().check(f(*args))


P, I, F = ct.c_void_p, ct.c_int, ct.c_float


class CudaResidentOps(TensorOps):
    # cuda is reserved for the older Numba DeviceNDArray path.
    cuda = False
    device_storage = True

    @staticmethod
    def flash_attention_inference(q,k,v,*,length,offset=0):
        """No LSE/autograd state; K/V may have capacity beyond the live prefix."""
        from .autodiff import is_grad_enabled
        if is_grad_enabled() or q.dims!=4 or k.dims!=4 or k.shape!=v.shape:
            raise ValueError('Inference attention requires no_grad and rank-4 Q/K/V')
        b,h,n,d=q.shape
        capacity=k.shape[2]
        if (k.shape!=(b,h,capacity,d) or not q.dtype==k.dtype==v.dtype or
            not q.backend is k.backend is v.backend or offset<0 or offset+n!=length or length>capacity):
            raise ValueError('Invalid inference attention layout or live length')
        q,k,v=(t.contiguous() for t in (q,k,v))
        out=CudaResidentOps.empty(q.shape,q.backend,q.dtype)
        call(runtime().flash_libraries[0],'mt_flash_inference',[P]*4+[I]*8,
            pointer(out,True),pointer(q,True),pointer(k,True),pointer(v,True),
            b,h,n,length,d,capacity,offset,dtype_code(q.dtype))
        return out

    @staticmethod
    def flash_attention_fw(q,k,v,query_lengths,key_lengths,causal):
        if not q.dtype == k.dtype == v.dtype:
            raise ValueError('FlashAttention Q/K/V must have the same dtype')
        q,k,v = (t.contiguous() for t in (q,k,v))
        query_lengths,key_lengths = query_lengths.contiguous(),key_lengths.contiguous()
        b,h,nq,d = q.shape
        nk = k.shape[2]
        out = CudaResidentOps.empty(q.shape,q.backend,q.dtype)
        lse = CudaResidentOps.empty((b,h,nq),q.backend)
        low = q.dtype != 'float32'
        # LSE/lengths always FP32, independent of activation dtype.
        args = [pointer(out,True),pointer(lse),pointer(q,True),pointer(k,True),pointer(v,True),
                pointer(query_lengths),pointer(key_lengths),b,h,nq,nk,d,causal]
        call(runtime().flash_libraries[0],'mt_flash_fw_typed' if low else 'mt_flash_fw',
             [P]*7+[I]*5+[ct.c_bool]+([I] if low else []),*(args+([dtype_code(q.dtype)] if low else [])))
        return out,lse,q,k,v,query_lengths,key_lengths

    @staticmethod
    def flash_attention_bw(grad,q,k,v,out,lse,query_lengths,key_lengths,causal):
        if not q.dtype == k.dtype == v.dtype == out.dtype:
            raise ValueError('FlashAttention saved dtype mismatch')
        grad = CudaResidentOps.cast(grad,'float32').contiguous()
        b,h,nq,d = q.shape
        nk = k.shape[2]
        dq,dk,dv = (CudaResidentOps.empty(t.shape,t.backend) for t in (q,k,v))
        workspace = getattr(q.backend, 'workspace', None)
        lease = (workspace.lease('flash_backward_delta', b*h*nq) if workspace is not None
                 else nullcontext(DeviceStorage(b*h*nq, allocator=getattr(q.backend, 'memory_pool', None))))
        with lease as delta:
            low = q.dtype != 'float32'
            args = [pointer(dq),pointer(dk),pointer(dv),delta.ptr,pointer(grad),
                    *(pointer(t,True) for t in (q,k,v,out)),pointer(lse),
                    pointer(query_lengths),pointer(key_lengths),b,h,nq,nk,d,causal]
            call(runtime().flash_libraries[1],'mt_flash_bw_typed' if low else 'mt_flash_bw',
                 [P]*12+[I]*5+[ct.c_bool]+([I] if low else []),*(args+([dtype_code(q.dtype)] if low else [])))
        return dq,dk,dv

    @staticmethod
    def empty(shape, backend, dtype='float32'):
        size = math.prod(shape)
        if size < 0 or size > 2**31-1 or any(d < 0 for d in shape):
            raise ValueError("Invalid/unsupported device tensor size")
        return Tensor.make(DeviceStorage(size, allocator=getattr(backend, 'memory_pool', None), dtype=dtype), tuple(shape), backend=backend)

    @staticmethod
    def full(shape, value, backend, dtype='float32'):
        out = CudaResidentOps.empty(shape, backend, dtype)
        out._tensor._storage.fill(value)
        return out

    @staticmethod
    def random(shape, backend, seed, dtype='float32'):
        out = CudaResidentOps.empty(shape, backend)
        r = runtime()
        r.check(r.lib.mt_random(pointer(out), out.size, seed))
        return CudaResidentOps.cast(out, dtype)

    @staticmethod
    def cast(a, dtype):
        """Raw cast used inside autograd Functions; no host transfer."""
        dtype = normalize_dtype(dtype)
        if a.dtype == dtype:
            return a
        out = CudaResidentOps.empty(a.shape, a.backend, dtype)
        return CudaResidentOps.map(operators.id)(a, out)

    @staticmethod
    def one_hot(ids, classes):
        if classes <= 0:
            raise ValueError("num_classes must be positive")
        ids = ids.contiguous()
        out = CudaResidentOps.empty(ids.shape+(classes,), ids.backend)
        r = runtime()
        r.check(r.lib.mt_one_hot(pointer(out),pointer(ids),out.size,classes))
        return out

    @staticmethod
    def sequence(shape, backend, width=0):
        out = CudaResidentOps.empty(shape, backend)
        r = runtime()
        r.check(r.lib.mt_sequence(pointer(out),out.size,width))
        return out

    @staticmethod
    def map(fn):
        def ret(a, out=None):
            if out is None:
                out = CudaResidentOps.empty(a.shape,a.backend,a.dtype)
            if out.backend is not a.backend:
                raise ValueError('Map requires one backend')
            if shape_broadcast(a.shape,out.shape) != out.shape:
                raise ValueError("Map output cannot broadcast input")
            if a.dtype == out.dtype == 'float32':
                call(runtime().combine,'mt_map',[P,Layout,P,Layout,I,I],
                     pointer(out),Layout.of(out),pointer(a),Layout.of(a),out.size,_IDS[fn])
            else:
                call(runtime().mixed,'mt_typed_map',[P,I,Layout,P,I,Layout,I,I],
                     pointer(out,True),dtype_code(out.dtype),Layout.of(out),
                     pointer(a,True),dtype_code(a.dtype),Layout.of(a),out.size,_IDS[fn])
            return out
        return ret

    @staticmethod
    def zip(fn):
        def ret(a,b):
            if a.backend is not b.backend:
                raise ValueError('Zip requires one backend')
            out = CudaResidentOps.empty(shape_broadcast(a.shape,b.shape),a.backend,promote_types(a.dtype,b.dtype))
            CudaResidentOps._zip_into(fn,out,a,b)
            return out
        return ret

    @staticmethod
    def _zip_into(fn,out,a,b):
        if a.dtype == b.dtype == out.dtype == 'float32':
            call(runtime().combine,'mt_zip',[P,Layout,P,Layout,P,Layout,I,I],
                 pointer(out),Layout.of(out),pointer(a),Layout.of(a),pointer(b),Layout.of(b),out.size,_IDS[fn])
        else:
            call(runtime().mixed,'mt_typed_zip',[P,I,Layout,P,I,Layout,P,I,Layout,I,I],
                 pointer(out,True),dtype_code(out.dtype),Layout.of(out),
                 pointer(a,True),dtype_code(a.dtype),Layout.of(a),
                 pointer(b,True),dtype_code(b.dtype),Layout.of(b),out.size,_IDS[fn])

    @staticmethod
    def accumulate_into(out, value):
        """Add a leaf gradient into its contiguous, persistent packed span."""
        if out.shape != value.shape or out.backend is not value.backend:
            raise ValueError('Packed gradient shape/backend mismatch')
        if not out._tensor.is_contiguous():
            raise ValueError('Packed gradient destination must be contiguous')
        # A permuted alias could otherwise race with this kernel's own writes.
        if out._tensor._storage.owner is value._tensor._storage.owner:
            value = value.clone()
        CudaResidentOps._zip_into(operators.add,out,out,value)

    @staticmethod
    def reduce(fn,start=0):
        identities = {operators.add: 0., operators.mul: 1., operators.max: -float('inf')}
        if fn not in identities:
            raise NotImplementedError("Only associative add/mul/max reductions are supported")
        def ret(a,dim):
            if not 0 <= dim < a.dims:
                raise ValueError("Invalid reduction dimension")
            shape = list(a.shape)
            shape[dim] = 1
            out = CudaResidentOps.empty(tuple(shape),a.backend)
            identity = identities[fn]
            if a.dtype == 'float32':
                call(runtime().combine,'mt_reduce',[P,Layout,P,Layout,I,I,F,I],
                     pointer(out),Layout.of(out),pointer(a),Layout.of(a),out.size,dim,identity,_IDS[fn])
            else:
                call(runtime().mixed,'mt_typed_reduce',[P,Layout,P,I,Layout,I,I,F,I],
                     pointer(out),Layout.of(out),pointer(a,True),dtype_code(a.dtype),Layout.of(a),
                     out.size,dim,identity,_IDS[fn])
            if start != identity:
                out = CudaResidentOps.zip(fn)(out,out._ensure_tensor(start))
            return out
        return ret

    @staticmethod
    def matrix_multiply(a,b,compute_dtype=None,output_dtype=None):
        if a.dims<2 or b.dims<2 or a.shape[-1]!=b.shape[-2]:
            raise ValueError("matmul needs (..., M, K) and (..., K, N)")
        if a.backend is not b.backend:
            raise ValueError('Matmul requires one backend')
        dtype = normalize_dtype(compute_dtype) if compute_dtype is not None else promote_types(a.dtype,b.dtype)
        output_dtype = normalize_dtype(output_dtype) if output_dtype is not None else dtype
        if output_dtype not in (dtype,'float32'):
            raise ValueError('GEMM output must match compute dtype or be float32')
        r = runtime()
        if dtype != 'float32':
            supported = r.mixed.mt_low_gemm_supported
            supported.argtypes, supported.restype = [I], I
            if not supported(dtype_code(dtype)):
                raise RuntimeError(f'{dtype} GEMM requires SM70+ (FP16) or SM80+ (BF16)')
        a,b = (CudaResidentOps.cast(t,dtype).contiguous() for t in (a,b))
        batch = shape_broadcast(a.shape[:-2],b.shape[:-2])
        m,k,n = a.shape[-2],a.shape[-1],b.shape[-1]
        out = CudaResidentOps.empty(batch+(m,n),a.backend,output_dtype)
        if not out.size:
            return out
        if k == 0:
            out._tensor._storage.fill(0)
            return out
        def offset(t,index):
            shift = len(batch)-(t.dims-2)
            # Permuted views can carry NumPy integer strides; ctypes pointers
            # require a Python int even when the view is already contiguous.
            return int(sum((0 if size==1 else index[j+shift])*stride
                       for j,(size,stride) in enumerate(zip(t.shape[:-2],t._tensor.strides[:-2]))))
        for ordinal,index in enumerate(np.ndindex(batch)):
            args = (r.blas,pointer(out,True)+ordinal*m*n*itemsize(out.dtype),
                    pointer(a,True)+offset(a,index)*itemsize(a.dtype),
                    pointer(b,True)+offset(b,index)*itemsize(b.dtype),m,n,k)
            if dtype == 'float32':
                r.check_blas(r.lib.mt_gemm(*args))
            else:
                f = r.mixed.mt_typed_gemm
                f.argtypes, f.restype = [P]*4+[I]*5,I
                r.check_blas(f(*args,dtype_code(dtype),dtype_code(out.dtype)))
        return out

    @staticmethod
    def attn_softmax_fw(inp,mask):
        if inp.dims!=4 or mask.shape!=(inp.shape[0],inp.shape[-1]):
            raise ValueError("softmax expects [B,H,Q,K] and a [B,K] additive mask")
        inp,mask = inp.contiguous(),mask.contiguous()
        if inp.dtype != 'float32':
            out = CudaResidentOps.empty(inp.shape,inp.backend,inp.dtype)
            b,h,q,w = inp.shape
            call(runtime().amp,'mt_amp_softmax_fw',[P]*3+[I]*5,
                 pointer(out,True),pointer(inp,True),pointer(mask),dtype_code(inp.dtype),b*h*q,h,q,w)
            return out
        out = CudaResidentOps.empty(inp.shape,inp.backend)
        call(runtime().softmax,'mt_softmax_fw',[P,P,P,I,I,I,I,ct.c_bool],
             pointer(out),pointer(inp),pointer(mask),*inp.shape,False)
        runtime().stats['d2d_bytes'] += out.size*4
        return out

    @staticmethod
    def attn_softmax_bw(grad,output):
        if grad.shape!=output.shape:
            raise ValueError("softmax gradient shape mismatch")
        grad,output = CudaResidentOps.cast(grad,'float32').contiguous(),output.contiguous()
        out = CudaResidentOps.empty(grad.shape,grad.backend)
        width = grad.shape[-1]
        if output.dtype != 'float32':
            call(runtime().amp,'mt_amp_softmax_bw',[P]*3+[I]*3,
                 pointer(out),pointer(grad),pointer(output,True),dtype_code(output.dtype),grad.size//width,width)
            return out
        call(runtime().softmax,'mt_softmax_bw',[P,P,P,I,I],
             pointer(out),pointer(grad),pointer(output),grad.size//width,width)
        runtime().stats['d2d_bytes'] += out.size*4
        return out

    @staticmethod
    def layernorm_fw(inp,gamma,beta):
        width = inp.shape[-1]
        if gamma.shape!=(width,) or beta.shape!=(width,):
            raise ValueError("LayerNorm affine shape mismatch")
        inp,gamma,beta = (t.contiguous() for t in (inp,gamma,beta))
        rows = inp.size//width
        out = CudaResidentOps.empty(inp.shape,inp.backend,inp.dtype)
        var = CudaResidentOps.empty((rows,),inp.backend)
        mean = CudaResidentOps.empty((rows,),inp.backend)
        if inp.dtype != 'float32':
            call(runtime().amp,'mt_amp_ln_fw',[P]*6+[I]*3,
                 pointer(out,True),pointer(var),pointer(mean),pointer(inp,True),pointer(gamma),pointer(beta),
                 dtype_code(inp.dtype),rows,width)
            return out,var,mean
        call(runtime().layernorm,'mt_layernorm_fw',[P]*6+[I,I],
             *(pointer(t) for t in (out,var,mean,inp,gamma,beta)),rows,width)
        return out,var,mean

    @staticmethod
    def layernorm_bw(grad,inp,gamma,beta,var,mean):
        grad = CudaResidentOps.cast(grad,'float32')
        grad,inp,gamma,beta,var,mean = (t.contiguous() for t in (grad,inp,gamma,beta,var,mean))
        dx,dg,db = (CudaResidentOps.empty(t.shape,t.backend) for t in (inp,gamma,beta))
        width=inp.shape[-1]
        if inp.dtype != 'float32':
            call(runtime().amp,'mt_amp_ln_bw',[P]*8+[I]*3,
                 pointer(dx),pointer(dg),pointer(db),pointer(grad),pointer(inp,True),pointer(gamma),pointer(var),pointer(mean),
                 dtype_code(inp.dtype),inp.size//width,width)
            return dx,dg,db
        call(runtime().layernorm,'mt_layernorm_bw',[P]*9+[I,I],
             *(pointer(t) for t in (dg,db,dx,grad,inp,gamma,beta,var,mean)),inp.size//width,width)
        return dx,dg,db
