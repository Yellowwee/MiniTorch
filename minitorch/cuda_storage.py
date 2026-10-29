"""Lazy CUDA runtime, shared ownership, explicit transfers and allocation metrics.

The resident backend uses device 0 and the legacy default stream only. Allocations
are released by their last owner, either to CUDA or an optional memory pool.
External streams and multi-device execution are unsupported.
"""
import ctypes as ct
from pathlib import Path
import weakref
import numpy as np
from .dtypes import normalize_dtype, itemsize, dtype_code, encode_host, decode_host


class CudaRuntime:
    def __init__(self):
        root = Path(__file__).resolve().parent / "cuda_kernels"
        self.lib = ct.CDLL(str(root / "cuda_runtime.so"))
        self.combine = ct.CDLL(str(root / "combine.so"))
        self.softmax = ct.CDLL(str(root / "softmax_kernel.so"))
        self.layernorm = ct.CDLL(str(root / "layernorm_kernel.so"))
        self._flash_libraries = None
        self._mixed_library = None
        self._amp_library = None
        self.stats = dict(active_bytes=0, peak_active_bytes=0, reserved_bytes=0,
                          peak_reserved_bytes=0, allocations=0, frees=0,
                          h2d_bytes=0, d2h_bytes=0, d2d_bytes=0,
                          cached_bytes=0, pool_hits=0, pool_misses=0, oom_retries=0)
        p, i, z = ct.c_void_p, ct.c_int, ct.c_size_t
        signatures = {
            "mt_malloc": [ct.POINTER(p), z], "mt_free": [p],
            "mt_copy": [p,p,z,i], "mt_sync": [], "mt_fill": [p,i,ct.c_float],
            "mt_random": [p,i,ct.c_ulonglong], "mt_one_hot": [p,p,i,i],
            "mt_sequence": [p,i,i], "mt_blas_create": [ct.POINTER(p)],
            "mt_blas_destroy": [p], "mt_gemm": [p,p,p,p,i,i,i],
            "mt_event_create": [ct.POINTER(p)], "mt_event_record": [p],
            "mt_event_elapsed": [ct.POINTER(ct.c_float),p,p], "mt_event_destroy": [p],
        }
        for name, args in signatures.items():
            f = getattr(self.lib,name)
            f.argtypes, f.restype = args, i
        self.lib.mt_error.argtypes = [i]
        self.lib.mt_error.restype = ct.c_char_p
        self.blas = p()
        self.check_blas(self.lib.mt_blas_create(ct.byref(self.blas)))
        weakref.finalize(self, self.lib.mt_blas_destroy, self.blas)

    def check(self, status):
        if status:
            raise RuntimeError(f"CUDA: {self.lib.mt_error(status).decode()}")

    @property
    def amp(self):
        if self._amp_library is None:
            path = Path(__file__).resolve().parent / 'cuda_kernels' / 'amp_kernels.so'
            try:
                self._amp_library = ct.CDLL(str(path))
            except OSError as error:
                raise RuntimeError('Build amp_kernels.so using compile_cuda.sh') from error
        return self._amp_library

    @property
    def mixed(self):
        if self._mixed_library is None:
            path = Path(__file__).resolve().parent / 'cuda_kernels' / 'mixed_precision.so'
            try:
                self._mixed_library = ct.CDLL(str(path))
            except OSError as error:
                raise RuntimeError('Build mixed_precision.so using compile_cuda.sh') from error
        return self._mixed_library

    @property
    def flash_libraries(self):
        if self._flash_libraries is None:
            root = Path(__file__).resolve().parent / "cuda_kernels"
            self._flash_libraries = tuple(ct.CDLL(str(root / name)) for name in (
                "flash_attention_kernel.so", "flash_attention_backward.so"))
        return self._flash_libraries

    @staticmethod
    def check_blas(status):
        if status:
            raise RuntimeError(f"cuBLAS status {status}")

    def synchronize(self):
        self.check(self.lib.mt_sync())

    def snapshot(self):
        return dict(self.stats)

    def reset_peaks(self):
        self.stats['peak_active_bytes'] = self.stats['active_bytes']
        self.stats['peak_reserved_bytes'] = self.stats['reserved_bytes']

    def _free(self, pointer, nbytes):
        status = self.lib.mt_free(pointer)
        if status == 0:
            self.stats['active_bytes'] -= nbytes
            self.stats['reserved_bytes'] -= nbytes
            self.stats['frees'] += 1


_runtime = None


def runtime():
    global _runtime
    if _runtime is None:
        _runtime = CudaRuntime()
    return _runtime


class RawDeviceBuffer:
    """Byte allocation for integer/pointer metadata and non-Tensor page slabs.

    Exact byte accounting, default stream only. Never casts metadata to FP32.
    Explicit close and GC finalization use the same idempotent release path.
    """
    def __init__(self, nbytes):
        if type(nbytes) is not int or nbytes < 1:
            raise ValueError('RawDeviceBuffer requires positive byte size')
        self.runtime = runtime()
        self.nbytes = nbytes
        self.pointer = ct.c_void_p()
        self.runtime.check(self.runtime.lib.mt_malloc(ct.byref(self.pointer),nbytes))
        s = self.runtime.stats
        s['allocations'] += 1
        s['active_bytes'] += nbytes
        s['reserved_bytes'] += nbytes
        s['peak_active_bytes'] = max(s['peak_active_bytes'],s['active_bytes'])
        s['peak_reserved_bytes'] = max(s['peak_reserved_bytes'],s['reserved_bytes'])
        self._release = weakref.finalize(self,self.runtime._free,self.pointer,nbytes)

    @property
    def ptr(self):
        if not self._release.alive:
            raise ValueError('RawDeviceBuffer is closed')
        return self.pointer.value

    def upload(self, array, byte_offset=0):
        a = np.ascontiguousarray(array)
        if a.dtype.hasobject or type(byte_offset) is not int or byte_offset < 0 or byte_offset+a.nbytes > self.nbytes:
            raise ValueError('Invalid raw metadata upload')
        self.runtime.check(self.runtime.lib.mt_copy(self.ptr+byte_offset,a.ctypes.data,a.nbytes,1))
        self.runtime.stats['h2d_bytes'] += a.nbytes

    def close(self):
        self._release()


class _Allocation:
    def __init__(self, size, allocator=None, dtype='float32'):
        self.runtime = runtime()
        self.capacity = int(size)
        self.dtype = normalize_dtype(dtype)
        self.version = 0  # Incremented by packed optimizer in-place updates.
        self.pointer = ct.c_void_p()
        # CUDA may return nullptr for zero bytes; retain a valid allocation.
        nbytes = max(1, self.capacity) * itemsize(self.dtype)
        if allocator is not None:
            self.pointer, self.nbytes = allocator.acquire(nbytes)
            weakref.finalize(self, allocator.release, self.pointer, self.nbytes)
            return
        self.nbytes = nbytes
        self.runtime.check(self.runtime.lib.mt_malloc(ct.byref(self.pointer), nbytes))
        s = self.runtime.stats
        s['allocations'] += 1
        s['active_bytes'] += nbytes
        s['reserved_bytes'] += nbytes
        s['peak_active_bytes'] = max(s['peak_active_bytes'], s['active_bytes'])
        s['peak_reserved_bytes'] = max(s['peak_reserved_bytes'], s['reserved_bytes'])
        weakref.finalize(self, self.runtime._free, self.pointer, nbytes)


class DeviceStorage:
    """A typed span; offsets are elements and views cannot reinterpret dtype."""
    is_device_storage = True

    def __init__(self, size, owner=None, offset=0, allocator=None, dtype=None):
        if int(size) < 0 or int(offset) < 0:
            raise ValueError('Device storage size/offset must be nonnegative')
        self.dtype = normalize_dtype(dtype if dtype is not None else (owner.dtype if owner is not None else 'float32'))
        if owner is not None and self.dtype != owner.dtype:
            raise ValueError('Storage views cannot reinterpret dtype; use Tensor.to(dtype=...)')
        self.owner = owner if owner is not None else _Allocation(size, allocator, self.dtype)
        self.offset, self.size = int(offset), int(size)
        if self.offset < 0 or self.size < 0 or self.offset+self.size > self.owner.capacity:
            raise ValueError("Device storage view exceeds allocation")

    @property
    def ptr(self):
        return self.owner.pointer.value + self.offset * self.itemsize

    @property
    def itemsize(self):
        return itemsize(self.dtype)

    @property
    def nbytes(self):
        return self.size * self.itemsize

    def __len__(self):
        return self.size

    @classmethod
    def from_numpy(cls, value, allocator=None, dtype='float32'):
        a = encode_host(value, dtype)
        result = cls(a.size, allocator=allocator, dtype=dtype)
        r = runtime()
        r.check(r.lib.mt_copy(result.ptr, a.ctypes.data, a.nbytes, 1))
        r.stats['h2d_bytes'] += a.nbytes
        return result

    def to_numpy(self):
        out = np.empty(self.size, dtype='uint16' if self.dtype == 'bfloat16' else self.dtype)
        r = runtime()
        r.check(r.lib.mt_copy(out.ctypes.data, self.ptr, out.nbytes, 2))
        r.stats['d2h_bytes'] += out.nbytes
        return decode_host(out, self.dtype)

    def fill(self, value):
        r = runtime()
        if self.dtype == 'float32':
            r.check(r.lib.mt_fill(self.ptr, self.size, float(value)))
        else:
            f = r.mixed.mt_typed_fill
            f.argtypes, f.restype = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_float], ct.c_int
            r.check(f(self.ptr, dtype_code(self.dtype), self.size, float(value)))

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self.size)
            if step != 1:
                raise ValueError("DeviceStorage supports contiguous spans only")
            return DeviceStorage(max(0,stop-start), self.owner, self.offset+start)
        index = int(index)
        if not 0 <= index < self.size:
            raise IndexError(index)
        return float(self[index:index+1].to_numpy()[0])

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            view = self[index]
            if np.isscalar(value):
                view.fill(value)
                return
        else:
            if not 0 <= int(index) < self.size:
                raise IndexError(index)
            view = self[int(index):int(index)+1]
        a = encode_host(value, self.dtype)
        if len(a) != len(view):
            raise ValueError("Storage assignment size mismatch")
        r = runtime()
        r.check(r.lib.mt_copy(view.ptr, a.ctypes.data, a.nbytes, 1))
        r.stats['h2d_bytes'] += a.nbytes


class CudaEvent:
    def __init__(self):
        self.runtime = runtime()
        self.handle = ct.c_void_p()
        self.runtime.check(self.runtime.lib.mt_event_create(ct.byref(self.handle)))
        weakref.finalize(self, self.runtime.lib.mt_event_destroy, self.handle)

    def record(self):
        self.runtime.check(self.runtime.lib.mt_event_record(self.handle))

    def seconds_since(self, start):
        ms = ct.c_float()
        self.runtime.check(self.runtime.lib.mt_event_elapsed(ct.byref(ms),start.handle,self.handle))
        return ms.value / 1000
