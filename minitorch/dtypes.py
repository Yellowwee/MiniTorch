"""Resident floating-point types and portable BF16 host transport.

BF16 has no dependency on a NumPy extension: transport uses uint16 bit patterns,
while public NumPy exports decode to float32. Host backends remain FP32.
"""
import numpy as np

FLOAT32 = 'float32'
FLOAT16 = 'float16'
BFLOAT16 = 'bfloat16'
_CODES = {FLOAT32: 0, FLOAT16: 1, BFLOAT16: 2}


def normalize_dtype(dtype):
    if isinstance(dtype, str):
        name = dtype
    else:
        name = np.dtype(dtype).name
    if name not in _CODES:
        raise ValueError(f'Unsupported dtype {dtype!r}; use float32, float16 or bfloat16')
    return name


def dtype_code(dtype):
    return _CODES[normalize_dtype(dtype)]


def itemsize(dtype):
    return 4 if normalize_dtype(dtype) == FLOAT32 else 2


def promote_types(a, b):
    a, b = normalize_dtype(a), normalize_dtype(b)
    return a if a == b else FLOAT32


def encode_host(value, dtype):
    dtype = normalize_dtype(dtype)
    if dtype != BFLOAT16:
        return np.ascontiguousarray(value, dtype=dtype).reshape(-1)
    bits = np.ascontiguousarray(value, dtype=np.float32).reshape(-1).view(np.uint32)
    # Round-to-nearest, ties-to-even; keep NaNs as NaNs rather than infinity.
    rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    nan = ((bits & 0x7f800000) == 0x7f800000) & ((bits & 0x007fffff) != 0)
    return np.where(nan, (bits >> 16).astype(np.uint16) | np.uint16(0x40), rounded).astype(np.uint16)


def decode_host(value, dtype):
    if normalize_dtype(dtype) == BFLOAT16:
        return (value.astype(np.uint32) << 16).view(np.float32)
    return value


def validate_backend_dtype(backend, dtype):
    dtype = normalize_dtype(dtype)
    if dtype != FLOAT32 and not backend.device_storage:
        raise ValueError('Low precision requires CudaResidentOps (device storage)')
    return dtype
