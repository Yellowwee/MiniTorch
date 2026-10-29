"""Opt-in packed FP32 Adam for the single-stream resident CUDA backend.

Construct before the first forward. Four persistent flat buffers hold weights,
gradients and moments. Parameter identity and shape are preserved; storage is
rebound once. Each step launches one update kernel, with a small metadata upload.
"""
import ctypes as ct
import math
from pathlib import Path
import weakref
import numpy as np

from .optim import Optimizer
from .tensor import Tensor
from .cuda_storage import DeviceStorage, runtime


class FusedAdam(Optimizer):
    def __init__(self, parameters, lr=1e-3, beta1=.9, beta2=.999, eps=1e-8):
        super().__init__(list(parameters))
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self._validate_hyperparameters()
        self.entries = []
        seen = {}
        backend = None
        # Validate everything before allocating/rebinding model weights.
        for p in self.parameters:
            t = p.value
            if t is None:
                continue
            if not isinstance(t, Tensor) or not t.backend.device_storage or not t.is_leaf() or t.dtype != 'float32':
                raise ValueError('FusedAdam requires leaf FP32 resident CUDA tensors')
            if backend is not None and t.backend is not backend:
                raise ValueError('FusedAdam parameters must use one backend')
            backend = t.backend
            previous = getattr(t, '_packed_optimizer', None)
            if previous is not None and previous() is not None:
                raise ValueError('Parameter already belongs to a packed optimizer')
            if t.grad is not None:
                raise ValueError('Construct FusedAdam before backward (existing gradients would be lost)')
            if id(t) in seen:
                seen[id(t)]['aliases'].append(p)
                continue
            storage = t._tensor._storage
            for entry in self.entries:
                other = entry['tensor']._tensor._storage
                if (storage.owner is other.owner and
                    max(storage.offset, other.offset) < min(storage.offset+len(storage), other.offset+len(other))):
                    raise ValueError('Distinct parameter tensors with overlapping storage are unsupported; tie the same Tensor object')
            entry = dict(tensor=t, aliases=[p], shape=t.shape, step=0)
            self.entries.append(entry)
            seen[id(t)] = entry
        if backend is None:
            raise ValueError('FusedAdam requires at least one tensor parameter')
        self.backend = backend
        self.runtime = runtime()
        try:
            self.library = ct.CDLL(str(Path(__file__).resolve().parent/'cuda_kernels'/'fused_adam.so'))
        except OSError as error:
            raise RuntimeError('Build fused_adam.so with compile_cuda.sh before using FusedAdam') from error
        self.kernel = self.library.mt_packed_adam
        self.kernel.argtypes = [ct.c_void_p]*6+[ct.c_int]+[ct.c_float]*5
        self.kernel.restype = ct.c_int
        self.size = sum(e['tensor'].size for e in self.entries)
        if self.size > 2**31-1:
            raise ValueError('Packed Adam supports at most int32 elements')
        allocator = getattr(backend, 'memory_pool', None)
        # Persistent arrays never need per-step pooling. Exact-size allocations
        # avoid rounding four model-sized arrays up to power-of-two buckets.
        self.weights, self.gradients, self.exp_avg, self.exp_avg_sq = [
            DeviceStorage(self.size) for _ in range(4)]
        self.exp_avg.fill(0); self.exp_avg_sq.fill(0)
        tiles = []
        offset = 0
        for index, entry in enumerate(self.entries):
            t = entry['tensor']
            end = offset+t.size
            entry['offset'] = offset
            entry['packed'] = Tensor.make(self.weights[offset:end], t.shape, backend=backend)
            entry['gradient'] = Tensor.make(self.gradients[offset:end], t.shape, backend=backend)
            backend.id_map(t, entry['packed'])
            tiles.extend((start, min(256, end-start), index) for start in range(offset, end, 256))
            offset = end
        tile_array = np.asarray(tiles, dtype=np.int32).reshape(-1, 3)
        # DeviceStorage supplies owned bytes; this private buffer contains int32 metadata.
        self.tiles = DeviceStorage(tile_array.size, allocator=allocator)
        self._upload(self.tiles, tile_array)
        self.tile_count = len(tiles)
        self.step_sizes = DeviceStorage(len(self.entries), allocator=allocator)
        self._host_steps = np.full(len(self.entries), -1, dtype=np.float32)
        self.kernel_launches = 0
        self.closed = False
        for entry in self.entries:
            t = entry['tensor']
            # Same Tensor object: shared/tied Parameter containers remain shared.
            t._tensor = entry['packed']._tensor
            t._grad_buffer = entry['gradient']
            t._packed_optimizer = weakref.ref(self)

    def _validate_hyperparameters(self):
        if not all(math.isfinite(x) for x in (self.lr, self.beta1, self.beta2, self.eps)):
            raise ValueError('Adam hyperparameters must be finite')
        if self.lr < 0 or self.eps <= 0 or not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError('Adam requires lr>=0, eps>0 and betas in [0,1)')

    def _upload(self, storage, array):
        if array.nbytes:
            self.runtime.check(self.runtime.lib.mt_copy(storage.ptr, array.ctypes.data, array.nbytes, 1))
            self.runtime.stats['h2d_bytes'] += array.nbytes

    def _validate_bindings(self):
        if self.closed:
            raise RuntimeError('FusedAdam is closed')
        for entry in self.entries:
            t = entry['tensor']
            if (any(p.value is not t for p in entry['aliases']) or
                t._tensor is not entry['packed']._tensor or not t.is_leaf()):
                raise RuntimeError('Packed parameter binding changed; reconstruct the optimizer before a new forward')
            if t.grad is not None and (t.grad.shape != t.shape or t.grad.backend is not self.backend):
                raise ValueError('Parameter gradient shape/backend mismatch')

    def zero_grad(self):
        self._validate_bindings()
        # First contribution overwrites the slot. None must remain distinguishable
        # from a computed zero gradient so unused parameters do not advance Adam.
        for entry in self.entries:
            entry['tensor'].grad = None

    def step(self):
        self._validate_hyperparameters()
        self._validate_bindings()
        self._host_steps.fill(-1)
        active = []
        for index, entry in enumerate(self.entries):
            t = entry['tensor']
            if t.grad is None:
                continue
            if t.grad is not entry['gradient']:
                self.backend.id_map(t.grad, entry['gradient'])
            step = entry['step']+1
            self._host_steps[index] = self.lr*math.sqrt(1-self.beta2**step)/(1-self.beta1**step)
            active.append(entry)
        if not active:
            return
        self._upload(self.step_sizes, self._host_steps)
        self.runtime.check(self.kernel(
            self.weights.ptr, self.gradients.ptr, self.exp_avg.ptr, self.exp_avg_sq.ptr,
            self.tiles.ptr, self.step_sizes.ptr, self.tile_count,
            self.beta1, self.beta2, 1-self.beta1, 1-self.beta2, self.eps))
        self.kernel_launches += int(self.tile_count > 0)
        self.weights.owner.version += 1
        for entry in active:
            entry['step'] += 1

    def memory_metrics(self):
        return dict(packed_parameter_bytes=self.size*4, packed_gradient_bytes=self.size*4,
                    packed_state_bytes=self.size*8, packed_metadata_bytes=len(self.tiles)*4+len(self.step_sizes)*4,
                    update_kernel_launches=self.kernel_launches)

    def close(self):
        """Unbind gradient sinks; weights/visible gradients keep their own owners."""
        if self.closed:
            return
        for entry in self.entries:
            t = entry['tensor']
            t._grad_buffer = None
            t._packed_optimizer = None
        self.closed = True
        self.entries.clear()
        self.weights = self.gradients = self.exp_avg = self.exp_avg_sq = None
        self.tiles = self.step_sizes = None
