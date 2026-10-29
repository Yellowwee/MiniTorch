"""Scoped low-precision compute and dynamic scaling for FP32 master weights.

The scope inserts differentiable casts at matmul and fused-op boundaries. Backward
uses saved operand dtypes, independent of the scope active during backward.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from .dtypes import normalize_dtype
import math
import ctypes as ct

_compute_dtype = ContextVar('minitorch_matmul_dtype', default=None)


def autocast_dtype():
    return _compute_dtype.get()


@contextmanager
def autocast(dtype='bfloat16', enabled=True):
    dtype = normalize_dtype(dtype)
    if enabled and dtype == 'float32':
        raise ValueError('autocast dtype must be float16 or bfloat16')
    token = _compute_dtype.set(dtype if enabled else None)
    try:
        yield
    finally:
        _compute_dtype.reset(token)


class GradScaler:
    """Single-optimizer-per-cycle scaler for resident FP32 master parameters.

    Accumulate scaled microbatch gradients, unscale once, optionally clip, then
    step and update. Nonfinite detection skips the entire optimizer call. With
    enabled=False, the scale stays one but finite-gradient checking still runs.
    """
    def __init__(self, enabled=True, init_scale=65536., growth_factor=2.,
                 backoff_factor=.5, growth_interval=2000, min_scale=1., max_scale=2.**24):
        values=(init_scale,growth_factor,backoff_factor,min_scale,max_scale)
        if not all(math.isfinite(x) for x in values):
            raise ValueError('Scaler settings must be finite')
        if not (0 < min_scale <= init_scale <= max_scale <= 2.**120 and min_scale >= 2.**-120):
            raise ValueError('Require 2^-120 <= min_scale <= init_scale <= max_scale <= 2^120')
        if growth_factor<=1 or not 0<backoff_factor<1:
            raise ValueError('Require growth_factor>1 and 0<backoff_factor<1')
        if isinstance(growth_interval,bool) or not isinstance(growth_interval,int) or growth_interval<1:
            raise ValueError('growth_interval must be a positive integer')
        self.enabled=bool(enabled)
        self._scale=float(init_scale) if self.enabled else 1.
        self.growth_factor=float(growth_factor)
        self.backoff_factor=float(backoff_factor)
        self.growth_interval=growth_interval
        self.min_scale=float(min_scale); self.max_scale=float(max_scale)
        self._growth_tracker=0
        self.skipped_steps=0; self.successful_steps=0
        self._optimizer=None; self._stage='ready'; self._finite=None

    def get_scale(self):
        return self._scale

    def scale(self, loss):
        if self._stage!='ready':
            raise RuntimeError('Call update() before scaling the next loss; accumulate before unscale_()')
        if not loss.backend.device_storage or loss.size!=1:
            raise ValueError('GradScaler requires a scalar loss on the resident backend')
        return loss.to(dtype='float32') * self._scale

    @staticmethod
    def _gradients(optimizer):
        from .tensor import Tensor
        grads=[]; seen={}; spans=[]; backend=None
        for parameter in optimizer.parameters:
            t=parameter.value
            if t is None: continue
            if not isinstance(t,Tensor) or not t.backend.device_storage or t.dtype!='float32' or not t.is_leaf():
                raise ValueError('GradScaler requires leaf FP32 resident master parameters')
            if backend is not None and t.backend is not backend:
                raise ValueError('GradScaler requires one resident backend')
            backend=t.backend
            g=t.grad
            if g is None: continue
            if id(g) in seen:
                if seen[id(g)] is not t:
                    raise ValueError('Distinct parameters must not share a gradient object')
                continue
            if g.backend is not backend or g.dtype!='float32' or g.shape!=t.shape:
                raise ValueError('Master gradients must be FP32 with matching shape/backend')
            seen[id(g)]=t
            storage=g._tensor._storage
            start,end=storage.ptr,storage.ptr+storage.nbytes
            if any(start<b and a<end for a,b in spans):
                raise ValueError('Overlapping gradient views cannot be unscaled twice')
            spans.append((start,end)); grads.append((t,g))
        if not grads:
            raise ValueError('No gradients to unscale; run backward before step')
        return grads

    def unscale_(self, optimizer):
        if self._stage!='ready':
            raise RuntimeError('unscale_() may run only once per update cycle')
        if hasattr(optimizer,'_validate_bindings'):
            optimizer._validate_bindings()
        gradients=self._gradients(optimizer) # Validate all before any mutation.
        from .cuda_storage import DeviceStorage, runtime
        r=runtime()
        flag=DeviceStorage(1,allocator=getattr(gradients[0][0].backend,'memory_pool',None))
        flag.fill(0)
        f=r.amp.mt_unscale
        f.argtypes,f.restype=[ct.c_void_p,ct.c_int,ct.c_float,ct.c_void_p],ct.c_int
        for t,g in gradients:
            g=g.detach().contiguous()
            r.check(f(g._tensor._storage.ptr,g.size,1./self._scale,flag.ptr))
            # Preserve the packed optimizer's persistent gradient span when contiguous.
            if t.grad._tensor is not g._tensor:
                t.grad=g
        found=ct.c_int()
        r.check(r.lib.mt_copy(ct.byref(found),flag.ptr,4,2))
        r.stats['d2h_bytes']+=4
        self._optimizer=optimizer; self._stage='unscaled'; self._finite=(found.value==0)
        return self._finite

    def step(self, optimizer):
        if self._stage=='ready':
            self.unscale_(optimizer)
        if self._optimizer is not optimizer or self._stage!='unscaled':
            raise RuntimeError('Use one optimizer and one step per update cycle')
        if self._finite:
            optimizer.step()
            self.successful_steps+=1
        else:
            self.skipped_steps+=1
        self._stage='stepped'
        return self._finite

    def update(self):
        if self._stage!='stepped':
            raise RuntimeError('Call step() before update()')
        if self.enabled:
            if not self._finite:
                self._scale=max(self.min_scale,self._scale*self.backoff_factor)
                self._growth_tracker=0
            else:
                self._growth_tracker+=1
                if self._growth_tracker>=self.growth_interval:
                    self._scale=min(self.max_scale,self._scale*self.growth_factor)
                    self._growth_tracker=0
        self._optimizer=None; self._stage='ready'; self._finite=None

    def state_dict(self):
        if self._stage!='ready':
            raise RuntimeError('Save scaler state after update()')
        return dict(enabled=self.enabled,init_scale=self._scale,growth_factor=self.growth_factor,
                    backoff_factor=self.backoff_factor,growth_interval=self.growth_interval,
                    min_scale=self.min_scale,max_scale=self.max_scale,growth_tracker=self._growth_tracker,
                    skipped_steps=self.skipped_steps,successful_steps=self.successful_steps)

    def load_state_dict(self,state):
        if self._stage!='ready':
            raise RuntimeError('Restore scaler state between update cycles')
        state=dict(state)
        counts={k:state.pop(k) for k in ('growth_tracker','skipped_steps','successful_steps')}
        if any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in counts.values()):
            raise ValueError('Scaler counters must be nonnegative integers')
        # Disabled scaling can have a configured min_scale above its fixed scale 1.
        enabled=state['enabled']
        scale=state['init_scale']
        if not enabled: state['init_scale']=min(state['max_scale'],max(state['min_scale'],1.))
        restored=GradScaler(**state)
        if counts['growth_tracker']>=restored.growth_interval or (not enabled and scale!=1.):
            raise ValueError('Invalid scaler state')
        restored._growth_tracker=counts['growth_tracker']
        restored.skipped_steps=counts['skipped_steps']; restored.successful_steps=counts['successful_steps']
        self.__dict__.update(restored.__dict__)
