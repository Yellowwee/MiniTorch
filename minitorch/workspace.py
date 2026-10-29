"""Named, grow-only scratch slots for explicitly scoped native launchers.

Saved activations and output tensors never occupy these slots. A lease ends
after all uses have been enqueued on the resident backend's default stream.
Its contents may be overwritten by the next lease of the same name: do not
retain scratch data for later consumers or save it in an autograd Context.
"""
from contextlib import contextmanager
from .cuda_storage import DeviceStorage
from .dtypes import normalize_dtype


class Workspace:
    def __init__(self, pool):
        self.pool = pool
        self.slots = {}
        self.busy = set()

    @contextmanager
    def lease(self, name, elements, dtype='float32'):
        if name in self.busy:
            raise RuntimeError(f'Workspace slot {name!r} is already in use')
        elements = int(elements)
        dtype = normalize_dtype(dtype)
        if elements < 0:
            raise ValueError('Workspace size must be nonnegative')
        current = self.slots.get(name)
        if current is None or len(current) < elements or current.dtype != dtype:
            current = DeviceStorage(elements, allocator=self.pool, dtype=dtype)
            self.slots[name] = current
        self.busy.add(name)
        try:
            yield current[:elements]
        finally:
            self.busy.remove(name)

    @property
    def nbytes(self):
        return sum(s.owner.nbytes for s in self.slots.values())

    def clear(self):
        if self.busy:
            raise RuntimeError('Cannot clear workspace with active leases')
        self.slots.clear()
