"""Bounded byte-based caching allocator for the resident default stream.

Only allocations released by their last owner enter the free lists. Reuse is
safe without a host synchronize because all consumers and subsequent writers
are enqueued on the same CUDA stream. External streams/threads are unsupported.
"""
import ctypes as ct
import weakref


def _clear_bins(runtime, bins):
    for capacity in list(bins):
        blocks = bins[capacity]
        while blocks:
            pointer = blocks[-1]
            runtime.check(runtime.lib.mt_free(pointer))
            blocks.pop()
            runtime.stats['reserved_bytes'] -= capacity
            runtime.stats['cached_bytes'] -= capacity
            runtime.stats['frees'] += 1
        del bins[capacity]


class MemoryPool:
    """Power-of-two capacity buckets, >=256-byte alignment from cudaMalloc.

    cache_limit_bytes bounds *idle* memory, not live allocations. No splitting
    or shared slab means a view can never outlive the allocation it references.
    """
    def __init__(self, runtime, cache_limit_bytes=1024**3):
        if cache_limit_bytes < 0:
            raise ValueError('cache limit must be nonnegative')
        self.runtime = runtime
        self.cache_limit_bytes = int(cache_limit_bytes)
        self.bins = {}
        self.cached_bytes = 0
        weakref.finalize(self, _clear_bins, runtime, self.bins)

    @staticmethod
    def bucket(nbytes):
        return 1 << (max(256, int(nbytes))-1).bit_length()

    def acquire(self, nbytes):
        capacity = self.bucket(nbytes)
        blocks = self.bins.get(capacity)
        s = self.runtime.stats
        if blocks:
            pointer = blocks.pop()
            self.cached_bytes -= capacity
            s['cached_bytes'] -= capacity
            s['pool_hits'] += 1
        else:
            s['pool_misses'] += 1
            pointer = ct.c_void_p()
            status = self.runtime.lib.mt_malloc(ct.byref(pointer), capacity)
            if status == 2:  # cudaErrorMemoryAllocation: reclaim idle blocks once.
                self.empty_cache()
                s['oom_retries'] += 1
                status = self.runtime.lib.mt_malloc(ct.byref(pointer), capacity)
            self.runtime.check(status)
            s['allocations'] += 1
            s['reserved_bytes'] += capacity
        s['active_bytes'] += capacity
        s['peak_active_bytes'] = max(s['peak_active_bytes'], s['active_bytes'])
        s['peak_reserved_bytes'] = max(s['peak_reserved_bytes'], s['reserved_bytes'])
        return pointer, capacity

    def release(self, pointer, capacity):
        s = self.runtime.stats
        if self.cached_bytes + capacity <= self.cache_limit_bytes:
            self.bins.setdefault(capacity, []).append(pointer)
            self.cached_bytes += capacity
            s['cached_bytes'] += capacity
        else:
            self.runtime.check(self.runtime.lib.mt_free(pointer))
            s['reserved_bytes'] -= capacity
            s['frees'] += 1
        s['active_bytes'] -= capacity

    def empty_cache(self):
        """Release idle blocks only. Live tensors/views remain valid."""
        _clear_bins(self.runtime, self.bins)
        self.cached_bytes = 0
