"""Synchronized wall timing and raw, unsmoothed CSV metrics."""
import csv
import time
from pathlib import Path


def synchronize(backend):
    if backend.device_storage:
        from .cuda_storage import runtime
        runtime().synchronize()
    # The legacy host CUDA ABI synchronizes before returning each result.
    elif backend.cuda:
        from numba import cuda
        cuda.synchronize()


def memory_snapshot(backend):
    if not backend.device_storage:
        # Host ABI allocations are inside old C wrappers and not instrumented.
        return {}
    from .cuda_storage import runtime
    result = runtime().snapshot()
    workspace = getattr(backend, 'workspace', None)
    result['workspace_bytes'] = workspace.nbytes if workspace is not None else 0
    return result


def measure(backend, fn):
    """Return (result, synchronized wall seconds). Includes Python dispatch."""
    synchronize(backend)
    start = time.perf_counter()
    result = fn()
    synchronize(backend)
    return result, time.perf_counter()-start


class CsvMetrics:
    def __init__(self, path):
        self.path = Path(path) if path else None
        self.file = None
        self.writer = None

    def __enter__(self):
        if self.path:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            # Fail before training rather than silently overwrite another run.
            self.file = self.path.open('x',newline='',encoding='utf-8')
        return self

    def log(self, row):
        if self.file is None:
            return
        if self.writer is None:
            self.writer = csv.DictWriter(self.file,fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()

    def __exit__(self, *exc):
        if self.file is not None:
            self.file.close()
