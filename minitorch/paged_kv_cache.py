"""Inference-only paged KV storage, with exclusive pages and a shared slab pool.

Fixed batch/equal live lengths per cache; multiple independent caches may share
one pool. Single CUDA default stream, no thread-safe scheduling or prefix sharing.
A physical page ID owns K/V slots in every layer. Slabs never move or resize.
"""
import ctypes as ct
from functools import lru_cache
from pathlib import Path
import weakref
import numpy as np
from .amp import autocast_dtype
from .cuda_storage import RawDeviceBuffer
from .cuda_resident_ops import CudaResidentOps as Ops, call, pointer, P, I
from .dtypes import dtype_code, itemsize
from .kv_cache import CacheBase


@lru_cache(maxsize=1)
def _library():
    try:
        return ct.CDLL(str(Path(__file__).parent/'cuda_kernels/paged_attention_kernel.so'))
    except OSError as error:
        raise RuntimeError('Build paged_attention_kernel.so with compile_cuda.sh') from error


class KVPagePool:
    """On-demand slabs; max_bytes bounds KV payload, metadata reported separately.

    Each slab: [layers, 2, pages_per_slab, heads, page_size, head_dim].
    Free pages stay in the pool until close(). GPU counters include idle slabs.
    The small uint64 slab-address table is allocated at construction time.
    """
    def __init__(self, model, *, page_size=16, max_bytes=256*1024**2, pages_per_slab=4):
        if not model.backend.device_storage or model.training:
            raise ValueError('Page pool requires device storage and model.eval()')
        layers = model.transformer_layers()
        layout = tuple((l.attention.n_head,l.attention.attn_hidden_dim) for l in layers)
        if not layout or len(set(layout)) != 1 or any(not l.attention._flash_supported() for l in layers):
            raise ValueError('Page pool requires uniform head layout and head_dim<=128')
        if type(page_size) is not int or page_size not in (8,16,32):
            raise ValueError('page_size must be 8, 16 or 32')
        if type(max_bytes) is not int or type(pages_per_slab) is not int or min(max_bytes,pages_per_slab)<1:
            raise ValueError('Positive KV budget and pages_per_slab required')
        self.model = weakref.ref(model)
        self.backend = model.backend
        self.layout = layout
        self.layers = len(layout)
        self.heads,self.dim = layout[0]
        self.dtype = autocast_dtype() or 'float32'
        self.page_size = page_size
        self.page_bytes = 2*self.layers*self.heads*self.dim*page_size*itemsize(self.dtype)
        if max_bytes < self.page_bytes:
            raise ValueError('KV budget must hold at least one page')
        self.pages_per_slab = min(pages_per_slab,max_bytes//self.page_bytes)
        self.slab_bytes = self.pages_per_slab*self.page_bytes
        self.max_slabs = max_bytes//self.slab_bytes
        self.max_pages = self.max_slabs*self.pages_per_slab
        if self.max_pages > 2**31-1:
            raise ValueError('Page IDs must fit int32')
        self.max_bytes = max_bytes
        self._slabs = []
        self._free = []
        self._leased = {}
        self._issued = set()
        self.page_acquires = self.page_releases = self.page_reuses = 0
        self.closed = False
        self.slab_table = RawDeviceBuffer(self.max_slabs*8)
        self.slab_table.upload(np.zeros(self.max_slabs,np.uint64))

    def acquire(self, count, owner):
        if self.closed or type(count) is not int or count<0:
            raise ValueError('Closed pool or invalid page count')
        if count > self.max_pages-len(self._leased):
            raise MemoryError('KV page budget exhausted')
        # Commit leases only after every required slab is available. An allocation
        # failure may leave reusable idle slabs, but never partially leased pages.
        while len(self._free)<count:
            index = len(self._slabs)
            slab = RawDeviceBuffer(self.slab_bytes)
            try:
                self.slab_table.upload(np.array([slab.ptr],np.uint64),index*8)
            except Exception:
                slab.close()
                raise
            self._slabs.append(slab)
            start = index*self.pages_per_slab
            self._free.extend(reversed(range(start,start+self.pages_per_slab)))
        pages = [self._free.pop() for _ in range(count)]
        self.page_acquires += count
        self.page_reuses += sum(p in self._issued for p in pages)
        self._issued.update(pages)
        self._leased.update((p,owner) for p in pages)
        return pages

    def release(self, pages, owner):
        pages = list(pages)
        if self.closed or len(set(pages)) != len(pages) or any(p not in self._leased or self._leased[p] is not owner for p in pages):
            raise ValueError('Invalid page ownership or duplicate release')
        for p in pages:
            del self._leased[p]
            self._free.append(p)
        self.page_releases += len(pages)

    def memory_stats(self):
        reserved = len(self._slabs)*self.slab_bytes
        assigned = len(self._leased)*self.page_bytes
        return dict(kv_pool_reserved_bytes=reserved,kv_pool_assigned_bytes=assigned,
            kv_pool_free_bytes=reserved-assigned,kv_pool_metadata_bytes=0 if self.closed else self.slab_table.nbytes,
            kv_pool_used_pages=len(self._leased),kv_pool_free_pages=len(self._free),
            kv_pool_max_pages=self.max_pages,kv_pool_page_acquires=self.page_acquires,
            kv_pool_page_releases=self.page_releases,kv_pool_page_reuses=self.page_reuses)

    def close(self):
        if self.closed: return
        if self._leased:
            raise ValueError('Cannot close page pool while caches own pages')
        for slab in self._slabs: slab.close()
        self._slabs.clear()
        self._free.clear()
        self.slab_table.close()
        self.closed = True


def _release_cache(pool, owner, pages, table, owns_pool):
    # Mutable page lists are shared with the cache, without retaining the cache.
    owned = [p for row in pages for p in row]
    if owned: pool.release(owned,owner)
    for row in pages: row.clear()
    table.close()
    if owns_pool: pool.close()


class PagedKVCache(CacheBase):
    def __init__(self, model, batch_size=1, capacity=None, *, page_size=16,
                 page_pool=None, kv_budget_bytes=256*1024**2, pages_per_slab=4,paged_prefill_impl='flash'):
        if paged_prefill_impl not in ('flash','paged_reference'):
            raise ValueError('Choose flash or paged_reference prefill')
        self.paged_prefill_impl=paged_prefill_impl
        super().__init__(model,batch_size,capacity)
        self._owns_pool = page_pool is None
        pool = page_pool if page_pool is not None else KVPagePool(model,page_size=page_size,
            max_bytes=kv_budget_bytes,pages_per_slab=pages_per_slab)
        if (pool.closed or pool.model() is not model or pool.backend is not self.backend or
            pool.layout != self.layout or pool.dtype != self.dtype or pool.page_size != page_size):
            if self._owns_pool: pool.close()
            raise ValueError('Page pool/model/backend/layout/precision/page_size mismatch')
        self.pool = pool
        self.page_size = page_size
        self.table_columns = (self.capacity+page_size-1)//page_size
        if self.batch_size*self.table_columns>2**31-1:
            if self._owns_pool: pool.close()
            raise ValueError('Page table is too large')
        self._owner = object()
        self._pages = [[] for _ in range(batch_size)]
        self._host_table = np.full((batch_size,self.table_columns),-1,np.int32)
        try:
            self.block_table = RawDeviceBuffer(self._host_table.nbytes)
        except Exception:
            if self._owns_pool: pool.close()
            raise
        self._new_pages = []
        self._previous_counts = None
        self._release = weakref.finalize(self,_release_cache,pool,self._owner,self._pages,self.block_table,self._owns_pool)

    def validate(self, model, idx, prefill):
        super().validate(model,idx,prefill)
        if self.pool.closed:
            raise ValueError('Page pool is closed')

    def prepare_append(self, tokens):
        super().prepare_append(tokens)
        self._previous_counts = [len(row) for row in self._pages]
        required = (self.length+tokens+self.page_size-1)//self.page_size
        count = sum(required-len(row) for row in self._pages)
        try:
            self._new_pages = self.pool.acquire(count,self._owner)
            cursor = 0
            for row in self._pages:
                extra = required-len(row)
                row.extend(self._new_pages[cursor:cursor+extra])
                cursor += extra
            if count:
                self._host_table.fill(-1)
                for b,row in enumerate(self._pages): self._host_table[b,:len(row)] = row
                self.block_table.upload(self._host_table)
        except Exception:
            self.abort_append()
            raise

    def _attention(self, layer, q, k, v, *, prefill):
        self._check_attention(layer,q,k,v)
        b,h,n,d = q.shape
        q,k,v = (t.contiguous() for t in (q,k,v))
        self._append_layer_kv(layer,k,v)
        if prefill and self.paged_prefill_impl=='flash':
            out=Ops.flash_attention_inference(q,k,v,length=n)
            self.last_attention_path='flash_prefill'
        else:
            out=self._paged_attention(layer,q)
            self.last_attention_path='paged_reference_prefill' if prefill else 'paged_decode'
        self._written_layers.add(layer)
        return out

    def _append_layer_kv(self,layer,k,v):
        b,h,n,d=k.shape
        pool = self.pool
        common = (self.page_size,pool.pages_per_slab,len(pool._slabs),self.table_columns,layer,len(self.layout))
        call(_library(),'mt_paged_append',[P]*4+[I]*12,
            pointer(k,True),pointer(v,True),pool.slab_table.ptr,self.block_table.ptr,
            b,h,n,d,*common,self.length,dtype_code(self.dtype))

    def _paged_attention(self,layer,q):
        b,h,n,d=q.shape
        pool=self.pool
        common=(self.page_size,pool.pages_per_slab,len(pool._slabs),self.table_columns,layer,len(self.layout))
        out = Ops.empty(q.shape,self.backend,self.dtype)
        call(_library(),'mt_paged_attention',[P]*4+[I]*13,
            pointer(out,True),pointer(q,True),pool.slab_table.ptr,self.block_table.ptr,
            b,h,n,self.length+n,d,*common,self.length,dtype_code(self.dtype))
        return out

    def commit_append(self):
        super().commit_append()
        self._new_pages = []
        self._previous_counts = None

    def abort_append(self):
        if self._new_pages:
            self.pool.release(self._new_pages,self._owner)
            for row,count in zip(self._pages,self._previous_counts): del row[count:]
        self._new_pages = []
        self._previous_counts = None
        super().abort_append()

    @property
    def nbytes(self):
        return sum(map(len,self._pages))*self.pool.page_bytes

    def memory_stats(self):
        capacity = self.nbytes
        live = self.batch_size*self.length*self.pool.page_bytes//self.page_size
        result = dict(kv_live_bytes=live,kv_capacity_bytes=capacity,kv_unused_bytes=capacity-live,
            kv_slot_utilization=live/capacity if capacity else None,
            kv_metadata_bytes=0 if self.closed else self.block_table.nbytes,
            kv_allocator_slack_bytes=0)
        # Pool totals are shared: do not sum them across caches using this pool.
        result.update(self.pool.memory_stats())
        return result

    def reset(self):
        if self.closed: raise ValueError('Cannot reset a closed cache')
        self.abort_append()
        self.pool.release([p for row in self._pages for p in row],self._owner)
        for row in self._pages: row.clear()
        self._host_table.fill(-1)
        super().reset()

    def close(self):
        if self.closed: return
        self.abort_append()
        self._release()
        self.closed,self.valid,self.length = True,False,0
