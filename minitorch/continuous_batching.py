"""Single-thread, iteration-level continuous batching with a shared paged KV pool.

Decode batches have one token per request, with independent context lengths.
Prefills execute separately after decode, under a per-iteration admission limit.
This is not a distributed serving engine or mixed prefill/decode CUDA kernel.
"""
from collections import deque
from dataclasses import dataclass, field
import math
import time
import numpy as np
from .autodiff import no_grad
from .amp import autocast
from .kv_cache import greedy_token
from .paged_kv_cache import KVPagePool
from .tensor_functions import tensor_from_numpy


@dataclass
class GenerationRequest:
    request_id: str
    prompt_ids: tuple
    max_new_tokens: int
    eos_token_id: int | None
    submitted_at: float
    capacity: int
    reserved_pages: int
    state: str = 'queued'
    finish_reason: str | None = None
    admitted_at: float | None = None
    finished_at: float | None = None
    output_ids: list = field(default_factory=list)
    token_times: list = field(default_factory=list)
    cache: object = field(default=None,repr=False)

    @property
    def done(self):
        return self.state in ('completed','cancelled','rejected','failed')

    @property
    def ttft_seconds(self):
        return self.token_times[0]-self.submitted_at if self.token_times else None


class ContinuousBatcher:
    """Submit requests between steps; each step yields newly available tokens.

    FIFO admission reserves each request's *maximum* page demand logically, but
    allocates physical pages only as tokens arrive. This conservative policy
    prevents active requests deadlocking on KV growth. The KV budget excludes
    model/temporary tensors and page-table metadata. One CUDA default stream;
    the owner must serialize submit/step/cancel/close and keep weights unchanged.
    """
    def __init__(self, model, *, max_batch_size=4, page_size=16,
                 kv_budget_bytes=256*1024**2, pages_per_slab=4, precision='fp32',
                 max_prefills_per_step=1, prefill_token_budget=512):
        for value in (max_batch_size,max_prefills_per_step,prefill_token_budget):
            if type(value) is not int or value<1:
                raise ValueError('Positive integer batch size and prefill limits required')
        if precision not in ('fp32','fp16','bf16'):
            raise ValueError('precision must be fp32, fp16 or bf16')
        if any(l.attention.attention_impl!='flash' for l in model.transformer_layers()):
            raise ValueError('Continuous batching requires FlashAttention')
        self.model, self.precision = model,precision
        self.max_batch_size = max_batch_size
        self.max_prefills_per_step = max_prefills_per_step
        self.prefill_token_budget = prefill_token_budget
        with no_grad(),self._precision_context():
            self.pool = KVPagePool(model,page_size=page_size,max_bytes=kv_budget_bytes,pages_per_slab=pages_per_slab)
        self.requests, self.active, self.pending = {},{},deque()
        self.reserved_pages = 0
        self.peak_used_pages = 0
        self.iteration = 0
        self.last_step = None
        self.closed = False

    def _precision_context(self):
        return autocast('float16' if self.precision=='fp16' else 'bfloat16',enabled=self.precision!='fp32')

    @property
    def has_work(self):
        return bool(self.active or self.pending)

    def submit(self, request_id, prompt_ids, *, max_new_tokens=32, eos_token_id=None):
        if self.closed: raise ValueError('Batcher is closed')
        if not isinstance(request_id,str) or not request_id or request_id in self.requests:
            raise ValueError('Nonempty unique request ID required')
        ids = np.asarray(prompt_ids)
        if (ids.ndim!=1 or ids.size<1 or ids.dtype.kind not in 'iu' or
            np.any(ids<0) or np.any(ids>=self.model.n_vocab)):
            raise ValueError('prompt_ids must be nonempty 1-D integer vocabulary IDs')
        if type(max_new_tokens) is not int or max_new_tokens<1:
            raise ValueError('Positive integer max_new_tokens required')
        if eos_token_id is not None and (type(eos_token_id) is not int or not 0<=eos_token_id<self.model.n_vocab):
            raise ValueError('Invalid EOS token ID')
        capacity = int(ids.size)+max_new_tokens-1
        if capacity>self.model.n_positions:
            raise ValueError('Prompt plus output exceeds model position capacity')
        request = GenerationRequest(request_id,tuple(int(x) for x in ids),max_new_tokens,eos_token_id,
                                    time.perf_counter(),capacity,math.ceil(capacity/self.pool.page_size))
        self.requests[request_id] = request
        if request.reserved_pages>self.pool.max_pages:
            request.state,request.finish_reason,request.finished_at = 'rejected','kv_budget',time.perf_counter()
        else:
            self.pending.append(request)
        return request

    def _finish(self, request, state, reason):
        if request.request_id in self.active:
            del self.active[request.request_id]
            self.reserved_pages -= request.reserved_pages
        if request.cache is not None:
            request.cache.close()
            request.cache = None
        request.state,request.finish_reason,request.finished_at = state,reason,time.perf_counter()

    def cancel(self, request_id):
        request = self.requests[request_id]
        if request.done: return False
        if request.state=='queued': self.pending.remove(request)
        self._finish(request,'cancelled','cancelled')
        return True

    def _emit(self, request, token_id, arrival):
        request.output_ids.append(int(token_id))
        request.token_times.append(arrival)
        reason = ('eos' if token_id==request.eos_token_id else
                  'length' if len(request.output_ids)>=request.max_new_tokens else None)
        if reason is not None: self._finish(request,'completed',reason)
        return dict(request_id=request.request_id,token_id=int(token_id),
                    output_index=len(request.output_ids)-1,arrival_time=arrival,finish_reason=reason)

    def step(self):
        if self.closed: raise ValueError('Batcher is closed')
        start = time.perf_counter()
        events,decode_ids,prefill_ids = [],list(self.active),[]
        prefill_tokens = 0
        with no_grad(),self._precision_context():
            # All active requests share one projection/MLP/attention model forward.
            if decode_ids:
                batch = [self.active[rid] for rid in decode_ids]
                try:
                    ids = tensor_from_numpy(np.array([[r.output_ids[-1]] for r in batch],np.float32),self.model.backend)
                    logits = self.model.decode_batch(ids,[r.cache for r in batch])
                    self.peak_used_pages = max(self.peak_used_pages,self.pool.memory_stats()['kv_pool_used_pages'])
                    tokens = greedy_token(logits).to_numpy()[:,0]
                    arrival = time.perf_counter()
                    events.extend(self._emit(r,t,arrival) for r,t in zip(batch,tokens))
                except Exception:
                    for request in batch:
                        if not request.done: self._finish(request,'failed','decode_error')
                    raise
            decode_seconds = time.perf_counter()-start
            # Decode first: completed requests release pages before admissions.
            while self.pending and len(self.active)<self.max_batch_size and len(prefill_ids)<self.max_prefills_per_step:
                request = self.pending[0]
                if self.reserved_pages+request.reserved_pages>self.pool.max_pages: break
                # A single oversized prompt may run alone, ensuring progress.
                if prefill_ids and prefill_tokens+len(request.prompt_ids)>self.prefill_token_budget: break
                self.pending.popleft()
                self.active[request.request_id] = request
                self.reserved_pages += request.reserved_pages
                request.state,request.admitted_at = 'active',time.perf_counter()
                try:
                    request.cache = self.model.create_kv_cache(1,request.capacity,cache_impl='paged',
                        page_size=self.pool.page_size,page_pool=self.pool)
                    ids = tensor_from_numpy(np.array([request.prompt_ids],np.float32),self.model.backend)
                    logits,_ = self.model.prefill(ids,request.cache)
                    self.peak_used_pages = max(self.peak_used_pages,self.pool.memory_stats()['kv_pool_used_pages'])
                    token = greedy_token(logits).to_numpy()[0,0]
                    events.append(self._emit(request,token,time.perf_counter()))
                except Exception:
                    self._finish(request,'failed','prefill_error')
                    raise
                prefill_ids.append(request.request_id)
                prefill_tokens += len(request.prompt_ids)
        capacity_bytes = sum(r.cache.nbytes for r in self.active.values())
        live_bytes = sum(r.cache.length*self.pool.page_bytes//self.pool.page_size for r in self.active.values())
        self.last_step = dict(iteration=self.iteration,decode_request_ids=decode_ids,
            decode_batch_size=len(decode_ids),prefill_request_ids=prefill_ids,prefill_tokens=prefill_tokens,
            active_request_ids=list(self.active),queued_requests=len(self.pending),
            logical_reserved_pages=self.reserved_pages,decode_seconds=decode_seconds,
            kv_live_bytes=live_bytes,kv_capacity_bytes=capacity_bytes,
            kv_slot_utilization=live_bytes/capacity_bytes if capacity_bytes else None,
            peak_used_pages=self.peak_used_pages,
            step_seconds=time.perf_counter()-start,**self.pool.memory_stats())
        self.iteration += 1
        return events

    def close(self):
        if self.closed: return
        for request in tuple(self.requests.values()):
            if not request.done: self.cancel(request.request_id)
        self.pool.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()
