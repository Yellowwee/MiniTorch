"""Real MiniTorch request lifetimes under a KV budget (NOT continuous batching).

All live requests retain their caches; one request executes at a time in a
deterministic round-robin trace. CUDA temporary/model memory is reported but is
not charged to the KV-only budget. Workers isolate allocator history per case.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import subprocess
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import minitorch as mt
from minitorch.amp import autocast_dtype
from minitorch.dtypes import itemsize
from minitorch.inference_metrics import shared_cache_memory
from minitorch.kv_cache import greedy_token
from minitorch.memory_pool import MemoryPool
from minitorch.paged_kv_cache import KVPagePool
from minitorch.profiling import CsvMetrics,memory_snapshot,synchronize
from minitorch.training_precision import TrainingPrecision
from project.benchmark_inference import build_prompt_cases,clear_idle,validate_logits

CASES=('full_prefix','cached_known','cached_upper','paged')
DEFAULT_TRACE=(
    dict(id='A',arrival=0,prompt_tokens=32,new_tokens=8,max_new_tokens=64),
    dict(id='B',arrival=0,prompt_tokens=512,new_tokens=24,max_new_tokens=128),
    dict(id='C',arrival=10,prompt_tokens=128,new_tokens=12,max_new_tokens=64),
    dict(id='D',arrival=16,prompt_tokens=256,new_tokens=6,max_new_tokens=64),
)


def cache_capacity(spec,case):
    return spec['prompt_tokens']+(spec['new_tokens'] if case=='cached_known' else spec['max_new_tokens'])-1


def contiguous_allocation_bytes(model,capacity,batch_size=1):
    """Match each actual K/V tensor's allocator bucket, not just payload bytes."""
    size=itemsize(autocast_dtype() or 'float32')
    bucket=MemoryPool.bucket if getattr(model.backend,'memory_pool',None) is not None else int
    return sum(2*bucket(batch_size*layer.attention.n_head*layer.attention.attn_hidden_dim*capacity*size)
               for layer in model.transformer_layers())


def projected_paged_bytes(pool,caches,*,live_after,cache=None,new_capacity=None,batch_size=1,pending_pages=0):
    """Total pool slabs + slab table + all request tables after this append."""
    stats=pool.memory_stats()
    existing=cache.nbytes//pool.page_bytes if cache is not None else 0
    pages=batch_size*math.ceil(live_after/pool.page_size)-existing+pending_pages
    additional_slabs=math.ceil(max(0,pages-stats['kv_pool_free_pages'])/pool.pages_per_slab)
    tables=sum(c.block_table.nbytes for c in caches)
    if cache is None:
        if new_capacity is None: raise ValueError('new_capacity required for admission')
        tables+=batch_size*math.ceil(new_capacity/pool.page_size)*4
    return stats['kv_pool_reserved_bytes']+additional_slabs*pool.slab_bytes+stats['kv_pool_metadata_bytes']+tables


def validate_trace(trace,model):
    if not trace or len({s['id'] for s in trace})!=len(trace): raise ValueError('Nonempty trace with unique IDs required')
    for s in trace:
        if (any(type(s[k]) is not int for k in ('arrival','prompt_tokens','new_tokens','max_new_tokens')) or
            s['arrival']<0 or min(s['prompt_tokens'],s['new_tokens'])<1 or s['max_new_tokens']<s['new_tokens']):
            raise ValueError('Invalid request dimensions or declared output limit')
        if cache_capacity(s,'cached_upper')>model.n_positions: raise ValueError('Trace exceeds model positions')


def run_trace(model,prompts,trace,case,policy,*,budget_bytes,page_size=16,pages_per_slab=4):
    from minitorch.cuda_storage import runtime
    if case not in CASES: raise ValueError('Unknown trace case')
    validate_trace(trace,model)
    if budget_bytes<1: raise ValueError('Positive KV budget required')
    b=model.backend;clear_idle(b);runtime().reset_peaks()
    active={};pool=None;rows=[];completed={};failed=[];start=time.perf_counter()
    def caches(): return [s['cache'] for s in active.values() if s['cache'] is not None]
    def record(tick,event,request_id):
        synchronize(b)
        memory=memory_snapshot(b)
        rows.append(dict(case=case,event_index=len(rows),tick=tick,event=event,request_id=request_id,
            elapsed_seconds=time.perf_counter()-start,live_requests=len(active),budget_bytes=budget_bytes,
            gpu_active_bytes=memory['active_bytes'],gpu_reserved_bytes=memory['reserved_bytes'],
            gpu_peak_active_bytes=memory['peak_active_bytes'],gpu_peak_reserved_bytes=memory['peak_reserved_bytes'],
            tensor_pool_idle_bytes=memory['cached_bytes'],**shared_cache_memory(caches(),pool)))
    try:
        with mt.no_grad(),policy.context():
            if case=='paged':
                pool=KVPagePool(model,page_size=page_size,max_bytes=budget_bytes,pages_per_slab=pages_per_slab)
            record(-1,'start','')
            end=max(s['arrival']+s['new_tokens'] for s in trace)
            for tick in range(end):
                for spec in (s for s in trace if s['arrival']==tick):
                    capacity=cache_capacity(spec,case)
                    history=np.asarray(prompts[spec['prompt_tokens']],np.float32).copy()
                    if history.shape!=(1,spec['prompt_tokens']): raise ValueError('Trace requires batch=1 exact-length prompts')
                    used=shared_cache_memory(caches(),pool)['kv_total_reserved_bytes']
                    # Same-tick arrivals have tables but have not executed prefill
                    # yet. Charge their promised pages before admitting another.
                    pending=sum(math.ceil(s['spec']['prompt_tokens']/page_size) for s in active.values()
                                if s['cache'] is not None and s['cache'].length==0)
                    projected=(projected_paged_bytes(pool,caches(),live_after=spec['prompt_tokens'],new_capacity=capacity,pending_pages=pending)
                        if pool else used+contiguous_allocation_bytes(model,capacity) if case!='full_prefix' else 0)
                    if projected>budget_bytes:
                        failed.append(dict(id=spec['id'],reason='admission_budget',tick=tick,projected_bytes=projected))
                        record(tick,'reject_admission',spec['id']);continue
                    cache=None
                    if case!='full_prefix':
                        cache=model.create_kv_cache(1,capacity,cache_impl='paged' if pool else 'contiguous',
                            page_size=page_size,page_pool=pool)
                    active[spec['id']]=dict(spec=spec,cache=cache,history=history,next_ids=None,generated=[])
                    record(tick,'admit',spec['id'])
                for request_id in list(active):
                    state=active[request_id];cache=state['cache'];spec=state['spec'];step=len(state['generated'])
                    if pool:
                        projected=projected_paged_bytes(pool,caches(),live_after=spec['prompt_tokens']+step,cache=cache)
                        if projected>budget_bytes:
                            failed.append(dict(id=request_id,reason='growth_budget',tick=tick,projected_bytes=projected))
                            cache.close();del active[request_id]
                            state['cache']=None;state['next_ids']=None
                            clear_idle(b)
                            record(tick,'reject_growth',request_id);continue
                    idx=state['next_ids'] if cache is not None and step else mt.tensor_from_numpy(state['history'],b)
                    logits=(model(idx,last_only=True) if cache is None else
                            model.prefill(idx,cache)[0] if step==0 else model.decode_step(idx,cache))
                    next_ids=greedy_token(logits);host=next_ids.to_numpy()
                    state['next_ids']=next_ids;state['generated'].append(int(host[0,0]))
                    if cache is None: state['history']=np.concatenate([state['history'],host],axis=1)
                    del logits,idx,next_ids
                    record(tick,'token',request_id)
                    if len(state['generated'])==spec['new_tokens']:
                        completed[request_id]=state['generated']
                        if cache is not None: cache.close()
                        state['cache']=None;state['next_ids']=None
                        del active[request_id]
                        # Return released contiguous allocations to CUDA too.
                        # Apply the same generic-pool drain to all cases; the
                        # explicit shared paged pool intentionally stays live.
                        clear_idle(b)
                        record(tick,'finish',request_id)
            record(end,'drained','')
            summary=dict(case=case,completed_requests=len(completed),failed_requests=failed,
                peak_live_requests=max(r['live_requests'] for r in rows),
                peak_kv_live_bytes=max(r['kv_live_bytes'] for r in rows),
                peak_kv_reserved_bytes=max(r['kv_total_reserved_bytes'] for r in rows),
                peak_active_bytes=runtime().stats['peak_active_bytes'],peak_reserved_bytes=runtime().stats['peak_reserved_bytes'],
                page_reuses=rows[-1]['kv_pool_page_reuses'],kv_budget_bytes=budget_bytes,
                kv_budget_applies=case!='full_prefix',trace_seconds=time.perf_counter()-start)
    finally:
        for state in active.values():
            if state['cache'] is not None: state['cache'].close()
            state['next_ids']=None
        active.clear()
        if pool is not None: pool.close()
    record(end,'pool_closed','')
    return dict(summary=summary,events=rows,generations=completed)


def capacity_probe(model,prompt,case,policy,*,budget_bytes,max_live_requests,page_size=16,pages_per_slab=4,
                   new_tokens=32,max_new_tokens=128):
    """Resident cache capacity at prefill; not future completion/serving capacity."""
    if case=='full_prefix': return dict(admitted_at_prefill=None,capped=None,reason='No persistent KV cache')
    from minitorch.cuda_storage import runtime
    if max_live_requests<1: raise ValueError('Positive max_live_requests required')
    spec=dict(prompt_tokens=prompt.shape[1],new_tokens=new_tokens,max_new_tokens=max_new_tokens)
    capacity=cache_capacity(spec,case)
    if capacity>model.n_positions: raise ValueError('Probe exceeds model positions')
    b=model.backend;clear_idle(b);runtime().reset_peaks();pool=None;caches=[]
    try:
        with mt.no_grad(),policy.context():
            if case=='paged': pool=KVPagePool(model,page_size=page_size,max_bytes=budget_bytes,pages_per_slab=pages_per_slab)
            for _ in range(max_live_requests):
                projected=(projected_paged_bytes(pool,caches,live_after=prompt.shape[1],new_capacity=capacity)
                    if pool else shared_cache_memory(caches)['kv_total_reserved_bytes']+contiguous_allocation_bytes(model,capacity))
                if projected>budget_bytes: break
                cache=model.create_kv_cache(1,capacity,cache_impl='paged' if pool else 'contiguous',
                    page_size=page_size,page_pool=pool)
                caches.append(cache)
                logits,_=model.prefill(mt.tensor_from_numpy(prompt,b),cache)
                del logits
            synchronize(b)
            return dict(admitted_at_prefill=len(caches),capped=len(caches)==max_live_requests,
                max_live_requests=max_live_requests,prompt_tokens=prompt.shape[1],capacity_per_request=capacity,
                future_growth_guaranteed=False if pool else True,
                peak_active_bytes=runtime().stats['peak_active_bytes'],peak_reserved_bytes=runtime().stats['peak_reserved_bytes'],
                **shared_cache_memory(caches,pool))
    finally:
        for cache in caches: cache.close()
        if pool is not None: pool.close()


def generation_comparison(results):
    """Only completed matched requests are comparable; report exclusions."""
    ids=set().union(*(r['generations'] for r in results.values()))
    matched={rid:{case:r['generations'][rid] for case,r in results.items() if rid in r['generations']} for rid in sorted(ids)}
    return {rid:dict(compared_cases=list(values),all_cases_completed=len(values)==len(results),
        identical=all(v==next(iter(values.values())) for v in values.values()) if len(values)>1 else None)
        for rid,values in matched.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='distilbert/distilgpt2');p.add_argument('--revision',default='main')
    p.add_argument('--cache-dir');p.add_argument('--local-files-only',action='store_true')
    p.add_argument('--precision',choices=['fp32','fp16','bf16'],default='fp32')
    p.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES))
    p.add_argument('--kv-budget-mb',type=float,default=256,help='MiB, including KV allocator rounding and tables; excludes model/temporary tensors')
    p.add_argument('--page-size',type=int,choices=[8,16,32],default=16)
    p.add_argument('--pages-per-slab',type=int,default=4)
    p.add_argument('--max-live-requests',type=int,default=32,help='Capacity probe safety cap, not a serving batch')
    p.add_argument('--output-dir');p.add_argument('--no-plots',action='store_true')
    p.add_argument('--worker-case',choices=CASES,help=argparse.SUPPRESS)
    args=p.parse_args()
    if not np.isfinite(args.kv_budget_mb) or args.kv_budget_mb<=0 or min(args.pages_per_slab,args.max_live_requests)<1:
        p.error('Positive finite budget and positive sizes required')
    if len(set(args.cases))!=len(args.cases): p.error('Duplicate cases')
    output=Path(args.output_dir or f'output/kv_memory_{time.time_ns()}');output.mkdir(parents=True,exist_ok=False)
    if args.worker_case is None:
        results={}
        for case in args.cases:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker-case',case,
                '--output-dir',str(output/case),'--no-plots'],check=True)
            results[case]=json.loads((output/case/'result.json').read_text(encoding='utf-8'))
        with CsvMetrics(output/'events.csv') as writer:
            for result in results.values():
                for row in result['events']: writer.log(row)
        summary=dict(cases={k:dict(**v['summary'],capacity_probe=v['capacity_probe']) for k,v in results.items()},
            generation_comparison=generation_comparison(results))
        (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        (output/'raw.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        (output/'config.json').write_text(json.dumps(dict(**vars(args),trace=DEFAULT_TRACE,
            execution='Sequential round-robin requests with simultaneously live caches; NOT continuous batching',
            budget_scope='KV slabs/tables or contiguous allocations including bucket rounding; general tensor pool drained at request completion/rejection; model and temporary tensors excluded from KV budget but included in GPU counters',
            capacity_scope='Prefill-resident caches at 128 tokens; capped result is a lower bound; paged requests are NOT guaranteed room for later decode',
            cached_known='Capacity knows actual final length: prompt + new_tokens - 1',
            cached_upper='Capacity reserves declared output limit: prompt + max_new_tokens - 1',
            paged='Same declared limit as cached_upper; physical pages grow on demand; idle slabs retained',
            worker_configs={c:str(Path(c)/'config.json') for c in args.cases}),indent=2),encoding='utf-8')
        if not args.no_plots:
            from project.inference_plots import plot_trace
            plot_trace(output)
        print(f'Results: {output.resolve()}',flush=True);return
    from minitorch.kernel_config import KernelConfig,make_backend
    from minitorch.pretrained_gpt2 import load_pretrained,from_gpt2_state
    print(f'Loading real model for {args.worker_case}...',flush=True)
    reference,tokenizer=load_pretrained(args.model,args.revision,args.cache_dir,args.local_files_only)
    cases=build_prompt_cases(tokenizer,1);prompts={c['prompt_tokens']:c['input_ids'] for c in cases}
    cfg=KernelConfig(storage_mode='device',memory_mode='planned',attention_impl='flash',use_kv_cache=True)
    b=make_backend(cfg);model=from_gpt2_state(reference.config,reference.state_dict(),b)
    # Validation uses a separate ample cache budget, outside the measured trace.
    validation=validate_logits(reference,model,prompts[32],args.precision,
        cache_impl='paged' if args.worker_case=='paged' else 'contiguous',page_size=args.page_size)
    import torch,transformers
    try: gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv'],text=True)
    except (OSError,subprocess.CalledProcessError): gpu='unavailable'
    config=dict(**vars(args),execution=cfg.to_dict(),model_config=reference.config.to_dict(),
        resolved_revision=getattr(reference.config,'_commit_hash',None),gpu=gpu,python=sys.version,
        torch_version=torch.__version__,transformers_version=transformers.__version__,validation=validation,
        token_ids={str(n):ids.astype(int).tolist() for n,ids in prompts.items()})
    (output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    del reference
    policy=TrainingPrecision(args.precision,'device');budget=int(args.kv_budget_mb*1024**2)
    result=run_trace(model,prompts,DEFAULT_TRACE,args.worker_case,policy,budget_bytes=budget,
        page_size=args.page_size,pages_per_slab=args.pages_per_slab)
    print(f"Trace {args.worker_case}: {result['summary']}",flush=True)
    result['capacity_probe']=capacity_probe(model,prompts[128],args.worker_case,policy,budget_bytes=budget,
        max_live_requests=args.max_live_requests,page_size=args.page_size,pages_per_slab=args.pages_per_slab)
    (output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__': main()
