"""Real MiniTorch serial vs continuous-batching trace with scheduled arrivals.

Input lengths are automatic 32/128/256/512-token cases, not handwritten prompts.
Serial uses the same paged scheduler at max_batch_size=1. Continuous combines
decode rows into one forward. Prefill remains per request, without chunking.
"""
import argparse
import gc
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import minitorch as mt
from minitorch.continuous_batching import ContinuousBatcher
from minitorch.kv_cache import greedy_token
from minitorch.kernel_config import KernelConfig,make_backend
from minitorch.pretrained_gpt2 import load_pretrained,from_gpt2_state
from minitorch.profiling import CsvMetrics,memory_snapshot,synchronize
from minitorch.training_precision import TrainingPrecision
from project.benchmark_inference import build_prompt_cases,clear_idle

DEFAULT_TRACE = [
    dict(id='A',arrival_ms=0,prompt_tokens=32,new_tokens=12),
    dict(id='B',arrival_ms=0,prompt_tokens=128,new_tokens=8),
    dict(id='C',arrival_ms=30,prompt_tokens=256,new_tokens=16),
    dict(id='D',arrival_ms=60,prompt_tokens=512,new_tokens=8),
    dict(id='E',arrival_ms=100,prompt_tokens=32,new_tokens=12),
    dict(id='F',arrival_ms=180,prompt_tokens=128,new_tokens=4),
]


def validate_trace(trace,model):
    if not isinstance(trace,list) or not trace: raise ValueError('Nonempty request trace required')
    seen = set()
    for row in trace:
        if not isinstance(row,dict) or not isinstance(row.get('id'),str) or not row['id'] or row['id'] in seen:
            raise ValueError('Trace requires unique string request IDs')
        seen.add(row['id'])
        if (type(row.get('arrival_ms')) not in (int,float) or not np.isfinite(row['arrival_ms']) or row['arrival_ms']<0 or
            type(row.get('prompt_tokens')) is not int or row['prompt_tokens'] not in (32,128,256,512) or
            type(row.get('new_tokens')) is not int or row['new_tokens']<1 or
            row['prompt_tokens']+row['new_tokens']-1>model.n_positions):
            raise ValueError('Invalid trace arrival or token dimensions')


def reference_tokens(model,prompts,trace,policy):
    """Untimed, independent contiguous-cache generation for numerical validation."""
    result = {}
    with mt.no_grad(),policy.context():
        for row in trace:
            prompt = prompts[row['prompt_tokens']]
            cache = model.create_kv_cache(1,len(prompt)+row['new_tokens']-1)
            try:
                logits,_ = model.prefill(mt.tensor_from_numpy(np.array([prompt],np.float32),model.backend),cache)
                output = []
                for step in range(row['new_tokens']):
                    if step: logits = model.decode_step(token,cache)
                    token = greedy_token(logits)
                    output.append(int(token.to_numpy()[0,0]))
                result[row['id']] = output
            finally: cache.close()
    return result


def run_trace(model,prompts,trace,*,mode,precision,max_batch_size,page_size,budget_bytes,
              pages_per_slab,max_prefills_per_step,prefill_token_budget,repeat=0,is_warmup=False):
    if mode not in ('serial','continuous'): raise ValueError('Unknown benchmark mode')
    validate_trace(trace,model)
    clear_idle(model.backend)
    from minitorch.cuda_storage import runtime
    runtime().reset_peaks()
    pending = sorted(trace,key=lambda r:r['arrival_ms'])
    index,steps,events = 0,[],[]
    engine = ContinuousBatcher(model,precision=precision,max_batch_size=1 if mode=='serial' else max_batch_size,
        page_size=page_size,kv_budget_bytes=budget_bytes,pages_per_slab=pages_per_slab,
        max_prefills_per_step=max_prefills_per_step,prefill_token_budget=prefill_token_budget)
    synchronize(model.backend)
    start = time.perf_counter()
    try:
        while index<len(pending) or engine.has_work:
            elapsed_ms = (time.perf_counter()-start)*1000
            while index<len(pending) and pending[index]['arrival_ms']<=elapsed_ms:
                spec = pending[index]
                engine.submit(spec['id'],prompts[spec['prompt_tokens']],max_new_tokens=spec['new_tokens'])
                index += 1
            if not engine.has_work:
                if index<len(pending): time.sleep(min(.001,max(0,pending[index]['arrival_ms']/1000-(time.perf_counter()-start))))
                continue
            new_events = engine.step()
            for event in new_events:
                events.append(dict(**event,elapsed_seconds=event['arrival_time']-start))
            steps.append(dict(mode=mode,repeat=repeat,is_warmup=is_warmup,
                elapsed_seconds=time.perf_counter()-start,**engine.last_step))
        end = time.perf_counter()
        rows = []
        for spec in trace:
            request = engine.requests[spec['id']]
            scheduled = start+spec['arrival_ms']/1000
            gaps = np.diff(request.token_times)
            rows.append(dict(mode=mode,repeat=repeat,is_warmup=is_warmup,id=spec['id'],
                prompt_tokens=spec['prompt_tokens'],max_new_tokens=spec['new_tokens'],
                arrival_ms=spec['arrival_ms'],state=request.state,finish_reason=request.finish_reason,
                generated_tokens=len(request.output_ids),token_ids=request.output_ids,
                ttft_ms=(request.token_times[0]-scheduled)*1000 if request.token_times else None,
                admission_wait_ms=(request.admitted_at-scheduled)*1000 if request.admitted_at is not None else None,
                scheduling_delay_ms=(request.submitted_at-scheduled)*1000,
                latency_ms=(request.finished_at-scheduled)*1000,
                decode_token_p50_ms=float(np.median(gaps)*1000) if len(gaps) else None,
                decode_token_p95_ms=float(np.percentile(gaps,95)*1000) if len(gaps) else None))
        pool_stats = engine.pool.memory_stats()
        assert pool_stats['kv_pool_used_pages']==0 and engine.reserved_pages==0
        completed = [r for r in rows if r['state']=='completed']
        ttft = [r['ttft_ms'] for r in completed]
        counts = [s['decode_batch_size'] for s in steps if s['decode_batch_size']]
        summary = dict(mode=mode,repeat=repeat,is_warmup=is_warmup,elapsed_seconds=end-start,
            completed_requests=len(completed),rejected_requests=sum(r['state']=='rejected' for r in rows),
            generated_tokens=sum(r['generated_tokens'] for r in rows),
            output_tokens_per_sec=sum(r['generated_tokens'] for r in rows)/(end-start),
            requests_per_sec=len(completed)/(end-start),
            ttft_p50_ms=float(np.median(ttft)) if ttft else None,
            ttft_p95_ms=float(np.percentile(ttft,95)) if ttft else None,
            mean_decode_batch_size=statistics.mean(counts) if counts else 0,
            max_decode_batch_size=max(counts,default=0),
            peak_used_pages=engine.peak_used_pages,
            **memory_snapshot(model.backend),**pool_stats)
    finally: engine.close()
    summary['post_close_active_bytes'] = memory_snapshot(model.backend)['active_bytes']
    return dict(summary=summary,requests=rows,steps=steps,events=events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='distilgpt2')
    parser.add_argument('--revision',default='main')
    parser.add_argument('--cache-dir')
    parser.add_argument('--local-files-only',action='store_true')
    parser.add_argument('--precision',choices=['fp32','fp16','bf16'],default='fp32')
    parser.add_argument('--modes',nargs='+',choices=['serial','continuous'],default=['serial','continuous'])
    parser.add_argument('--max-batch-size',type=int,default=4)
    parser.add_argument('--max-prefills-per-step',type=int,default=1)
    parser.add_argument('--prefill-token-budget',type=int,default=512)
    parser.add_argument('--page-size',type=int,choices=[8,16,32],default=16)
    parser.add_argument('--pages-per-slab',type=int,default=4)
    parser.add_argument('--kv-budget-mb',type=float,default=256)
    parser.add_argument('--warmup',type=int,default=1)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--trace',type=Path,help='JSON list: id, arrival_ms, prompt_tokens, new_tokens')
    parser.add_argument('--output-dir',type=Path,default=None)
    args = parser.parse_args()
    if (min(args.max_batch_size,args.max_prefills_per_step,args.prefill_token_budget,args.pages_per_slab,args.repeats)<1 or
        args.warmup<0 or not np.isfinite(args.kv_budget_mb) or args.kv_budget_mb<=0 or len(set(args.modes))!=len(args.modes)):
        parser.error('Invalid sizes, budget, repetitions or duplicate modes')
    output = args.output_dir or Path(f'output/continuous_batching_{time.time_ns()}')
    output.mkdir(parents=True,exist_ok=False)
    trace = json.loads(args.trace.read_text()) if args.trace else DEFAULT_TRACE
    print(f'Loading {args.model} and validating independent reference generations...',flush=True)
    reference,tokenizer = load_pretrained(args.model,args.revision,args.cache_dir,args.local_files_only)
    cfg = KernelConfig(storage_mode='device',memory_mode='planned',attention_impl='flash',use_kv_cache=True)
    backend = make_backend(cfg)
    model = from_gpt2_state(reference.config,reference.state_dict(),backend)
    del reference
    gc.collect()
    validate_trace(trace,model)
    prompts = {c['prompt_tokens']:c['input_ids'][0].astype(int).tolist() for c in build_prompt_cases(tokenizer,1)}
    policy = TrainingPrecision(args.precision,'device')
    gold = reference_tokens(model,prompts,trace,policy)
    try:
        hardware = subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv'],text=True)
    except (OSError,subprocess.CalledProcessError): hardware = 'nvidia-smi unavailable'
    (output/'config.json').write_text(json.dumps(dict(**vars(args),trace_spec=trace,engine='minitorch',
        gpu=hardware,python=sys.version,prompt_token_ids=prompts,
        metric_scope='Scheduled arrivals to CPU token delivery; model/tokenizer loading, shared-pool initialization and reference validation excluded. Per-request cache allocation included. Request TTFT includes admission wait. Throughput includes idle arrival intervals.',
        admission='FIFO; logical maximum-output page reservations; physical pages grow on demand',
        execution='single process, alternate mode order per repeat; sequential prefill, batched ragged decode',
        budget_scope='KV slab payload only; model/workspace and metadata excluded'),default=str,indent=2))
    runs = []
    try:
        for repeat in range(args.warmup+args.repeats):
            for mode in (args.modes if repeat%2==0 else list(reversed(args.modes))):
                result = run_trace(model,prompts,trace,mode=mode,precision=args.precision,
                    max_batch_size=args.max_batch_size,page_size=args.page_size,budget_bytes=int(args.kv_budget_mb*1024**2),
                    pages_per_slab=args.pages_per_slab,max_prefills_per_step=args.max_prefills_per_step,
                    prefill_token_budget=args.prefill_token_budget,repeat=repeat,is_warmup=repeat<args.warmup)
                for request in result['requests']:
                    request['matches_reference'] = request['token_ids']==gold[request['id']] if request['state']=='completed' else None
                runs.append(result)
                assert all(r['matches_reference'] is not False for r in result['requests']), 'Generation mismatch'
                s = result['summary']
                print(f"{mode} repeat={repeat}: {s['output_tokens_per_sec']:.2f} tok/s; TTFT p50={s['ttft_p50_ms']} ms; max decode batch={s['max_decode_batch_size']}",flush=True)
    finally:
        (output/'raw.json').write_text(json.dumps(dict(runs=runs,reference_token_ids=gold),indent=2))
    summaries = {}
    for mode in args.modes:
        measured = [r for r in runs if r['summary']['mode']==mode and not r['summary']['is_warmup']]
        seconds = sum(r['summary']['elapsed_seconds'] for r in measured)
        times = [q['ttft_ms'] for r in measured for q in r['requests'] if q['ttft_ms'] is not None]
        summaries[mode] = dict(output_tokens_per_sec=sum(r['summary']['generated_tokens'] for r in measured)/seconds,
            completed_requests=sum(r['summary']['completed_requests'] for r in measured),
            rejected_requests=sum(r['summary']['rejected_requests'] for r in measured),
            ttft_p50_ms=float(np.median(times)) if times else None,
            ttft_p95_ms=float(np.percentile(times,95)) if times else None,
            max_decode_batch_size=max(r['summary']['max_decode_batch_size'] for r in measured),
            all_completed_outputs_match_reference=all(q['matches_reference'] for r in measured for q in r['requests'] if q['state']=='completed'))
    if all(m in summaries for m in ('serial','continuous')):
        equal_work = summaries['serial']['completed_requests']==summaries['continuous']['completed_requests'] and not any(summaries[m]['rejected_requests'] for m in ('serial','continuous'))
        summaries['continuous_vs_serial'] = dict(equal_completed_work=equal_work,
            throughput_ratio=summaries['continuous']['output_tokens_per_sec']/summaries['serial']['output_tokens_per_sec'] if equal_work else None)
    for name in ('requests','steps','events'):
        with CsvMetrics(output/f'{name}.csv') as csv:
            for run in runs:
                for row in run[name]:
                    csv.log(dict(mode=run['summary']['mode'],repeat=run['summary']['repeat'],is_warmup=run['summary']['is_warmup'],**{k:v for k,v in row.items() if k not in ('mode','repeat','is_warmup')}))
    (output/'summary.json').write_text(json.dumps(summaries,indent=2))
    print(json.dumps(summaries,indent=2),flush=True)
    print(f'Results: {output.resolve()}',flush=True)


if __name__=='__main__': main()
