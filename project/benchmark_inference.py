"""Pretrained GPT-2 inference in MiniTorch: matched KV-cache TTFT/throughput.

Transformers only loads weights/tokenizer and supplies an untimed CPU reference.
All timed inference uses MiniTorch's kernels and KV cache.
"""
import argparse
import gc
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import minitorch as mt
from minitorch.kernel_config import KernelConfig,make_backend
from minitorch.kv_cache import greedy_token
from minitorch.pretrained_gpt2 import load_pretrained,from_gpt2_state
from minitorch.training_precision import TrainingPrecision
from minitorch.profiling import CsvMetrics,measure,memory_snapshot,synchronize
from minitorch.inference_metrics import cache_memory


PROMPT_LENGTHS=(32,128,256,512)
BENCHMARK_TEXT=(
    'Artificial intelligence helps scientists study the natural world. '
    'Researchers collect observations, compare explanations, and test their predictions. '
    'Reliable experiments require clear methods and careful measurements. '
    'Computers can analyze large collections of data, while people interpret the results '
    'and decide which questions to investigate next. '
)


def build_prompt_cases(tokenizer,batch_size):
    """Fixed prose repeated and token-truncated to exact, reproducible lengths."""
    cases=[]
    for length in PROMPT_LENGTHS:
        text=BENCHMARK_TEXT
        tokens=tokenizer.encode(text,add_special_tokens=False)
        if not tokens: raise ValueError('Built-in benchmark text tokenization is empty')
        while len(tokens)<length:
            text+=BENCHMARK_TEXT
            extended=tokenizer.encode(text,add_special_tokens=False)
            if len(extended)<=len(tokens):
                raise ValueError('Tokenizer must produce longer input for repeated benchmark text')
            tokens=extended
        prompt=encode_prompt(tokenizer,text,batch_size,length)
        assert prompt.shape==(batch_size,length)
        cases.append(dict(prompt_tokens=length,source_text=text,input_ids=prompt))
    return cases


def encode_prompt(tokenizer,text,batch_size,max_tokens=None):
    tokens=tokenizer.encode(text,add_special_tokens=False)
    if max_tokens is not None: tokens=tokens[:max_tokens]
    if not tokens: raise ValueError('Prompt tokenization is empty')
    return np.repeat(np.asarray(tokens,dtype=np.float32)[None,:],batch_size,axis=0)


def validate_logits(reference,model,prompt,precision,*,cache_impl='contiguous',page_size=16,kv_budget_bytes=256*1024**2,paged_prefill_impl='flash'):
    """CPU HF vs MiniTorch FP32; then prefill/decode parity at requested precision."""
    import torch
    with torch.inference_mode():
        expected=reference(torch.tensor(prompt.astype(np.int64)),use_cache=False).logits[:,-1:,:].float().numpy()
    with mt.no_grad():
        ids=mt.tensor_from_numpy(prompt,model.backend)
        fp32=model(ids,last_only=True).to_numpy()
        np.testing.assert_allclose(fp32,expected,rtol=2e-3,atol=3e-3,
            err_msg='Pretrained mapping differs from Transformers; benchmark stopped')
        with TrainingPrecision(precision,'device').context():
            full=model(ids,last_only=True)
            cache=model.create_kv_cache(prompt.shape[0],model.n_positions,cache_impl=cache_impl,
                page_size=page_size,kv_budget_bytes=kv_budget_bytes,paged_prefill_impl=paged_prefill_impl)
            try:
                cached,cache=model.prefill(ids,cache)
                tol={'fp32':.003,'fp16':.03,'bf16':.15}[precision]
                np.testing.assert_allclose(cached.to_numpy(),full.to_numpy(),rtol=tol,atol=tol)
                if prompt.shape[1]<model.n_positions:
                    token=greedy_token(full)
                    expanded=np.concatenate([prompt,token.to_numpy()],axis=1)
                    step=model.decode_step(token,cache)
                    full_step=model(mt.tensor_from_numpy(expanded,model.backend),last_only=True)
                    np.testing.assert_allclose(step.to_numpy(),full_step.to_numpy(),rtol=tol,atol=tol)
            finally: cache.close()
    return dict(passed=True,hf_fp32_max_absolute_error=float(np.max(np.abs(fp32-expected))),
        hf_fp32_last_token_argmax_equal=bool(np.array_equal(fp32.argmax(-1),expected.argmax(-1))),
        cache_prefill_checked=True,cache_decode_checked=prompt.shape[1]<model.n_positions,precision=precision)


def run_request(model,tokenizer,text,policy,*,cached=None,mode=None,batch_size,new_tokens,max_prompt_tokens,repeat,warmup,
                page_size=16,kv_budget_bytes=256*1024**2,pages_per_slab=4,paged_prefill_impl='flash'):
    """TTFT starts before tokenization/cache allocation and ends at first CPU IDs."""
    from minitorch.cuda_storage import runtime
    mode=mode or ('cached' if cached else 'full_prefix')
    if mode not in ('full_prefix','cached','paged'): raise ValueError('Unknown inference mode')
    if new_tokens<1 or batch_size<1: raise ValueError('Positive token count and batch size required')
    cached=mode!='full_prefix'
    b=model.backend;synchronize(b);baseline=memory_snapshot(b);runtime().reset_peaks()
    start=time.perf_counter()
    history=encode_prompt(tokenizer,text,batch_size,max_prompt_tokens)
    prompt_length=history.shape[1]
    if prompt_length+new_tokens-1>model.n_positions:
        raise ValueError('Prompt + new_tokens - 1 exceeds model position capacity')
    rows=[];outputs=[];cache=None;next_ids=None;setup=0.;first=None;previous=start
    try:
        with mt.no_grad(),policy.context():
            if cached:
                cache,setup=measure(b,lambda:model.create_kv_cache(batch_size,prompt_length+new_tokens-1,
                    cache_impl='paged' if mode=='paged' else 'contiguous',page_size=page_size,
                    kv_budget_bytes=kv_budget_bytes,pages_per_slab=pages_per_slab,paged_prefill_impl=paged_prefill_impl))
            for step in range(new_tokens):
                before=memory_snapshot(b)
                idx=next_ids if cached and step else mt.tensor_from_numpy(history,b)
                def forward():
                    if not cached: return model(idx,last_only=True)
                    if step==0: return model.prefill(idx,cache)[0]
                    return model.decode_step(idx,cache)
                logits,forward_seconds=measure(b,forward)
                next_ids=greedy_token(logits)
                host_ids=next_ids.to_numpy()
                arrival=time.perf_counter()
                if first is None: first=arrival
                after=memory_snapshot(b)
                rows.append(dict(mode=mode,prompt_tokens=prompt_length,repeat=repeat,is_warmup=int(warmup),step=step,
                    phase='prefill' if step==0 else 'decode',context_length=prompt_length+step,
                    attention_path=cache.last_attention_path if cache else 'full_prefix_flash',
                    forward_seconds=forward_seconds,inter_token_seconds=arrival-previous,
                    cache_setup_seconds=setup if step==0 else 0.,cache_bytes=cache.nbytes if cache else 0,
                    h2d_bytes=after['h2d_bytes']-before['h2d_bytes'],d2h_bytes=after['d2h_bytes']-before['d2h_bytes'],
                    gpu_active_bytes=after['active_bytes'],gpu_reserved_bytes=after['reserved_bytes'],
                    tensor_pool_idle_bytes=after['cached_bytes'],**cache_memory(cache)))
                previous=arrival;outputs.append(host_ids[:,0].astype(int).tolist())
                if not cached: history=np.concatenate([history,host_ids],axis=1)
                del logits,idx
            elapsed=arrival-start;decode=arrival-first
            request=dict(mode=mode,repeat=repeat,is_warmup=int(warmup),batch_size=batch_size,
                prompt_tokens=prompt_length,generated_tokens=batch_size*new_tokens,
                ttft_seconds=first-start,request_seconds=elapsed,decode_seconds=decode,
                cache_setup_seconds=setup,prefill_forward_seconds=rows[0]['forward_seconds'],
                prefill_attention_path=rows[0]['attention_path'],
                decode_token_p50_ms=1000*float(np.median([r['inter_token_seconds'] for r in rows[1:]])) if new_tokens>1 else None,
                decode_token_p95_ms=1000*float(np.percentile([r['inter_token_seconds'] for r in rows[1:]],95)) if new_tokens>1 else None,
                decode_tokens=batch_size*(new_tokens-1),output_tokens_per_sec=batch_size*new_tokens/elapsed,
                decode_tokens_per_sec=batch_size*(new_tokens-1)/decode if new_tokens>1 else None,
                peak_active_bytes=runtime().stats['peak_active_bytes'],
                peak_reserved_bytes=runtime().stats['peak_reserved_bytes'],
                baseline_active_bytes=baseline['active_bytes'],baseline_reserved_bytes=baseline['reserved_bytes'],
                incremental_peak_active_bytes=runtime().stats['peak_active_bytes']-baseline['active_bytes'],
                peak_kv_live_bytes=max(r['kv_live_bytes'] for r in rows),
                peak_kv_capacity_bytes=max(r['kv_capacity_bytes'] for r in rows),
                peak_kv_reserved_bytes=max(r['kv_total_reserved_bytes'] for r in rows),
                final_kv_slot_utilization=rows[-1]['kv_slot_utilization'])
    finally:
        if cache is not None: cache.close()
    next_ids=None
    released=memory_snapshot(b)
    request.update(post_release_active_bytes=released['active_bytes'],post_release_reserved_bytes=released['reserved_bytes'])
    sequences=np.asarray(outputs).T.tolist()
    return request,rows,dict(mode=mode,prompt_tokens=prompt_length,repeat=repeat,is_warmup=warmup,token_ids=sequences,
        generated_text=[tokenizer.decode(s,skip_special_tokens=True) for s in sequences])


def summarize(requests):
    result={}
    for mode in ('full_prefix','cached','paged'):
        rows=[r for r in requests if r['mode']==mode and not r['is_warmup']]
        if not rows: continue
        ttft=[r['ttft_seconds'] for r in rows]
        seconds=sum(r['request_seconds'] for r in rows);decode=sum(r['decode_seconds'] for r in rows)
        result[mode]=dict(request_count=len(rows),median_ttft_ms=1000*statistics.median(ttft),
            p95_ttft_ms=1000*float(np.percentile(ttft,95)),
            output_tokens_per_sec=sum(r['generated_tokens'] for r in rows)/seconds,
            decode_tokens_per_sec=sum(r['decode_tokens'] for r in rows)/decode if decode>0 else None,
            mean_request_seconds=seconds/len(rows),peak_active_bytes=max(r['peak_active_bytes'] for r in rows))
        for key in ('peak_reserved_bytes','incremental_peak_active_bytes','peak_kv_live_bytes','peak_kv_capacity_bytes','peak_kv_reserved_bytes'):
            result[mode][key]=max(r.get(key,0) for r in rows)
        values=[r.get('final_kv_slot_utilization') for r in rows if r.get('final_kv_slot_utilization') is not None]
        result[mode]['mean_final_kv_slot_utilization']=statistics.mean(values) if values else None
        for field in ('cache_setup_seconds','prefill_forward_seconds'):
            values=[r[field] for r in rows if field in r]
            result[mode]['median_'+field]=statistics.median(values) if values else None
    for target,base in (('cached','full_prefix'),('paged','cached'),('paged','full_prefix')):
        if target in result and base in result:
            result[f'{target}_vs_{base}']=dict(
                ttft_reduction_percent=100*(1-result[target]['median_ttft_ms']/result[base]['median_ttft_ms']),
                output_throughput_ratio=result[target]['output_tokens_per_sec']/result[base]['output_tokens_per_sec'])
    return result


def summarize_suite(requests,generated,validation):
    """Never mix different input lengths into a single throughput figure."""
    by_length={}
    for length in PROMPT_LENGTHS:
        rows=[r for r in requests if r['prompt_tokens']==length]
        result=summarize(rows)
        measured=[g for g in generated if g['prompt_tokens']==length and not g['is_warmup']]
        matches=[]
        for repeat in sorted({g['repeat'] for g in measured}):
            pair={g['mode']:g['token_ids'] for g in measured if g['repeat']==repeat}
            matches.append(len(pair)>1 and all(ids==next(iter(pair.values())) for ids in pair.values()))
        result['all_measured_greedy_sequences_identical']=all(matches) if matches and len({g['mode'] for g in measured})>1 else None
        result['validation']=validation[str(length)]
        by_length[str(length)]=result
    return dict(prompt_lengths=list(PROMPT_LENGTHS),by_prompt_tokens=by_length)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='distilbert/distilgpt2',help='GPT-2-family HF repository or local directory')
    p.add_argument('--revision',default='main')
    p.add_argument('--cache-dir',default=None)
    p.add_argument('--local-files-only',action='store_true')
    p.add_argument('--new-tokens',type=int,default=32)
    p.add_argument('--batch-size',type=int,default=1,help='Repeat the same unpadded prompt')
    p.add_argument('--precision',choices=['fp32','bf16','fp16'],default='fp32')
    p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--output-dir',default=None)
    p.add_argument('--modes',nargs='+',choices=['full_prefix','cached','paged'],default=['full_prefix','cached','paged'])
    p.add_argument('--page-size',type=int,choices=[8,16,32],default=16)
    p.add_argument('--kv-budget-mb',type=float,default=256,help='Paged KV slab payload budget (MiB); metadata reported separately')
    p.add_argument('--pages-per-slab',type=int,default=4)
    p.add_argument('--paged-prefill-impl',choices=['flash','paged_reference'],default='flash')
    p.add_argument('--no-plots',action='store_true')
    p.add_argument('--worker-mode',choices=['full_prefix','cached','paged'],help=argparse.SUPPRESS)
    args=p.parse_args()
    if min(args.new_tokens,args.batch_size,args.repeats)<1 or args.warmup<0:
        p.error('positive sizes/repeats and nonnegative warmup required')
    if not np.isfinite(args.kv_budget_mb) or args.kv_budget_mb<=0 or args.pages_per_slab<1:
        p.error('positive finite KV budget and pages_per_slab required')
    if len(set(args.modes))!=len(args.modes): p.error('duplicate modes')
    output=Path(args.output_dir or f'output/pretrained_inference_{time.time_ns()}');output.mkdir(parents=True,exist_ok=False)
    if args.worker_mode is None:
        requests=[];generated=[];steps=[];validation={str(n):{} for n in PROMPT_LENGTHS}
        for mode in args.modes:
            child=output/mode
            subprocess.run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],
                '--worker-mode',mode,'--output-dir',str(child),'--no-plots'],check=True)
            payload=json.loads((child/'raw.json').read_text(encoding='utf-8'))
            requests.extend(payload['requests']);steps.extend(payload['steps']);generated.extend(payload['generated'])
            for length,value in payload['validation'].items(): validation[length][mode]=value
        write_results(output,requests,steps,generated,validation)
        (output/'config.json').write_text(json.dumps(dict(**vars(args),isolated_process_per_mode=True,
            worker_configs={m:str(Path(m)/'config.json') for m in args.modes}),indent=2),encoding='utf-8')
        if not args.no_plots:
            from project.inference_plots import plot_inference
            plot_inference(output)
        print(f'Results: {output.resolve()}',flush=True)
        return
    random.seed(11111);np.random.seed(11111)
    print(f'Loading {args.model} on CPU; adapting pretrained weights to MiniTorch...',flush=True)
    reference,tokenizer=load_pretrained(args.model,args.revision,args.cache_dir,args.local_files_only)
    cases=build_prompt_cases(tokenizer,args.batch_size)
    if max(PROMPT_LENGTHS)+args.new_tokens-1>reference.config.n_positions:
        raise ValueError('512 + new_tokens - 1 exceeds checkpoint position capacity')
    cfg=KernelConfig(storage_mode='device',memory_mode='planned',attention_impl='flash',use_kv_cache=True)
    b=make_backend(cfg);model=from_gpt2_state(reference.config,reference.state_dict(),b)
    validation={}
    for case in cases:
        length=case['prompt_tokens']
        print(f'Validating {length}-token input...',flush=True)
        validation[str(length)]=validate_logits(reference,model,case['input_ids'],args.precision,
            cache_impl='paged' if args.worker_mode=='paged' else 'contiguous',page_size=args.page_size,
            kv_budget_bytes=int(args.kv_budget_mb*1024**2),paged_prefill_impl=args.paged_prefill_impl)
    (output/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    try:
        hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv'],text=True)
    except (OSError,subprocess.CalledProcessError): hardware='nvidia-smi unavailable'
    import torch,transformers
    config=dict(**vars(args),prompt_lengths=list(PROMPT_LENGTHS),execution=cfg.to_dict(),
        prompt_source='built-in repeated English prose, truncated by token ID to exact length',
        prompts=[dict(prompt_tokens=c['prompt_tokens'],source_text=c['source_text'],
            token_ids=c['input_ids'][0].astype(int).tolist(),
            input_text=tokenizer.decode(c['input_ids'][0].astype(int).tolist(),skip_special_tokens=False)) for c in cases],
        resolved_revision=getattr(reference.config,'_commit_hash',None),model_config=reference.config.to_dict(),
        parameter_count=sum(p.numel() for p in reference.parameters()),engine='minitorch',
        torch_version=torch.__version__,transformers_version=transformers.__version__,python=sys.version,gpu=hardware,
        eos_policy='fixed new_tokens; EOS does not stop generation',
        ttft_scope='built-in source tokenization/truncation, cache allocation, H2D, forward, argmax, first IDs on CPU; model load/validation excluded',
        throughput_scope='output tokens / whole request time, including TTFT; text rendering and file logging excluded',
        decode_scope='tokens after first / time from first arrival to final arrival',
        batch_semantics='identical unpadded prompts; sequential batches, aggregate batch throughput',
        memory_scope='MiniTorch tracked CUDA allocations only, not CUDA context/library/driver memory; active includes bucket rounding; reserved includes idle tensor pool and free KV pages',
        kv_scope='full_prefix KV utilization is N/A; paged reserved includes all slabs and tables; cached capacity knows final length',
        isolation='one process per mode; clear idle tensor pool/workspace before each input length; warmups retained within each length')
    (output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    del reference;gc.collect()
    policy=TrainingPrecision(args.precision,'device');requests=[];generated=[];all_steps=[]
    with CsvMetrics(output/'steps.csv') as step_csv,CsvMetrics(output/'requests.csv') as request_csv:
        for case in cases:
            length=case['prompt_tokens']
            clear_idle(b)
            for repeat in range(args.warmup+args.repeats):
                for mode in [args.worker_mode]:
                    request,steps,tokens=run_request(model,tokenizer,case['source_text'],policy,mode=mode,batch_size=args.batch_size,
                        new_tokens=args.new_tokens,max_prompt_tokens=length,repeat=repeat,warmup=repeat<args.warmup,
                        page_size=args.page_size,kv_budget_bytes=int(args.kv_budget_mb*1024**2),pages_per_slab=args.pages_per_slab,
                        paged_prefill_impl=args.paged_prefill_impl)
                    requests.append(request);generated.append(tokens);request_csv.log(request)
                    all_steps.extend(steps)
                    for row in steps: step_csv.log(row)
                    print(f"prompt={length} {request['mode']} repeat={repeat}: TTFT={request['ttft_seconds']*1000:.2f} ms, throughput={request['output_tokens_per_sec']:.2f} tokens/s",flush=True)
    summary=summarize_suite(requests,generated,validation)
    (output/'generations.json').write_text(json.dumps(generated,indent=2,ensure_ascii=False),encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (output/'raw.json').write_text(json.dumps(dict(requests=requests,steps=all_steps,generated=generated,validation=validation),indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2));print(f'Results: {output.resolve()}')


def clear_idle(backend):
    synchronize(backend);gc.collect()
    if getattr(backend,'workspace',None) is not None: backend.workspace.clear()
    if getattr(backend,'memory_pool',None) is not None: backend.memory_pool.empty_cache()


def write_results(output,requests,steps,generated,validation):
    for name,rows in (('steps',steps),('requests',requests)):
        with CsvMetrics(output/f'{name}.csv') as writer:
            for row in rows: writer.log(row)
    summary=summarize_suite(requests,generated,validation)
    for name,data in (('summary',summary),('generations',generated),('validation',validation),
                      ('raw',dict(requests=requests,steps=steps,generated=generated,validation=validation))):
        (output/f'{name}.json').write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')


if __name__=='__main__': main()
