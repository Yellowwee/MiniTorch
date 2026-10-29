"""Matched full-prefix vs KV-cache greedy inference, without remote datasets."""
import argparse
import json
from pathlib import Path
import random
import statistics
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import minitorch as mt
from minitorch.kernel_config import KernelConfig, configure_kernels, make_backend
from minitorch.kv_cache import greedy_token
from minitorch.training_precision import TrainingPrecision
from minitorch.profiling import CsvMetrics, measure, memory_snapshot, synchronize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt-length',type=int,default=32)
    parser.add_argument('--new-tokens',type=int,default=16)
    parser.add_argument('--batch-size',type=int,default=1)
    parser.add_argument('--hidden',type=int,default=64)
    parser.add_argument('--heads',type=int,default=2)
    parser.add_argument('--vocab',type=int,default=256)
    parser.add_argument('--precision',choices=['fp32','bf16','fp16'],default='fp32')
    parser.add_argument('--warmup',type=int,default=1)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output-dir',default=None)
    args=parser.parse_args()
    if min(args.prompt_length,args.new_tokens,args.batch_size,args.hidden,args.heads,args.vocab,args.repeats)<1 or args.warmup<0:
        parser.error('positive sizes/repeats and nonnegative warmup required')
    if args.hidden%args.heads or args.hidden//args.heads>128:
        parser.error('hidden must be divisible by heads, head_dim <=128')
    output=Path(args.output_dir or f'output/inference_{time.time_ns()}')
    output.mkdir(parents=True,exist_ok=False)
    random.seed(11111);np.random.seed(11111)
    cfg=KernelConfig(storage_mode='device',memory_mode='planned',attention_impl='flash',use_kv_cache=True)
    b=make_backend(cfg)
    capacity=args.prompt_length+args.new_tokens-1
    model=mt.DecoderLM(n_vocab=args.vocab,n_embd=args.hidden,n_head=args.heads,
        n_positions=capacity,p_dropout=0,backend=b)
    configure_kernels(model,cfg);model.eval()
    policy=TrainingPrecision(args.precision,'device')
    prompt=np.random.randint(args.vocab,size=(args.batch_size,args.prompt_length)).astype('float32')
    (output/'config.json').write_text(json.dumps(dict(**vars(args),execution=cfg.to_dict(),
        seed=11111,capacity=capacity,scope='synchronized wall time; cache setup separate; token total includes argmax and ID download; no EOS early exit'),indent=2),encoding='utf-8')
    records=[]; generated={}
    with CsvMetrics(output/'steps.csv') as csv,mt.no_grad(),policy.context():
        for cached in (False,True):
            mode='cached' if cached else 'full_prefix'
            for repeat in range(args.warmup+args.repeats):
                b.workspace.clear();b.memory_pool.empty_cache()
                history=prompt.copy();cache=None;next_ids=None;tokens=[]
                setup=0.
                try:
                    if cached:
                        cache,setup=measure(b,lambda:model.create_kv_cache(args.batch_size,capacity))
                    for step in range(args.new_tokens):
                        synchronize(b)
                        before=memory_snapshot(b)
                        from minitorch.cuda_storage import runtime
                        runtime().reset_peaks()
                        start=time.perf_counter()
                        idx=next_ids if cached and step else mt.tensor_from_numpy(history,b)
                        def forward():
                            if not cached: return model(idx,last_only=True)
                            if step==0: return model.prefill(idx,cache)[0]
                            return model.decode_step(idx,cache)
                        logits,seconds=measure(b,forward)
                        next_ids,argmax_seconds=measure(b,lambda:greedy_token(logits))
                        host_ids=next_ids.to_numpy()
                        total=time.perf_counter()-start
                        after=memory_snapshot(b)
                        row=dict(mode=mode,repeat=repeat,is_warmup=int(repeat<args.warmup),
                            step=step,phase='prefill' if step==0 else 'decode',
                            context_length=args.prompt_length+step,forward_seconds=seconds,
                            argmax_seconds=argmax_seconds,token_total_seconds=total,
                            cache_setup_seconds=setup if step==0 else 0.,
                            cache_bytes=cache.nbytes if cache else 0,
                            peak_active_bytes=after['peak_active_bytes'],
                            h2d_bytes=after['h2d_bytes']-before['h2d_bytes'],
                            d2h_bytes=after['d2h_bytes']-before['d2h_bytes'])
                        records.append(row);csv.log(row)
                        tokens.append(host_ids[:,0].astype(int).tolist())
                        if not cached: history=np.concatenate([history,host_ids],axis=1)
                        del logits,idx
                    generated[mode]=tokens
                finally:
                    if cache is not None: cache.close()
    summary={}
    for mode in ('full_prefix','cached'):
        rows=[r for r in records if r['mode']==mode and not r['is_warmup']]
        prefill=[r for r in rows if r['phase']=='prefill']
        decode=[r for r in rows if r['phase']=='decode']
        summary[mode]=dict(
            median_prefill_seconds=statistics.median(r['forward_seconds'] for r in prefill),
            median_decode_seconds=statistics.median(r['forward_seconds'] for r in decode) if decode else None,
            mean_decode_token_total_seconds=statistics.mean(r['token_total_seconds'] for r in decode) if decode else None,
            generated_tokens_per_sec=args.batch_size*len(rows)/sum(r['token_total_seconds']+r['cache_setup_seconds'] for r in rows),
            peak_active_bytes=max(r['peak_active_bytes'] for r in rows),
            cache_bytes=max(r['cache_bytes'] for r in rows))
    summary['identical_greedy_tokens']=generated['cached']==generated['full_prefix']
    (output/'tokens.json').write_text(json.dumps(generated,indent=2),encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2));print(f'Raw inference results: {output.resolve()}')


if __name__=='__main__':
    main()
