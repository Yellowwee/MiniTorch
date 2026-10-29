"""Raw reference/FlashAttention wall-time and owned-memory comparison (FP32)."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import minitorch as mt
from minitorch.kernel_config import KernelConfig,make_backend
from minitorch.cuda_storage import runtime
from minitorch.profiling import CsvMetrics,measure


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lengths',nargs='+',type=int,default=[39,128,512,1024])
    parser.add_argument('--batch-size',type=int,default=1)
    parser.add_argument('--heads',type=int,default=2)
    parser.add_argument('--head-dim',type=int,default=32)
    parser.add_argument('--steps',type=int,default=10)
    parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--causal',action='store_true')
    parser.add_argument('--seed',type=int,default=11111)
    parser.add_argument('--memory-mode',choices=['default','pooled','planned'],default='default')
    parser.add_argument('--pool-cache-mb',type=int,default=1024)
    parser.add_argument('--csv',default=None)
    args=parser.parse_args()
    if min(args.lengths+[args.batch_size,args.heads,args.steps,args.head_dim])<1 or args.head_dim>128 or args.warmup<0:
        parser.error('positive sizes/steps, head-dim<=128, warmup>=0 required')
    backend=make_backend(KernelConfig(attention_impl='flash',storage_mode='device',
        memory_mode=args.memory_mode,pool_cache_mb=args.pool_cache_mb))
    r=runtime()
    path=Path(args.csv or f'output/flash_attention_{time.time_ns()}.csv')
    path.parent.mkdir(parents=True,exist_ok=True)
    config={**vars(args),'dtype':'float32','scope':'synchronized API wall time, including autograd and allocation; no data loading or logging',
            'memory_scope':'owned allocations only, excludes CUDA context and cuBLAS workspace',
            'dropout_p':0,'causal_alignment':'top-left','python':sys.version}
    with CsvMetrics(path) as csv:
        path.with_suffix('.config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        for n in args.lengths:
            rng=np.random.default_rng(args.seed)
            shape=(args.batch_size,args.heads,n,args.head_dim)
            q,k,v=[mt.tensor_from_numpy(rng.normal(size=shape).astype('float32'),backend,True) for _ in range(3)]
            dy=mt.tensor_from_numpy(rng.normal(size=shape).astype('float32'),backend)
            for impl in ('reference','flash'):
                def forward():
                    if impl=='flash': return mt.flash_attention(q,k,v,causal=args.causal)
                    scores=(q@k.permute(0,1,3,2))/(args.head_dim**.5)
                    if args.causal:
                        scores=scores+backend.ops.sequence((1,1,n,n),backend,width=n)
                    return mt.nn.softmax(scores,3)@v
                for step in range(args.warmup+args.steps):
                    for t in (q,k,v): t.zero_grad_()
                    gc.collect();r.synchronize();r.reset_peaks();before=r.snapshot()
                    y,fw=measure(backend,forward)
                    _,bw=measure(backend,lambda: y.backward(dy))
                    after=r.snapshot()
                    csv.log(dict(implementation=impl,length=n,batch_size=args.batch_size,heads=args.heads,
                        memory_mode=args.memory_mode,
                        head_dim=args.head_dim,causal=args.causal,step=step,is_warmup=int(step<args.warmup),
                        forward_seconds=fw,backward_seconds=bw,step_seconds=fw+bw,
                        tokens_per_sec=args.batch_size*n/(fw+bw),
                        peak_active_bytes=after['peak_active_bytes'],
                        incremental_peak_bytes=after['peak_active_bytes']-before['active_bytes'],
                        allocations=after['allocations']-before['allocations'],
                        frees=after['frees']-before['frees'],
                        pool_hits=after['pool_hits']-before['pool_hits'],
                        pool_misses=after['pool_misses']-before['pool_misses'],
                        active_bytes=after['active_bytes'],reserved_bytes=after['reserved_bytes'],
                        cached_bytes=after['cached_bytes'],peak_reserved_bytes=after['peak_reserved_bytes'],
                        h2d_bytes=after['h2d_bytes']-before['h2d_bytes'],
                        d2h_bytes=after['d2h_bytes']-before['d2h_bytes']))
                    del y
                print(f'{impl}: length={n} done')
            del q,k,v,dy
    print(f'Raw benchmark: {path.resolve()}')


if __name__=='__main__':
    main()
