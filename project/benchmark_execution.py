"""Offline reproducible operator/training benchmark; no dataset or W&B needed."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import platform
import random
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import minitorch as mt
from minitorch.kernel_config import KernelConfig, configure_kernels, make_backend, make_optimizer
from minitorch.profiling import CsvMetrics, measure, memory_snapshot, synchronize
from minitorch.training_precision import TrainingPrecision


def native_seconds(kind, x, grad, g, b, repeats):
    from minitorch.cuda_storage import runtime
    from minitorch.cuda_resident_ops import CudaResidentOps as Ops, pointer, call, P, I
    r=runtime()
    output=ct.c_float()
    if kind.startswith('softmax'):
        mask=x.zeros((x.shape[0],x.shape[-1]))
        y=Ops.attn_softmax_fw(x,mask)
        out=Ops.empty(x.shape,x.backend)
        lib=r.softmax
        if kind.endswith('fw'):
            signature=[P,P,P,I,I,I,I,ct.c_bool]
            args=[pointer(out),pointer(x),pointer(mask),*x.shape,False]
        else:
            signature=[P,P,P,I,I]
            args=[pointer(out),pointer(grad),pointer(y),x.size//x.shape[-1],x.shape[-1]]
    else:
        y,var,mean=Ops.layernorm_fw(x,g,b)
        lib=r.layernorm
        if kind.endswith('fw'):
            signature=[P]*6+[I,I]
            args=[pointer(t) for t in (y,var,mean,x,g,b)]+[x.size//x.shape[-1],x.shape[-1]]
        else:
            dx,dg,db=(Ops.empty(t.shape,t.backend) for t in (x,g,b))
            signature=[P]*9+[I,I]
            args=[pointer(t) for t in (dg,db,dx,grad,x,g,b,var,mean)]+[x.size//x.shape[-1],x.shape[-1]]
    call(lib,'mt_bench_'+kind,[ct.POINTER(ct.c_float),I]+signature,ct.byref(output),repeats,*args)
    return output.value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--storage-mode',choices=['host','device'],default='device')
    p.add_argument('--implementation',choices=['reference','fused','flash'],default='fused')
    p.add_argument('--memory-mode',choices=['default','pooled','planned'],default='default')
    p.add_argument('--pool-cache-mb',type=int,default=1024)
    p.add_argument('--optimizer-impl',choices=['adam','fused_adam'],default='adam')
    p.add_argument('--kind',choices=['softmax','layernorm','train'],default='softmax')
    p.add_argument('--lengths',type=int,nargs='+',default=[39,128,512,1024])
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--heads',type=int,default=2)
    p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--vocab',type=int,default=256)
    p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--steps',type=int,default=10)
    p.add_argument('--seed',type=int,default=11111)
    p.add_argument('--csv',default=None)
    p.add_argument('--precision',choices=['fp32','bf16','fp16'],default='fp32')
    p.add_argument('--loss-scale',type=float,default=65536.)
    args=p.parse_args()
    if args.steps<1 or args.warmup<0 or min(args.lengths)<1:
        p.error('positive steps/lengths and nonnegative warmup required')
    if args.implementation=='flash' and args.kind!='train':
        p.error('flash is whole attention; use --kind train or benchmark_flash_attention.py')
    if args.optimizer_impl != 'adam' and args.kind != 'train':
        p.error('optimizer-impl applies to --kind train')
    if args.precision != 'fp32' and args.kind != 'train':
        p.error('AMP benchmark currently applies to --kind train')
    TrainingPrecision(args.precision,args.storage_mode,args.loss_scale)
    execution=KernelConfig.resolve(args.implementation!='reference',storage_mode=args.storage_mode,
        attention_impl='flash' if args.implementation=='flash' else None,
        memory_mode=args.memory_mode,pool_cache_mb=args.pool_cache_mb,optimizer_impl=args.optimizer_impl)
    backend=make_backend(execution)
    path=Path(args.csv or f'output/benchmark_{args.kind}_{args.storage_mode}_{time.time_ns()}.csv')
    path.parent.mkdir(parents=True,exist_ok=True)
    config={**vars(args),**execution.to_dict(),'python':sys.version,'platform':platform.platform(),
            'numpy':np.__version__,'gpu_native_scope':'preallocated native launch stream interval, includes D2D; excludes Python and allocation',
            'api_scope':'synchronized wall interval, includes Python, output allocation and autograd',
            'token_count':'padded input tokens; aggregate throughput=sum(tokens)/sum(step_seconds)'}
    if path.exists():
        raise FileExistsError(path)
    with path.with_suffix('.config.json').open('x',encoding='utf-8') as stream:
        json.dump(config,stream,indent=2)
    with CsvMetrics(path) as csv:
        for length in args.lengths:
            policy=TrainingPrecision(args.precision,args.storage_mode,args.loss_scale)
            np.random.seed(args.seed); random.seed(args.seed)
            def tensor(x,grad=False):
                return mt.tensor_from_numpy(np.asarray(x,dtype=np.float32),backend,grad)
            if args.kind=='train':
                model=mt.DecoderLM(n_vocab=args.vocab,n_embd=args.hidden,n_head=args.heads,
                    n_positions=length,p_dropout=0,ln_eps=1e-5,backend=backend)
                configure_kernels(model,execution)
                optimizer=make_optimizer(model.parameters(),execution,lr=0.001)
                ids=tensor(np.random.randint(args.vocab,size=(args.batch_size,length)))
                labels=tensor(np.random.randint(args.vocab,size=(args.batch_size*length,)))
                def forward():
                    with policy.context():
                        return mt.nn.softmax_loss(model(ids).view(args.batch_size*length,args.vocab),labels).mean()
                native_fw=native_bw=None
            else:
                shape=(args.batch_size,args.heads,length,length) if args.kind=='softmax' else (args.batch_size*length,args.hidden)
                x=tensor(np.random.randn(*shape),True)
                grad=tensor(np.random.randn(*shape))
                g=tensor(np.ones(args.hidden),True); b=tensor(np.zeros(args.hidden),True)
                mask=x.zeros((args.batch_size,length))
                def forward():
                    if args.kind=='softmax':
                        return x.attn_softmax(mask) if args.implementation=='fused' else mt.nn.softmax(x,3)
                    if args.implementation=='fused':
                        return x.layernorm(g,b)
                    center=x-x.mean(1)
                    return center/((center**2).mean(1)+1e-5)**0.5*g+b
                native_fw=native_bw=None
                if args.storage_mode=='device' and args.implementation=='fused':
                    # Warm up native launches before recording the repeated interval.
                    native_seconds(args.kind+'_fw',x,grad,g,b,3)
                    native_seconds(args.kind+'_bw',x,grad,g,b,3)
                    native_fw=native_seconds(args.kind+'_fw',x,grad,g,b,100)
                    native_bw=native_seconds(args.kind+'_bw',x,grad,g,b,100)
            for step in range(args.warmup+args.steps):
                update_launches_before = getattr(optimizer, 'kernel_launches', 0) if args.kind=='train' else 0
                if args.kind=='train': optimizer.zero_grad()
                else:
                    for t in (x,g,b): t.zero_grad_()
                synchronize(backend)
                before=memory_snapshot(backend)
                if backend.device_storage:
                    from minitorch.cuda_storage import runtime
                    runtime().reset_peaks()
                start=time.perf_counter()
                out,fw=measure(backend,forward)
                _,bw=measure(backend,lambda: policy.backward(out) if args.kind=='train' else out.backward(grad))
                opt=0.
                update_metrics={}
                if args.kind=='train': update_metrics,opt=measure(backend,lambda: policy.step(optimizer))
                wall=time.perf_counter()-start
                loss_value=float(out.item()) if args.kind=='train' else None
                after=memory_snapshot(backend)
                row=dict(length=length,step=step,is_warmup=int(step<args.warmup),
                    batch_size=args.batch_size,kind=args.kind,storage_mode=args.storage_mode,
                    memory_mode=args.memory_mode,precision=args.precision,
                    optimizer_impl=args.optimizer_impl,
                    implementation=args.implementation,forward_seconds=fw,backward_seconds=bw,
                    optimizer_seconds=opt,step_seconds=wall,
                    loss=loss_value,
                    tokens=args.batch_size*length if args.kind=='train' else None,
                    tokens_per_sec=args.batch_size*length/wall if args.kind=='train' else None,
                    native_forward_seconds=native_fw,native_backward_seconds=native_bw)
                row.update(update_metrics)
                for key,value in after.items():
                    row[key]=value-before.get(key,0) if key in ('allocations','frees','h2d_bytes','d2h_bytes','d2d_bytes','pool_hits','pool_misses','oom_retries') else value
                if args.kind=='train' and hasattr(optimizer,'memory_metrics'):
                    for key,value in optimizer.memory_metrics().items():
                        row[key]=value-update_launches_before if key=='update_kernel_launches' else value
                csv.log(row)
                del out
            print(f'{args.kind}: length={length} complete')
    print(f'Raw benchmark: {path.resolve()}')


if __name__=='__main__':
    main()
