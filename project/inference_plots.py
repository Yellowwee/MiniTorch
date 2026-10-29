"""Static PNG/SVG figures from benchmark raw data; no interpolation of runs."""
import json
from pathlib import Path
import numpy as np

COLORS={'full_prefix':'#7560bf','cached':'#e29b24','paged':'#269d91',
        'cached_known':'#e29b24','cached_upper':'#ca678d'}
MIB=1024**2


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none'})
    return plt


def save(plt,fig,output,name):
    directory=Path(output)/'figures';directory.mkdir(exist_ok=True)
    fig.savefig(directory/f'{name}.png',dpi=220,bbox_inches='tight')
    fig.savefig(directory/f'{name}.svg',bbox_inches='tight')
    plt.close(fig)


def plot_inference(output):
    output=Path(output);plt=plotting()
    from matplotlib.ticker import MaxNLocator
    raw=json.loads((output/'raw.json').read_text(encoding='utf-8'))
    summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))['by_prompt_tokens']
    modes=[m for m in ('full_prefix','cached','paged') if m in next(iter(summary.values()))]
    lengths=sorted(map(int,summary))
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ax,length in zip(axes.flat,lengths):
        for mode in modes:
            rows=[r for r in raw['steps'] if r['mode']==mode and r['prompt_tokens']==length and not r['is_warmup']]
            repeat=min(r['repeat'] for r in rows)
            rows=[r for r in rows if r['repeat']==repeat]
            if mode=='full_prefix': continue
            x=[r['context_length'] for r in rows]
            for field,style,label in (('kv_live_bytes','-','live'),('kv_capacity_bytes','--','assigned'),('kv_total_reserved_bytes',':','reserved incl. tables')):
                ax.plot(x,[r[field]/MIB for r in rows],style,color=COLORS[mode],label=f'{mode}: {label}',
                    drawstyle='default' if field=='kv_live_bytes' else 'steps-post')
        ax.set(title=f'Input: {length} tokens',xlabel='Cached context length (tokens)',ylabel='KV memory (MiB)')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=.2)
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=8)
    fig.suptitle('KV growth: first measured request; full_prefix has no persistent KV cache')
    save(plt,fig,output,'kv_growth')
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    x=np.arange(len(lengths));width=.8/max(1,len(modes))
    for j,mode in enumerate(modes):
        for ax,field,title in zip(axes,('peak_active_bytes','peak_reserved_bytes'),('Peak active allocations','Peak reserved allocations')):
            ax.bar(x+(j-(len(modes)-1)/2)*width,[summary[str(n)][mode][field]/MIB for n in lengths],width,
                   color=COLORS[mode],label=mode)
            ax.set(xticks=x,xticklabels=lengths,xlabel='Input tokens',ylabel='MiB',title=title)
            ax.grid(axis='y',alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3)
    fig.suptitle('MiniTorch tracked GPU memory (model included; CUDA context/library overhead excluded)')
    save(plt,fig,output,'gpu_peak_memory')
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for mode in modes:
        for ax,field,title in zip(axes,('output_tokens_per_sec','decode_tokens_per_sec'),('Whole request (including TTFT)','Decode after first token')):
            values=[summary[str(n)][mode][field] for n in lengths]
            ax.plot(lengths,[v if v is not None else np.nan for v in values],'-o',color=COLORS[mode],label=mode)
            ax.set(xlabel='Input tokens',ylabel='Output tokens/s',title=title,xticks=lengths)
            ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3)
    fig.suptitle('Throughput = total output tokens / total measured time; warmup excluded')
    save(plt,fig,output,'throughput')


def plot_trace(output):
    output=Path(output);plt=plotting()
    results=json.loads((output/'raw.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(len(results),2,figsize=(12,3*len(results)),squeeze=False,layout='constrained')
    for (case,result),(left,right) in zip(results.items(),axes):
        rows=result['events'];x=[r['event_index'] for r in rows]
        for field,style,label in (('kv_live_bytes','-','Live KV'),('kv_capacity_bytes','--','Assigned capacity'),
                                  ('kv_total_reserved_bytes',':','Reserved incl. tables'),('kv_pool_free_bytes','-.','Idle KV pool')):
            left.step(x,[r[field]/MIB for r in rows],where='post',linestyle=style,label=label)
        left.set(title=case,xlabel='Event index',ylabel='KV memory (MiB)');left.grid(alpha=.2)
        if case=='full_prefix': right.text(.5,.5,'KV utilization: N/A',ha='center',transform=right.transAxes)
        else:
            right.step(x,[100*r['kv_slot_utilization'] if r['kv_slot_utilization'] is not None else np.nan for r in rows],
                where='post',color=COLORS[case],label='Live / assigned capacity')
        right.set(xlabel='Event index',ylabel='Slot utilization (%)',ylim=(0,105));right.grid(alpha=.2)
        for ax in (left,right):
            for row in rows:
                if row['event'] in ('finish','reject_admission','reject_growth'):
                    ax.axvline(row['event_index'],color='gray',alpha=.15)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',fontsize=8,ncol=4)
    fig.suptitle('Sequential request-lifetime trace with coexisting caches (not continuous batching)\nVertical marks: completion/rejection; raw events.csv records request IDs')
    save(plt,fig,output,'kv_request_lifetimes')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir',help='Existing benchmark directory containing raw.json and summary.json')
    parser.add_argument('--trace',action='store_true',help='Render a benchmark_kv_memory run')
    args=parser.parse_args()
    (plot_trace if args.trace else plot_inference)(args.output_dir)
