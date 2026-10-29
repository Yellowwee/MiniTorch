"""Run matched FP32/BF16/FP16 synthetic training and summarize unsampled CSVs.

Each precision uses a fresh subprocess on the same GPU, model, data and seed.
This measures this implementation; a low precision run is not assumed faster.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time


def summarize(path):
    with open(path, newline='', encoding='utf-8') as stream:
        all_rows = list(csv.DictReader(stream))
    rows = [r for r in all_rows if not int(r['is_warmup'])]
    if not rows:
        raise ValueError('No measured steps')
    if any(not math.isfinite(float(r['loss'])) for r in all_rows):
        raise ValueError('Nonfinite training loss')
    seconds = sum(float(r['step_seconds']) for r in rows)
    if seconds <= 0:
        raise ValueError('Invalid elapsed time')
    skipped = sum(int(r['amp/overflow']) for r in rows)
    return dict(measured_steps=len(rows), skipped_steps=skipped,
        warmup_skipped_steps=sum(int(r['amp/overflow']) for r in all_rows if int(r['is_warmup'])),
        tokens_per_sec=sum(int(r['tokens']) for r in rows)/seconds,
        successful_update_tokens_per_sec=sum(int(r['tokens'])*int(r['train/update_applied']) for r in rows)/seconds,
        mean_step_seconds=seconds/len(rows), median_step_seconds=statistics.median(float(r['step_seconds']) for r in rows),
        mean_forward_seconds=statistics.mean(float(r['forward_seconds']) for r in rows),
        mean_backward_seconds=statistics.mean(float(r['backward_seconds']) for r in rows),
        peak_active_bytes=max(int(r.get('peak_active_bytes',0)) for r in rows),
        peak_reserved_bytes=max(int(r.get('peak_reserved_bytes',0)) for r in rows),
        first_loss=float(all_rows[0]['loss']), last_loss=float(rows[-1]['loss']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--implementation', choices=['fused','flash'], default='flash')
    parser.add_argument('--length', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--vocab', type=int, default=256)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--seed', type=int, default=11111)
    parser.add_argument('--loss-scale', type=float, default=65536.)
    args = parser.parse_args()
    output = Path(args.output_dir or f'output/precision_{time.time_ns()}')
    output.mkdir(parents=True, exist_ok=False)
    try:
        gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,clocks.sm',
            '--format=csv'], text=True)
    except (OSError, subprocess.CalledProcessError):
        gpu = 'nvidia-smi unavailable'
    (output/'environment.json').write_text(json.dumps(dict(arguments=vars(args),gpu=gpu,
        python=sys.version,executable=sys.executable),indent=2),encoding='utf-8')
    results = {}
    for precision in ('fp32','bf16','fp16'):
        path = output/f'{precision}.csv'
        command = [sys.executable,str(Path(__file__).with_name('benchmark_execution.py')),
            '--kind','train','--storage-mode','device','--memory-mode','planned',
            '--optimizer-impl','fused_adam','--implementation',args.implementation,
            '--precision',precision,'--csv',str(path)]
        for flag, value in [('lengths',args.length),('batch-size',args.batch_size),
            ('hidden',args.hidden),('heads',args.heads),('vocab',args.vocab),
            ('steps',args.steps),('warmup',args.warmup),('seed',args.seed),('loss-scale',args.loss_scale)]:
            command.extend(['--'+flag,str(value)])
        subprocess.run(command,check=True)
        results[precision] = summarize(path)
    base = results['fp32']['tokens_per_sec']
    for result in results.values():
        result['observed_throughput_ratio_vs_fp32'] = result['tokens_per_sec']/base
        result['final_loss_difference_vs_fp32'] = result['last_loss']-results['fp32']['last_loss']
        # Skipped updates are not equivalent training work, including during warmup.
        result['comparable_updates'] = not (result['skipped_steps'] or result['warmup_skipped_steps'])
    (output/'summary.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(results,indent=2))
    print(f'Results: {output.resolve()}')


if __name__ == '__main__':
    main()
