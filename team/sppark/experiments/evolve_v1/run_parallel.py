#!/usr/bin/env python
"""One-command fan-out for evolve_v1's `suite`: splits the requested (or remaining/incomplete)
experiment list into one contiguous shard per GPU, launches `python run.py suite --experiments
<shard>` as a subprocess pinned to each GPU via CUDA_VISIBLE_DEVICES, and streams every
subprocess's output into this process's own stdout with a [gpuN] tag -- so a single `nohup ...
run_parallel.py > one.log 2>&1 &` gives one combined, tail-able log instead of N separate files.

This file lives outside evolvelab/ and vendor/ on purpose: evoacclab... er, evolvelab's
workspace.source_hash() only hashes run.py, configs/base.yaml, evolvelab/*.py and
vendor/alphase/**/*.py, so adding this script never changes that hash and never invalidates the
provenance check on already-trained checkpoints (evolvelab/evaluation.py::load_run)."""
import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def catalog_names():
    sys.path[:0] = [str(ROOT), str(ROOT / 'vendor')]
    from evolvelab.catalog import catalog
    return list(catalog())


def remaining_names():
    status_path = ROOT / 'outputs/suite_status.json'
    done = set()
    if status_path.is_file():
        done = set(json.loads(status_path.read_text()).get('completed', []))
    names = [n for n in catalog_names() if n not in done]
    if not names:
        raise SystemExit('Nothing to do: every experiment is already in outputs/suite_status.json completed.')
    return names


def auto_gpus(max_used_mib=200, max_util_pct=5):
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f'--gpus auto requires a working nvidia-smi: {exc}')
    idle = []
    for line in out.strip().splitlines():
        idx, used, util = (v.strip() for v in line.split(','))
        if int(used) <= max_used_mib and int(util) <= max_util_pct:
            idle.append(idx)
    if not idle:
        raise SystemExit('--gpus auto found no idle GPU (nothing under the memory/utilization threshold).')
    return idle


def chunk(items, n):
    n = max(1, min(n, len(items)))
    size = -(-len(items) // n)  # ceil division
    return [items[i:i + size] for i in range(0, len(items), size)]


def pump(pipe, tag, lock):
    for line in iter(pipe.readline, ''):
        with lock:
            sys.stdout.write(f'[{tag}] {line}')
            sys.stdout.flush()
    pipe.close()


def main():
    p = argparse.ArgumentParser(description='Fan evolve_v1 suite out across multiple GPUs from a single process/command')
    p.add_argument('--gpus', default='auto', help='Comma-separated CUDA device indices (e.g. 0,1,2,5,6,7), or "auto" to pick currently idle GPUs via nvidia-smi (default)')
    p.add_argument('--experiments', nargs='+', default=None, help='Experiment names to run; default: everything not yet in outputs/suite_status.json completed list')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()

    gpus = auto_gpus() if args.gpus == 'auto' else [g.strip() for g in args.gpus.split(',') if g.strip()]
    names = args.experiments or remaining_names()
    shards = [s for s in chunk(names, len(gpus)) if s]
    gpus = gpus[:len(shards)]

    print(f'{len(names)} experiment(s) -> {len(shards)} shard(s) across GPUs {gpus}', flush=True)
    lock = threading.Lock()
    procs, threads = [], []
    for gpu, shard in zip(gpus, shards):
        cmd = [sys.executable, '-B', str(ROOT / 'run.py'), 'suite', '--experiments', *shard]
        if args.smoke:
            cmd.append('--smoke')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED='1')
        print(f'[gpu{gpu}] {len(shard)} experiment(s): {shard}', flush=True)
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
        procs.append((gpu, proc))
        t = threading.Thread(target=pump, args=(proc.stdout, f'gpu{gpu}', lock), daemon=True)
        t.start()
        threads.append(t)

    exit_codes = {gpu: proc.wait() for gpu, proc in procs}
    for t in threads:
        t.join(timeout=5)

    print('All shards finished:', exit_codes, flush=True)
    if any(code != 0 for code in exit_codes.values()):
        sys.exit(1)


if __name__ == '__main__':
    main()
