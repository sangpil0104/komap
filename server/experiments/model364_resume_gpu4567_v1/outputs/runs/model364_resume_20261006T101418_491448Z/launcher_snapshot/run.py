"""Copy preserved outputs into a new run and execute only unfinished stages."""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm
from scheduler import render, run_tasks, validate_tasks

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def inspect(recipe):
    if Path(sys.prefix).name != 'komap' or os.environ.get('CONDA_DEFAULT_ENV') != 'komap':
        raise RuntimeError('Use run.sh in conda komap')
    validate_tasks(recipe['tasks'])
    for row in json.loads((ROOT / 'source_manifest.json').read_text()):
        if hashlib.sha256((ROOT / row['path']).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'Source changed: {row["path"]}')
    parent = Path(recipe['parent_run'])
    status = json.loads((parent / 'status.json').read_text())
    if status['phase'] != 'failed_or_interrupted':
        raise ValueError('Parent must be stopped before resume')
    if set(status['completed']) != set(recipe['completed_reused']):
        raise ValueError('Parent completion status changed')
    full = json.loads((parent / 'launcher_snapshot/pipeline_tasks.json').read_text())
    for task in full['tasks']:
        if task['id'] not in status['completed']:
            continue
        for name in render(task['produces'], parent / 'source_snapshot', parent, 4):
            p = Path(name)
            if not p.exists() or (p.is_file() and p.stat().st_size == 0):
                raise ValueError(f'Completed artifact missing: {p}')
    for split, count in [('train', 70), ('valid', 20)]:
        if len(list((parent / 'data' / split / 'images').glob('*_image.png'))) != count:
            raise ValueError(f'Parent {split} data count changed')
    evaluation = next(t for t in recipe['tasks'] if t['id'] == 'evaluate_model361')['argv']
    size = int(evaluation[evaluation.index('--patch-size') + 1])
    scale = float(evaluation[evaluation.index('--input-scale') + 1])
    overlap = int(evaluation[evaluation.index('--overlap') + 1])
    native = round(size / scale)
    if native - overlap < max(1, native // 10):
        raise ValueError('Invalid Model361 tiling')
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                         '--format=csv,noheader'], text=True)
    devices = {int(line.split(',')[0]): line.split(',')[1].strip()
               for line in inventory.splitlines()}
    if not {4, 5, 6, 7} <= devices.keys():
        raise ValueError('GPUs 4,5,6,7 are required')
    print(json.dumps({'completed_reused': len(status['completed']),
                      'remaining_stages': len(recipe['tasks']), 'gpus': [4, 5, 6, 7],
                      'model361_native_tile': native, 'overlap': overlap,
                      'stride': native - overlap, 'training_started': False}, indent=2), flush=True)
    return parent, devices


def copy_preserved_outputs(parent, run):
    # Real copies: downstream writers can never mutate original checkpoints.
    source = parent / 'outputs'
    files = [p for p in source.rglob('*') if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    if shutil.disk_usage(run).free < total + 1024**3:
        raise RuntimeError('Insufficient disk for independent output copies')
    destination = run / 'outputs'
    destination.mkdir()
    for directory in source.rglob('*'):
        if directory.is_dir():
            (destination / directory.relative_to(source)).mkdir(parents=True, exist_ok=True)
    with tqdm(total=total, desc='Preserve checkpoints in new run', unit='B', unit_scale=True) as bar:
        for p in files:
            dest = destination / p.relative_to(source)
            shutil.copy2(p, dest)
            if dest.stat().st_size != p.stat().st_size:
                raise IOError(f'Copy size mismatch: {p}')
            bar.update(p.stat().st_size)
    # These stages have no resumable optimizer/iteration state. Only clear new copies.
    for rel in ['model257_full_image_stats_1x', 'model322_team_b4_group_oof/fold3',
                'model347_hrnet_inverse_ce_finetune/valid_eval',
                'model361_hrnet_scale3_finetune/valid_eval']:
        target = destination / rel
        if target.exists():
            shutil.rmtree(target)


def prepare(parent, recipe, workers):
    run = ROOT / 'outputs/runs' / datetime.now(timezone.utc).strftime('model364_resume_%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True, exist_ok=False)
    source = run / 'source_snapshot'
    shutil.copytree(ROOT / 'source', source)
    shutil.copytree(ROOT / 'adapters', run / 'adapters')
    launcher = run / 'launcher_snapshot'; launcher.mkdir()
    for name in ['run.py', 'run.sh', 'start_nohup.sh', 'scheduler.py', 'pipeline_tasks.json', 'README.md']:
        shutil.copy2(ROOT / name, launcher / name)
    for name in ['provenance.json', 'source_manifest.json']:
        shutil.copy2(ROOT / name, run / name)
    shutil.copy2(parent / 'dataset_manifest.json', run / 'dataset_manifest.json')
    shutil.copy2(parent / 'status.json', run / 'parent_status.json')
    (run / 'environment.freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    (run / 'local_patches.diff').write_text('Model361 evaluation only: --overlap 170 -> 114; input-scale=3, native_tile=171, stride=57.\n')
    # Dataset copies avoid old absolute CV symlinks; rebuild group links below.
    shutil.copytree(parent / 'data', run / 'data')
    copy_preserved_outputs(parent, run)
    env = os.environ.copy()
    env.update(PYTHONPATH=f"{run / 'adapters'}:{source / 'experiments'}:{source / 'experiments/historical_scripts'}",
               KOMAP_RESULTS_ROOT=str(REPO / 'results'), CUDA_VISIBLE_DEVICES='',
               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    subprocess.run([sys.executable, '-B', '-u'] + render(recipe['prepare_cv_argv'], source, run, workers),
                   env=env, cwd=run, check=True)
    write(run / 'resolved_commands.json', [{**t, 'argv': render(t['argv'], source, run, workers)}
                                         for t in recipe['tasks']])
    return run, source, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'train'], nargs='?', default='check')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or (args.execute and args.command != 'train'):
        parser.error('Positive workers and train --execute are required for execution')
    recipe = json.loads((ROOT / 'pipeline_tasks.json').read_text())
    parent, devices = inspect(recipe)
    if args.command != 'train' or not args.execute:
        print('CHECK OK. No training. Start with ./start_nohup.sh', flush=True)
        return
    # Lock both resume coordinator and parent coordinator; no overlap with original launcher.
    with (ROOT / 'outputs/pipeline.lock').open('a+') as lock, (parent.parents[1] / 'pipeline.lock').open('a+') as parent_lock:
        for handle in (lock, parent_lock):
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Another parent/resume launcher is active')
        run, source, env = prepare(parent, recipe, args.workers)
        print(f'RUN_DIRECTORY={run}\nREUSED_COMPLETED=26\nREMAINING=31', flush=True)
        run_tasks(recipe['tasks'], run, source, [4, 5, 6, 7], {g: devices[g] for g in [4, 5, 6, 7]},
                  sys.executable, env, workers=args.workers, lock_fd=lock.fileno())
        print('COMPLETE: ' + (run / 'result_path.txt').read_text().strip(), flush=True)


if __name__ == '__main__':
    main()
