"""Sequential suite: one torchrun job using GPUs 0-7, then the next trial."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from tqdm import tqdm
import torch

from boundarylab.config import (BASELINE_SOURCE, CHECKPOINT, ENCODER_WEIGHTS, PROJECT,
                                selected_arms, validate_plan)
from boundarylab.data import audit_data
from boundarylab.io import environment, fingerprint, read_json, require_komap, sha256, utc_stamp, write_json
from boundarylab.report import suite_report, verify_result

ROOT = Path(__file__).resolve().parent


def request_stop(signum, frame):
    # Route TERM/HUP through launch's child-process cleanup, just like Ctrl+C.
    raise KeyboardInterrupt(f'Received signal {signum}')


def source_files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.md','.json','.txt','.ini')
                  and not any(part in ('outputs','checks','__pycache__','.pytest_cache') for part in p.relative_to(ROOT).parts))


def preservation_check():
    saved = read_json(ROOT/'preservation.json')
    for name, digest in saved['files'].items():
        if sha256(name) != digest:
            raise ValueError(f'Preserved baseline artifact changed: {name}')
    return saved


def preflight(gpus):
    require_komap()
    if len(gpus) != 8 or set(gpus) != set(range(8)):
        raise ValueError('Use GPUs 0,1,2,3,4,5,6,7 exactly once')
    env = environment()
    expected = {'torch':'2.7.1+cu128', 'torchvision':'0.22.1+cu128',
                'segmentation-models-pytorch':'0.5.0', 'timm':'1.0.29',
                'opencv-python-headless':'4.13.0.92', 'albumentations':'2.0.8'}
    for name, version in expected.items():
        if env['packages'][name] != version:
            raise ValueError(f'{name}: expected {version}, got {env["packages"][name]}')
    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        raise RuntimeError('Eight CUDA GPUs must be visible for this protocol')
    preservation_check()
    rows = audit_data(PROJECT/'dataset')
    return dict(environment=env, data=rows, data_hash=fingerprint(rows),
                encoder_weights=str(ENCODER_WEIGHTS), encoder_sha256=sha256(ENCODER_WEIGHTS),
                reference_checkpoint=str(CHECKPOINT), reference_sha256=sha256(CHECKPOINT),
                gpus=gpus, gpu_names=[torch.cuda.get_device_name(i) for i in gpus])


def create_suite(args, inspected):
    outputs = ROOT/'outputs'
    outputs.mkdir(exist_ok=True)
    suite = outputs/utc_stamp()
    suite.mkdir()
    source = suite/'source'
    source.mkdir()
    files = source_files()
    hashes = {}
    for p in files:
        relative = p.relative_to(ROOT)
        dest = source/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        hashes[str(relative)] = sha256(dest)
    arms = selected_arms(args.experiments, steps=args.steps, crop=args.crop)
    plan = dict(schema_version=1, **inspected, arms=arms, results_root=str(PROJECT/'results'),
                source_files=hashes, source_hash=fingerprint(hashes), created_at=utc_stamp())
    write_json(suite/'plan.json', plan, exclusive=True)
    (suite/'trials').mkdir()
    (suite/'logs').mkdir()
    suite_report(suite)
    return suite, plan


def validate_resume(suite, inspected):
    if not suite.is_relative_to((ROOT/'outputs').resolve()):
        raise ValueError('Resume path must belong to this experiment outputs directory')
    plan = read_json(suite/'plan.json')
    validate_plan(plan['arms'])
    for key in ('data_hash','encoder_sha256','reference_sha256','environment','gpus'):
        if plan[key] != inspected[key]:
            raise ValueError(f'Resume mismatch: {key}. The old suite is preserved.')
    for name, digest in plan['source_files'].items():
        if sha256(suite/'source'/name) != digest:
            raise ValueError(f'Frozen source changed: {name}')
    return plan


def completed_trial(suite, arm, ids):
    marker_path = suite/'trials'/arm['id']/'complete.json'
    if not marker_path.is_file():
        return False
    marker = read_json(marker_path)
    if marker['arm_hash'] != fingerprint(arm):
        raise ValueError('Completed experiment configuration changed')
    summary = verify_result(marker['result'], ids)
    weights = summary['provenance']['weights']
    if sha256(weights) != marker['weights_sha256']:
        raise ValueError(f'Completed weights changed: {weights}')
    return True


def read_progress(run):
    path = run/'progress.json'
    if not path.is_file():
        return dict(phase='initializing', step=0, total=1)
    p = read_json(path)
    if p['phase'] == 'evaluate':
        p['step'] = sum(r['done'] for r in (read_json(f) for f in run.glob('eval_rank_*.json'))
                        if r['result'] == p['result'])
    return p


def launch(suite, plan, arm, remaining_train):
    run = suite/'trials'/arm['id']
    run.mkdir(exist_ok=True)
    attempt = utc_stamp()
    log_path = suite/'logs'/f'{arm["id"]}_{attempt}.log'
    cmd = [sys.executable, '-B', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
           str(suite/'source'/'worker.py'), '--suite', str(suite), '--experiment', arm['id']]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str,plan['gpus'])),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', NO_ALBUMENTATIONS_UPDATE='1',
               TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    tqdm.write(f'[{arm["id"]}] {arm["description"]}\n  log: {log_path}')
    with log_path.open('x') as log:
        process = subprocess.Popen(cmd, cwd=suite/'source', env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(suite/'status.json', dict(phase='running', experiment=arm['id'], pid=process.pid, log=str(log_path)))
        bar = tqdm(total=arm['config']['steps'] if arm['kind']=='train' else 20,
                   desc=arm['id'], position=1, unit='step', dynamic_ncols=True)
        phase = None
        started = time.monotonic()
        try:
            while process.poll() is None:
                p = read_progress(run)
                if p['phase'] != phase:
                    bar.reset(total=p['total'])
                    bar.set_description(arm['id']+' '+p['phase'])
                    phase = p['phase']
                bar.n = p['step']
                postfix = {}
                if 'loss' in p:
                    postfix['loss'] = f'{p["loss"]:.4f}'
                if 'eta_seconds' in p:
                    postfix['train_ETA_min'] = f'{p["eta_seconds"]/60:.1f}'
                    rate = p['elapsed_seconds']/max(p['step'], 1)
                    postfix['suite_train_ETA_h'] = f'{(p["eta_seconds"]+remaining_train*arm["config"]["steps"]*rate)/3600:.2f}'
                bar.set_postfix(postfix)
                bar.refresh()
                time.sleep(2)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            write_json(suite/'status.json', dict(phase='interrupted', experiment=arm['id'], log=str(log_path)))
            raise
        finally:
            bar.close()
        if process.returncode:
            write_json(suite/'status.json', dict(phase='failed', experiment=arm['id'], exit_code=process.returncode, log=str(log_path)))
            tail = '\n'.join(log_path.read_text(errors='replace').splitlines()[-40:])
            raise RuntimeError(f'{arm["id"]} failed ({process.returncode}).\n{tail}\nResume: bash {ROOT}/run.sh --resume {suite}')
        return time.monotonic()-started


def main():
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGHUP, request_stop)
    parser = argparse.ArgumentParser(description=__doc__)
    safe = parser.add_mutually_exclusive_group()
    safe.add_argument('--dry-run', action='store_true', help='Print protocol only; never train')
    safe.add_argument('--preflight', action='store_true', help='Read-only environment/data/preservation checks')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--steps', type=int, default=16000)
    parser.add_argument('--crop', type=int, default=1536)
    parser.add_argument('--experiments', nargs='+')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    require_komap()
    if args.dry_run:
        arms = selected_arms(args.experiments, steps=args.steps, crop=args.crop)
        print('Sequential: ONE experiment across GPUs 0-7; ImageNet encoder + fresh seed-32 decoder.')
        print(f'Train protocol: {args.steps} updates x global batch 8 = {args.steps*8} crops per arm; SyncBatchNorm; fixed final checkpoint.')
        for a in arms:
            print(f'{a["id"]:24s} {a["kind"]:5s} vs {a["contrast"] or "reference/control":24s} {a["change"]} {a["description"]}')
        return
    gpus = [int(g) for g in args.gpus.split(',')]
    inspected = preflight(gpus)
    if args.preflight:
        print(json.dumps(dict(environment=inspected['environment'], gpus=inspected['gpu_names'],
                              train_images=70, valid_images=20, data_hash=inspected['data_hash'],
                              encoder_sha256=inspected['encoder_sha256'], preservation='verified', training_started=False), indent=2))
        return
    suite, plan = (args.resume.resolve(), None) if args.resume else create_suite(args, inspected)
    # An OS-held lock, not a stale PID file, is authoritative for concurrent execution.
    with (suite/'suite.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This suite already has a running coordinator')
        if plan is None:
            plan = validate_resume(suite, inspected)
        print(f'SUITE: {suite}\nResume: bash {ROOT}/run.sh --resume {suite}', flush=True)
        ids = [r['image_id'] for r in plan['data'] if r['split']=='valid']
        progress = tqdm(total=len(plan['arms']), desc='suite', unit='trial', position=0, dynamic_ncols=True)
        for i, arm in enumerate(plan['arms']):
            if completed_trial(suite, arm, ids):
                tqdm.write(f'[{arm["id"]}] complete artifacts verified; reuse')
            else:
                later = sum(a['kind']=='train' for a in plan['arms'][i+1:])
                launch(suite, plan, arm, later)
                if not completed_trial(suite, arm, ids):
                    raise RuntimeError('Worker exited without a verified complete result')
            suite_report(suite)
            progress.update(1)
        progress.close()
        preservation_check()
        write_json(suite/'status.json', dict(phase='complete', completed=len(plan['arms'])))
        print(f'Completed. Comparison: {suite/"comparison.md"}', flush=True)


if __name__ == '__main__':
    main()
