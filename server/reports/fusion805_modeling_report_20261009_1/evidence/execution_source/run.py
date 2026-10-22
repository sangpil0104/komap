"""User-started single retraining of all components in the Model349 fusion805 bundle."""
import argparse
import fcntl
from pathlib import Path
import os
import shutil
import signal
import subprocess
import sys
import time

from experiment import ROOT, GPUS, environment, inventory, require_komap, read_json, write_json, stamp
from scheduler import run_tasks


def snapshot(run):
    destination = run / 'implementation'
    destination.mkdir()
    for name in [*read_json(ROOT / 'code_manifest.json'), 'code_manifest.json', 'input_manifest.json']:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    for folder in ['data', 'folds', 'initial_weights', 'expert_splits']:
        (destination / folder).symlink_to(ROOT / folder, target_is_directory=True)
    shutil.copy2(ROOT / 'plan.json', run / 'plan.json')
    (run / 'environment.freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    return destination


def call_managed(command, env, cwd, lock_fd):
    process = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    previous = {s: signal.signal(s, interrupted) for s in [signal.SIGTERM, signal.SIGINT]}
    try:
        process = subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True, pass_fds=(lock_fd,))
        code = process.wait()
        if code:
            raise RuntimeError(f'Post-training stage failed with exit {code}: {command[3]}')
    finally:
        for s in previous:
            signal.signal(s, signal.SIG_IGN)
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            except ProcessLookupError:
                pass
            process.wait()
        for s, handler in previous.items():
            signal.signal(s, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'check', 'train'], nargs='?', default='check')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute and args.command != 'train':
        parser.error('--execute is only valid with train')
    require_komap()
    plan = read_json(ROOT / 'plan.json')
    if args.command == 'plan':
        print('Fusion805: 27 training jobs + 16 preparation/inference stages; physical GPUs 2,3,4,5,6,7; conda komap.')
        for task in plan['tasks']:
            print(f"{task['id']}: {task['role']} seed={task['seed']} parents={task['deps']}")
        print('Fresh R: 14 models. Fresh E: B4 4 folds + HRNet 2 seeds x 4 folds. F: train70 OOF residual, 8 epochs.')
        print('Fixed stage1 R+E is the main result; stage2 R+E+F is also saved separately.')
        return
    from preflight import check
    report = check()
    (ROOT / 'checks').mkdir(exist_ok=True)
    write_json(ROOT / 'checks' / f'preflight_{stamp()}.json', report)
    if args.command != 'train' or not args.execute:
        print('PREFLIGHT PASSED. No training or inference started. User start: ./start_nohup.sh')
        return
    (ROOT / 'outputs').mkdir(exist_ok=True)
    with (ROOT / 'outputs/train.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('This experiment already has an active run.')
        run = ROOT / 'outputs' / f'fusion805_train_{stamp()}'
        run.mkdir(exist_ok=False)
        implementation = snapshot(run)
        write_json(run / 'preflight.json', report)
        print(f'RUN_DIRECTORY={run}', flush=True)
        env = environment(implementation)
        targets = {g: report['gpus'][g]['uuid'] for g in GPUS}
        started = time.monotonic()
        try:
            run_tasks(plan['tasks'], run, implementation / 'source', GPUS, targets, sys.executable, env,
                      lock_fd=lock.fileno(), idle_gpus=lambda: [g for g,d in inventory().items()
                                                            if d['used_mib'] <= 64 and d['utilization'] == 0])
            # Run finalization from the recorded implementation, with a managed process group.
            call_managed([sys.executable, '-B', '-u', str(implementation / 'finalize.py'),
                          '--run', str(run), '--gpus', ','.join(targets[g] for g in GPUS)],
                         env, run, lock.fileno())
            final = read_json(run / 'finished.json')
            print(f"COMPLETE: 27/27 trained | RESULT: {final['result']} | elapsed={(time.monotonic()-started)/3600:.2f}h", flush=True)
        except BaseException as error:
            write_json(run / 'failed.json', dict(error=repr(error), elapsed_seconds=time.monotonic()-started))
            raise


if __name__ == '__main__':
    main()
