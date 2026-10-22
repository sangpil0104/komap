"""Prepare/check by default; evaluate only when the user supplies --execute."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from common import ROOT, SOURCE, VALID, GPUS, stamp, read_json, write_json, require_komap


def snapshot(destination):
    destination.mkdir()
    shutil.copytree(SOURCE, destination / 'source', ignore=shutil.ignore_patterns('weights', '__pycache__'))
    for name in ['run.py', 'common.py', 'preflight.py', 'export_results.py', 'run.sh', 'start_nohup.sh',
                 'README.md', 'provenance.json', 'local_weights.json', 'data_manifest.json']:
        shutil.copy2(ROOT / name, destination / name)
    (destination / 'environment.freeze.txt').write_text(subprocess.check_output([
        sys.executable, '-m', 'pip', 'freeze'], text=True))


def stop_child(process):
    if process is None:
        return
    # The inference supervisor and every worker share this private process group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action')
    sub.add_parser('plan')
    checks = sub.add_parser('check')
    checks.add_argument('--load-models', action='store_true', help='Strictly load all 14 models on CPU; no forward.')
    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('--execute', action='store_true', help='Start the original exh14 valid20 evaluation.')
    finalize = sub.add_parser('export')
    finalize.add_argument('--run', required=True, type=Path, help='Completed run directory; creates a NEW report.')
    args = parser.parse_args()
    require_komap()
    if args.action == 'plan':
        print('Model349 exh14: fixed 14-checkpoint FP32 inference; valid20.')
        print('Physical GPUs: 2,3,4,5,6,7, addressed by UUID; up to six independent component workers.')
        print('Components: B4 322 folds 3,4; B4 326 folds 1,2; B4 329 folds 1,3,4; base; HRNet336; HRNet347; Native269.')
        print('Image mIoU: per-image four-class IoU (empty=0), then mean over 20 images.')
        print('Original TENT: five steps per image; checkpoint files stay unchanged.')
        print('User start: ./start_nohup.sh or ./run.sh evaluate --execute')
        return
    if args.action == 'export':
        from export_results import export
        print(f'RESULT: {export(args.run)}')
        return
    from preflight import check
    execute = args.action == 'evaluate' and args.execute
    lock = None
    try:
        if execute:
            lock = (ROOT / '.evaluate.lock').open('a+')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit('An evaluation from this experiment is already running.')
        report = check(load_models=getattr(args, 'load_models', False), require_idle=execute)
        (ROOT / 'checks').mkdir(exist_ok=True)
        check_path = ROOT / 'checks' / f'preflight_{stamp()}.json'
        write_json(check_path, report)
        print(f'CHECK PASSED: {check_path}', flush=True)
        if not execute:
            print('No inference, TENT, or training started. Use evaluate --execute to begin.')
            return
        run = ROOT / 'outputs' / f'exh14_{stamp()}'
        run.mkdir(parents=True, exist_ok=False)
        snapshot(run / 'source_snapshot')
        write_json(run / 'preflight.json', report)
        # Run a snapshot of the original code. Weights are private exact copies made during setup.
        frozen_source = run / 'source_snapshot/source'
        (frozen_source / 'weights').symlink_to(SOURCE / 'weights', target_is_directory=True)
        command = [sys.executable, '-B', '-u', str(frozen_source / 'infer_lite.py'),
                   '--input', str(VALID / 'images'), '--gt', str(VALID / 'masks'),
                   '--output', str(run / 'inference'),
                   '--gpus', ','.join(g['uuid'] for g in report['gpus']), '--keep-probabilities']
        write_json(run / 'command.json', command)
        print(f'RUN: {run}\nGPUs: {GPUS}', flush=True)
        child = None
        started = time.monotonic()
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Signal {signum}')
        old_handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            child = subprocess.Popen(command, cwd=frozen_source, start_new_session=True)
            code = child.wait()
            if code:
                raise RuntimeError(f'Original inference failed (exit {code}). Logs: {run}/inference/logs')
            from export_results import export
            destination = export(run)
            write_json(run / 'finished.json', dict(result=str(destination), elapsed_seconds=time.monotonic() - started))
            print(f'RESULT: {destination}', flush=True)
        except BaseException as error:
            write_json(run / 'failed.json', dict(error=repr(error), elapsed_seconds=time.monotonic() - started))
            raise
        finally:
            for s in old_handlers:
                signal.signal(s, signal.SIG_IGN)
            stop_child(child)
            for s, handler in old_handlers.items():
                signal.signal(s, handler)
    finally:
        if lock is not None:
            lock.close()


if __name__ == '__main__':
    main()
