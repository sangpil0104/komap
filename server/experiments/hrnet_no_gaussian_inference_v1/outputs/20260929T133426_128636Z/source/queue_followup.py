"""Named inference queue with deadline cleanup limited to its own process group."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from common import ROOT, DEADLINE, save_json


def alive(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().split()
        return stat[2] != 'Z'
    except FileNotFoundError:
        return False


def stop(child):
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    assert args.name.replace('_', '').isalnum()
    prefix = ROOT / 'checks' / args.name
    while args.wait_pid and alive(args.wait_pid):
        if time.time() >= DEADLINE:
            return
        print(f'Waiting for inference PID {args.wait_pid}; deadline in {DEADLINE-time.time():.0f}s', flush=True)
        time.sleep(30)
    records = []
    for job in json.loads(args.jobs.read_text()):
        if time.time() >= DEADLINE - 180:
            break
        log = ROOT / 'checks' / job['log']
        print('START', job['id'], log, flush=True)
        with log.open('x') as stream:
            child = subprocess.Popen([sys.executable, '-B', '-u', str(ROOT / job['script']), *job['args']],
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            save_json(prefix.with_suffix('.live.json'), dict(queue_pid=os.getpid(), child_pid=child.pid,
                      process_group=child.pid, job=job['id'], log=str(log)))
            try:
                while child.poll() is None:
                    if time.time() >= DEADLINE:
                        stop(child)
                        break
                    time.sleep(2)
            except BaseException:
                stop(child)
                raise
        records.append(dict(id=job['id'], exit_code=child.returncode, log=str(log)))
        save_json(prefix.with_suffix('.completed.json'), records)
        print('FINISHED', job['id'], child.returncode, flush=True)
    save_json(prefix.with_suffix('.done.json'), dict(pid=os.getpid(), completed=records, ended=time.time()))


if __name__ == '__main__':
    main()
