"""Sequential inference-only job queue; never shadows the standard queue module."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import ROOT,DEADLINE,save_json


def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def main():
    p=argparse.ArgumentParser();p.add_argument('--jobs',type=Path,required=True);p.add_argument('--wait-pid',type=int);args=p.parse_args()
    start=time.time()
    while args.wait_pid and alive(args.wait_pid):
        if time.time()>=DEADLINE:return
        print(f'Waiting for inference PID {args.wait_pid}; elapsed {time.time()-start:.0f}s; deadline in {DEADLINE-time.time():.0f}s',flush=True)
        time.sleep(30)
    jobs=json.loads(args.jobs.read_text());records=[]
    for job in jobs:
        if time.time()>=DEADLINE-180:break
        log=ROOT/'checks'/job['log'];print('START',job['id'],log,flush=True)
        with log.open('x') as f:
            child=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/job['script']),*job['args']],stdout=f,stderr=subprocess.STDOUT)
            save_json(ROOT/'checks'/'queue_live.json',dict(queue_pid=os.getpid(),child_pid=child.pid,job=job['id'],log=str(log)))
            try:
                while child.poll() is None:
                    if time.time()>=DEADLINE:
                        child.terminate()
                        try:child.wait(timeout=15)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        break
                    time.sleep(2)
            except BaseException:
                child.terminate();child.wait();raise
        records.append(dict(id=job['id'],exit_code=child.returncode,log=str(log)))
        save_json(ROOT/'checks'/'queue_completed.json',records)
        print('FINISHED',job['id'],child.returncode,flush=True)
        if child.returncode!=0:print('Job failure recorded; next independent inference condition may proceed.',flush=True)


if __name__=='__main__':main()
