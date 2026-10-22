"""User-started 100-training scheduler: 4 GPUs, deadline-aware launch, resumable tasks."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
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
from common import ROOT, read_json, write_json, environment, validate_plan, sha
from preflight import check, inventory

def estimate_seconds(task, completed):
    observed=[r['scheduler_seconds'] for r in completed.values()
              if r.get('kind')==task['kind'] and r.get('model')==task.get('model') and 'scheduler_seconds' in r]
    if observed:
        return max(task['expected_seconds'],sum(observed)/len(observed))*1.15
    return task['expected_seconds']*1.15

def can_launch(task, completed, remaining_seconds):
    return estimate_seconds(task,completed)<=remaining_seconds

def eligible(pending, completed):
    return sorted([t for t in pending.values() if set(t['deps'])<=completed.keys()],
                  key=lambda t:(t['priority'],t.get('seed',0),t['id']))

def tail(path):
    with path.open('rb') as f:
        f.seek(0,2); size=f.tell(); f.seek(max(0,size-1500))
        lines=f.read().decode(errors='replace').replace('\r','\n').splitlines()
    return next((s for s in reversed(lines) if s.strip()),'starting')

def stop_jobs(jobs):
    for j in jobs.values():
        try: os.killpg(j['process'].pid,signal.SIGTERM)
        except ProcessLookupError: pass
    deadline=time.monotonic()+8
    while any(j['process'].poll() is None for j in jobs.values()) and time.monotonic()<deadline:
        time.sleep(.1)
    for j in jobs.values():
        try: os.killpg(j['process'].pid,signal.SIGKILL)
        except ProcessLookupError: pass
        j['process'].wait(); j['handle'].close()

def snapshot(run):
    implementation=run/'implementation'; implementation.mkdir()
    for filename in read_json(ROOT/'code_manifest.json'):
        src=ROOT/filename; dst=implementation/filename
        dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
    for filename in ['code_manifest.json','input_manifest.json']:
        shutil.copy2(ROOT/filename,implementation/filename)
    (implementation/'cache').symlink_to(ROOT/'cache',target_is_directory=True)
    shutil.copy2(ROOT/'plan.json',run/'plan.json')
    (run/'completed').mkdir(); (run/'logs').mkdir()
    write_json(run/'implementation_manifest.json',
               {str(p.relative_to(implementation)):sha(p) for p in implementation.rglob('*')
                if p.is_file() and not p.is_relative_to(implementation/'cache')})
    (run/'environment.freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))

def run_tasks(run, gpus, devices, hours, reserve_hours, lock_fd, budget_mode):
    plan=validate_plan(read_json(run/'plan.json'))
    completed={p.stem:read_json(p) for p in (run/'completed').glob('*.json')}
    tasks={t['id']:t for t in plan['tasks']}
    pending={k:t for k,t in tasks.items() if k not in completed}
    jobs={}; started=time.monotonic(); last_report=-1e9
    horizon=(hours-reserve_hours)*3600 if budget_mode=='pause' else float('inf')
    events=run/'events.jsonl'
    session=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    progress=tqdm(total=100,initial=sum(r['kind']=='train' for r in completed.values()),
                  desc='Seed search training',unit='model',mininterval=5,dynamic_ncols=True)
    def event(kind,**data):
        with events.open('a') as f: f.write(json.dumps(dict(session=session,event=kind,**data))+'\n')
    def status(phase):
        elapsed=time.monotonic()-started
        eta=sum(estimate_seconds(t,completed) for t in pending.values() if t['kind']!='select')/4
        eta+=sum(max(0,estimate_seconds(j['task'],completed)-(time.monotonic()-j['start'])) for j in jobs.values())/4
        value=dict(phase=phase,session=session,elapsed_seconds=elapsed,estimated_remaining_seconds=eta,
                   budget_hours=hours,reserve_hours=reserve_hours,budget_mode=budget_mode,
                   training_completed=sum(r['kind']=='train' for r in completed.values()),
                   planned_training=100,completed=list(completed),pending=list(pending),
                   active=[dict(id=k,gpu=j['gpu'],pid=j['process'].pid,log=str(j['log'])) for k,j in jobs.items()])
        write_json(run/'status.json',value)
        return value
    def interrupted(signum,frame): raise KeyboardInterrupt(f'signal {signum}')
    handlers={s:signal.signal(s,interrupted) for s in [signal.SIGTERM,signal.SIGINT]}
    try:
        while pending or jobs:
            for identifier,job in list(jobs.items()):
                code=job['process'].poll()
                if code is None: continue
                job['handle'].close()
                if code: raise RuntimeError(f'{identifier} failed, exit={code}; log={job["log"]}')
                output=run/'completed'/f'{identifier}.json'
                record=read_json(output)
                record['scheduler_seconds']=time.monotonic()-job['start']
                write_json(output,record); completed[identifier]=record
                if record['kind']=='train': progress.update(1)
                event('finished',task=identifier,gpu=job['gpu'],seconds=record['scheduler_seconds'])
                del jobs[identifier]
            free=[g for g in gpus if g not in {j['gpu'] for j in jobs.values()}]
            cpu_busy=any(j['gpu'] is None for j in jobs.values())
            remaining=horizon-(time.monotonic()-started)
            for task in eligible(pending,completed):
                if not can_launch(task,completed,remaining): continue
                if task['kind']=='select':
                    if cpu_busy: continue
                    gpu=None; cpu_busy=True
                else:
                    if not free: continue
                    gpu=free.pop(0)
                env=environment(); env.update(MODEL349_USER_STARTED='1',
                    CUDA_VISIBLE_DEVICES='' if gpu is None else devices[gpu]['uuid'],
                    PYTHONHASHSEED=str(task.get('seed',0)))
                log=run/'logs'/f'{task["id"]}_{session}.log'; handle=log.open('x')
                command=[sys.executable,'-B','-u',str(run/'implementation/worker.py'),
                         '--run',str(run),'--task',task['id']]
                process=subprocess.Popen(command,env=env,cwd=run,stdout=handle,stderr=subprocess.STDOUT,
                                         start_new_session=True,pass_fds=(lock_fd,))
                jobs[task['id']]=dict(process=process,handle=handle,log=log,gpu=gpu,task=task,start=time.monotonic())
                del pending[task['id']]
                event('started',task=task['id'],gpu=gpu,cuda_visible_devices=env['CUDA_VISIBLE_DEVICES'],pid=process.pid)
                tqdm.write(f'START {task["id"]} GPU={gpu} | {log}')
            if time.monotonic()-last_report>=20:
                value=status('running'); elapsed=value['elapsed_seconds']/3600
                tqdm.write(f'Elapsed={elapsed:.2f}h | ETA≈{value["estimated_remaining_seconds"]/3600:.2f}h | '
                           f'Trained={value["training_completed"]}/100 | budget={hours:.1f}h')
                for name,j in jobs.items(): tqdm.write(f'[{name} GPU {j["gpu"]}] {tail(j["log"])}')
                last_report=time.monotonic()
            if not jobs:
                if pending:
                    status('budget_paused'); event('budget_paused',remaining=list(pending))
                    return False
                break
            time.sleep(.5)
        status('trained_all'); return True
    except BaseException as error:
        event('interrupted_or_failed',error=str(error)); stop_jobs(jobs); jobs.clear()
        status('interrupted_or_failed'); raise
    finally:
        progress.close()
        for s,h in handlers.items(): signal.signal(s,h)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['plan','check','train'],nargs='?',default='check')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--gpus',default='4,5,6,7')
    p.add_argument('--hours',type=float,default=10)
    p.add_argument('--reserve-hours',type=float,default=1.25)
    p.add_argument('--budget-mode',choices=['pause','finish-all'],default='pause',
                   help='pause stops launching work that cannot fit; finish-all may exceed --hours')
    p.add_argument('--resume',type=Path)
    args=p.parse_args()
    if args.execute and args.command!='train': p.error('--execute requires train')
    if args.resume and args.command!='train': p.error('--resume requires train')
    if not (0<args.reserve_hours<args.hours): p.error('Require 0 < reserve-hours < hours')
    gpus=[int(x.strip()) for x in args.gpus.split(',')]
    plan=validate_plan(read_json(ROOT/'plan.json'))
    print(f'100 training jobs; seeds 3407..3506; GPUs={gpus}; budget={args.hours}h ({args.budget_mode})',flush=True)
    if args.command=='plan':
        for m in ['269','347','336','329','326','123']:
            rows=[t for t in plan['tasks'] if t['kind']=='train' and t['model']==m]
            print(f'Model{m}: {len(rows)} models | seeds {[r["seed"] for r in rows]}')
        return
    info=check(gpus)
    if args.command=='check' or not args.execute:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        write_json(ROOT/'checks'/f'preflight_{stamp}.json',info)
        print('PREFLIGHT PASS: no training. User must invoke train --execute.',flush=True); return
    (ROOT/'outputs').mkdir(exist_ok=True)
    with (ROOT/'outputs/search.lock').open('a+') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('A seed100 search is already running')
        if args.resume:
            run=args.resume.resolve()
            if not run.is_relative_to(ROOT/'outputs/runs'): raise ValueError('Resume must point to this experiment run')
            for path,expected in read_json(run/'implementation_manifest.json').items():
                if sha(run/'implementation'/path)!=expected: raise ValueError(f'Snapshot changed: {path}')
        else:
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
            run=ROOT/'outputs/runs'/f'seed100_{stamp}'; run.mkdir(parents=True)
            snapshot(run); write_json(run/'preflight.json',info)
        print(f'RUN_DIRECTORY={run}',flush=True)
        done=run_tasks(run,gpus,inventory(),args.hours,args.reserve_hours,lock.fileno(),args.budget_mode)
        env=environment(); env['CUDA_VISIBLE_DEVICES']=''
        # Finalization is a process group too, so interrupting the coordinator
        # cannot leave an untracked finalizer modifying results in the background.
        process=subprocess.Popen([sys.executable,'-B','-u',str(run/'implementation/finalize.py'),
                                  '--run',str(run)],env=env,start_new_session=True,pass_fds=(lock.fileno(),))
        def final_interrupt(signum,frame): raise KeyboardInterrupt(f'signal {signum}')
        previous={s:signal.signal(s,final_interrupt) for s in [signal.SIGTERM,signal.SIGINT]}
        try:
            code=process.wait()
            if code: raise RuntimeError(f'Finalization failed: {code}; resume is safe')
        except BaseException:
            try: os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            process.wait(); raise
        finally:
            for s,h in previous.items(): signal.signal(s,h)
        value=read_json(run/'status.json'); value['phase']='complete' if done else 'budget_paused'
        value['result']=read_json(run/'latest_result.json'); write_json(run/'status.json',value)
        print('COMPLETE: 100/100' if done else f'BUDGET PAUSED. Resume: ./start_nohup.sh --resume {run}',flush=True)

if __name__=='__main__': main()
