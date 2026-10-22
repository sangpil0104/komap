"""GPU 4..7 r70 paired experiment; only the user starts training."""
import argparse, fcntl, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path
from tqdm import tqdm
from common import (ROOT,GPUS,GROUP_COUNTS,read_json,write_json,sha,environment,inventory,
                    require_komap,stamp,validate_plan,records,blocking_experiments)

def verify_hashes(values,root=None,desc='Verify artifacts'):
    for name,expected in tqdm(values.items(),desc=desc,unit='file',mininterval=5):
        path=Path(name) if root is None else root/name
        if not path.is_file() or sha(path)!=expected:raise ValueError(f'Missing/changed input: {path}')

def check(root=ROOT):
    require_komap();plan=validate_plan(read_json(root/'plan.json'))
    verify_hashes(read_json(root/'code_manifest.json'),root,'Verify frozen source')
    verify_hashes(read_json(root/'input_manifest.json'),None,'Verify baseline/data')
    for d in ['train_data']:
        for p in (root/d).rglob('*'):
            if p.is_symlink() and not p.exists():raise ValueError(f'Broken fold link: {p}')
    # Inventory reads physical IDs; it does not initialize CUDA or allocate a model.
    devices=inventory()
    assert set(devices)==set(GPUS)
    from preflight import dataset_check,recipe_check
    data=dataset_check(root);recipe=recipe_check(root)
    info=dict(ok=True,training_started=False,gpus=devices,training_count=plan['training_count'],
              data=data,recipe=recipe,blocking_experiments=blocking_experiments(),free_gib=shutil.disk_usage(root).free/1024**3)
    if info['free_gib']<30:raise RuntimeError('Need at least 30 GiB free for weights and paired inference.')
    return info

def snapshot(run):
    impl=run/'implementation';impl.mkdir()
    for name in read_json(ROOT/'code_manifest.json'):
        p=impl/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,p)
    for name in ['code_manifest.json','input_manifest.json']:shutil.copy2(ROOT/name,impl/name)
    for name in ['data','train_data']:(impl/name).symlink_to(ROOT/name,target_is_directory=True)
    shutil.copy2(ROOT/'plan.json',run/'plan.json')
    for name in ['completed','logs']:(run/name).mkdir()
    write_json(run/'implementation_manifest.json',{str(p.relative_to(impl)):sha(p) for p in impl.rglob('*') if p.is_file()})
    (run/'environment.freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))

def verify_completed(run):
    completed=records(run);tasks={t['id']:t for t in read_json(run/'plan.json')['tasks']}
    if set(completed)-set(tasks):raise ValueError('Unknown completion record')
    for identifier,row in completed.items():
        assert set(tasks[identifier]['deps'])<=set(completed),identifier
        assert all(row[k]==v for k,v in tasks[identifier].items()),identifier
    hashes={k:v for row in completed.values() for k,v in row['artifacts'].items()}
    verify_hashes(hashes,desc='Verify completed tasks')

def stop_jobs(jobs):
    for j in jobs.values():
        try:os.killpg(j['process'].pid,signal.SIGTERM)
        except ProcessLookupError:pass
    deadline=time.monotonic()+8
    while any(j['process'].poll() is None for j in jobs.values()) and time.monotonic()<deadline:time.sleep(.1)
    for j in jobs.values():
        try:os.killpg(j['process'].pid,signal.SIGKILL)
        except ProcessLookupError:pass
        j['process'].wait();j['handle'].close()

def tail(path):
    with path.open('rb') as f:
        f.seek(0,2);f.seek(max(0,f.tell()-1600));lines=f.read().decode(errors='replace').replace('\r','\n').splitlines()
    return next((s for s in reversed(lines) if s.strip()),'starting')

def estimate_work_eta(plan,completed,jobs,estimate,now):
    """Simulate the remaining DAG on four GPU slots and one CPU slot."""
    finish={k:0.0 for k in completed};slots={g:0.0 for g in GPUS};cpu=0.0
    for identifier,job in jobs.items():
        remaining=max(0.0,estimate(job['task'])-(now-job['start']));finish[identifier]=remaining
        if job['gpu'] is None:cpu=max(cpu,remaining)
        else:slots[job['gpu']]=remaining
    for task in plan['tasks']:
        if task['id'] in finish:continue
        ready=max((finish[d] for d in task['deps']),default=0.0)
        if task['kind']=='cpu':
            cpu=max(cpu,ready)+estimate(task);finish[task['id']]=cpu
        else:
            gpu=min(slots,key=slots.get);slots[gpu]=max(slots[gpu],ready)+estimate(task)
            finish[task['id']]=slots[gpu]
    return max(finish.values(),default=0.0)

def run_tasks(run,lock_fd):
    plan=validate_plan(read_json(run/'plan.json'));completed=records(run)
    pending={t['id']:t for t in plan['tasks'] if t['id'] not in completed};jobs={};session=stamp()
    started=time.monotonic();last_report=-1e9;last_inventory=-1e9;devices=inventory()
    previous_elapsed=read_json(run/'status.json').get('total_elapsed_seconds',0) if (run/'status.json').exists() else 0
    progress=tqdm(total=3,initial=sum(r['kind']=='train' for r in completed.values()),desc='R70 B4 training',unit='model',mininterval=5)
    blocked=[]
    def estimate(task):
        seen=[r['scheduler_seconds'] for r in completed.values() if r['kind']==task['kind'] and r['group']==task['group'] and 'scheduler_seconds' in r]
        return max(task['expected_seconds'],sum(seen)/len(seen) if seen else 0)
    def event(kind,**data):
        with (run/'events.jsonl').open('a') as f:f.write(json.dumps(dict(session=session,event=kind,**data))+'\n')
    def status(phase):
        elapsed=time.monotonic()-started
        eta=estimate_work_eta(plan,completed,jobs,estimate,time.monotonic())
        value=dict(phase=phase,session=session,total_elapsed_seconds=previous_elapsed+elapsed,
                   session_elapsed_seconds=elapsed,estimated_remaining_seconds=eta,
                   training_completed=sum(r['kind']=='train' for r in completed.values()),planned_training=3,blocking_experiments=blocked,
                   completed=list(completed),pending=list(pending),gpus=GPUS,
                   active=[dict(task=k,gpu=j['gpu'],pid=j['process'].pid,log=str(j['log'])) for k,j in jobs.items()])
        write_json(run/'status.json',value);return value
    def interrupted(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    handlers={s:signal.signal(s,interrupted) for s in [signal.SIGTERM,signal.SIGINT]}
    try:
        while pending or jobs:
            for identifier,job in list(jobs.items()):
                code=job['process'].poll()
                if code is None:continue
                job['handle'].close()
                if code:raise RuntimeError(f'{identifier} exited {code}; {job["log"]}')
                path=run/'completed'/f'{identifier}.json';row=read_json(path)
                row['scheduler_seconds']=time.monotonic()-job['start'];write_json(path,row);completed[identifier]=row
                if row['kind']=='train':progress.update(1)
                event('finished',task=identifier,gpu=job['gpu'],seconds=row['scheduler_seconds']);del jobs[identifier]
            if time.monotonic()-last_inventory>=10:
                devices=inventory();blocked=blocking_experiments();last_inventory=time.monotonic()
            busy={j['gpu'] for j in jobs.values()}
            free=[] if blocked else [g for g in GPUS if g not in busy and devices[g]['used_mib']<=64 and devices[g]['utilization']==0]
            cpu_busy=None in busy
            ready=sorted([t for t in pending.values() if set(t['deps'])<=completed.keys()],key=lambda t:(t['priority'],t.get('seed',0),t['id']))
            for task in ready:
                if task['kind']=='cpu':
                    if cpu_busy:continue
                    gpu=None;cpu_busy=True
                else:
                    if not free:continue
                    gpu=free.pop(0)
                env=environment(run/'implementation')
                env.update(R70_USER_STARTED='1',CUDA_VISIBLE_DEVICES='' if gpu is None else devices[gpu]['uuid'],
                           KOMAP_PHYSICAL_GPU='' if gpu is None else str(gpu),PYTHONHASHSEED=str(task.get('seed',0)))
                log=run/'logs'/f'{task["id"]}_{session}.log';handle=log.open('x')
                command=[sys.executable,'-B','-u',str(run/'implementation/worker.py'),'--run',str(run),'--task',task['id']]
                process=subprocess.Popen(command,env=env,cwd=run,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock_fd,))
                jobs[task['id']]=dict(process=process,handle=handle,log=log,gpu=gpu,task=task,start=time.monotonic())
                del pending[task['id']];event('started',task=task['id'],gpu=gpu,pid=process.pid,cuda_visible_devices=env['CUDA_VISIBLE_DEVICES'])
                tqdm.write(f'START {task["id"]} GPU={gpu} | {log}')
            if time.monotonic()-last_report>=20:
                value=status('running' if jobs else 'waiting_for_available_gpu')
                tqdm.write(f'Trained {value["training_completed"]}/3 | stages {len(completed)}/{len(plan["tasks"])} | elapsed={value["total_elapsed_seconds"]/3600:.2f}h | work ETA≈{value["estimated_remaining_seconds"]/3600:.2f}h (GPU queue time excluded)')
                for name,j in jobs.items():tqdm.write(f'[{name} GPU {j["gpu"]}] {tail(j["log"])}')
                if pending and not jobs:tqdm.write('Waiting: '+(', '.join(blocked) if blocked else 'physical GPU 4,5,6,7 are busy'))
                last_report=time.monotonic()
            if pending and not jobs and not ready:raise RuntimeError('Unresolved task dependencies')
            time.sleep(.5)
        status('complete');event('complete');return completed
    except BaseException as error:
        event('failed_or_interrupted',error=str(error));stop_jobs(jobs);jobs.clear();status('failed_or_interrupted');raise
    finally:
        progress.close()
        for s,h in handlers.items():signal.signal(s,h)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['plan','check','train'],nargs='?',default='check')
    p.add_argument('--execute',action='store_true');p.add_argument('--resume',type=Path);a=p.parse_args()
    if (a.execute or a.resume) and a.command!='train':p.error('--execute / --resume requires train')
    plan=validate_plan(read_json(ROOT/'plan.json'))
    print(f'Lite20 r70: 3 full-train70 B4 models, 21,000 iterations each, GPU 4,5,6,7; paired comparison to local lite20 S=0.',flush=True)
    if a.command=='plan':return
    info=check()
    if a.command!='train' or not a.execute:
        write_json(ROOT/'checks'/f'preflight_{stamp()}.json',info)
        print('PREFLIGHT PASS. No training. Start manually using ./start_nohup.sh',flush=True);return
    (ROOT/'outputs').mkdir(exist_ok=True)
    with (ROOT/'outputs/train.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('An r70 experiment is already running')
        if a.resume:
            run=a.resume.resolve()
            if not run.is_relative_to(ROOT/'outputs/runs'):raise ValueError('Resume must belong to this experiment')
            verify_hashes(read_json(run/'implementation_manifest.json'),run/'implementation','Verify run snapshot')
            assert read_json(run/'plan.json')==read_json(run/'implementation/plan.json')
            verify_completed(run)
        else:
            run=ROOT/'outputs/runs'/f'r70_{stamp()}';run.mkdir(parents=True);snapshot(run);write_json(run/'preflight.json',info)
        write_json(ROOT/'outputs/latest_run.json',dict(run=str(run)))
        print(f'RUN_DIRECTORY={run}',flush=True)
        completed=run_tasks(run,lock.fileno());print(f'COMPLETE 3/3. RESULT={completed["export"]["result"]}',flush=True)

if __name__=='__main__':main()
