"""Isolated nohup run, progress, immutable source snapshot and automatic result export."""
from pathlib import Path
import argparse, fcntl, os, shutil, signal, subprocess, sys, time
from tqdm import tqdm
from common import ROOT, GPUS, now, read, write, devices

def heartbeat(run,bar):
    rows=[read(p) for p in (run/'stages').glob('*.json')]
    done=sum(r['status']=='done' for r in rows)
    # Two measured final inference stages are not training stages.
    done+=sum((run/f'pred_{s}/inference_timing.json').is_file() for s in ['valid','test'])
    bar.update(done-bar.n)
    active=[r for r in rows if r['status']=='running']
    bar.set_postfix_str(','.join(r['name'] for r in active) or 'preparing / final inference')
    for r in active:
        p=Path(r['log']);tail=''
        if p.exists():
            with p.open('rb') as stream:
                stream.seek(max(0,p.stat().st_size-2048));lines=stream.read().decode(errors='replace').replace('\r','\n').splitlines()
            if lines:tail=lines[-1][-240:]
        print(f"[active {r['name']}] {tail}",flush=True)
    return [r for r in rows if r['status']=='failed']

def train(lock_held=False):
    lock=None
    if not lock_held:
        lock=(ROOT/'.run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from preflight import check
    checked=check();hardware=devices()
    busy=[i for i,r in hardware.items() if r['memory_used_mib']>800 or r['utilization_percent']>10]
    if busy:raise RuntimeError(f'GPUs currently busy: {busy}; no training started. Retry when available.')
    if os.environ.get('SMOKE')=='1':raise ValueError('This launcher is full training only; SMOKE is not supported.')
    run=ROOT/'outputs'/('retrain_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()))
    run.mkdir(parents=True);(run/'logs').mkdir();snapshot=run/'implementation';snapshot.mkdir()
    for p in ROOT.iterdir():
        if p.is_file() and p.suffix in ['.py','.sh','.json','.md','.diff']:shutil.copy2(p,snapshot/p.name)
    shutil.copytree(ROOT/'source',snapshot/'source')
    (snapshot/'data').symlink_to(ROOT/'data',target_is_directory=True)
    write(run/'preflight.json',checked)
    with (run/'environment.freeze.txt').open('w') as f:subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
    write(ROOT/'latest_run.json',{'run':str(run),'started_at':now()})
    env=dict(os.environ,DATA=str(ROOT/'data'),OUT=str(run),PYTHON=sys.executable,SMOKE='0',PLAN_ONLY='0')
    gpu_list=','.join(hardware[i]['uuid'] for i in GPUS)
    env['CUDA_VISIBLE_DEVICES']=gpu_list
    start=time.perf_counter();child=None
    def stopped(sig,frame):raise KeyboardInterrupt(f'Signal {sig}')
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stopped)
    write(run/'status.json',dict(status='running',started_at=now(),physical_gpus=GPUS,hardware=hardware))
    print(f'RUN={run}\n33 training jobs; 51 total stages; physical GPUs={GPUS}',flush=True)
    try:
        child=subprocess.Popen(['bash',str(snapshot/'source/run_retrain.sh'),gpu_list],env=env,cwd=snapshot/'source',start_new_session=True)
        with tqdm(total=51,desc='lite13 stages (ETA estimate)',unit='stage',mininterval=10,dynamic_ncols=True) as bar:
            while child.poll() is None:
                failed=heartbeat(run,bar)
                if failed:raise RuntimeError('Stage failed: '+','.join(r['name'] for r in failed))
                time.sleep(10)
            heartbeat(run,bar)
        if child.returncode:raise subprocess.CalledProcessError(child.returncode,child.args)
        if any(read(p)['status']!='done' for p in (run/'stages').glob('*.json')):raise RuntimeError('Incomplete stage receipts')
        write(run/'pipeline_completion.json',dict(ok=True,wall_seconds=time.perf_counter()-start,finished_at=now()))
        # Execute the exact exporter captured at launch.
        subprocess.run([sys.executable,'-B',str(snapshot/'export_results.py'),'--run',str(run)],check=True)
        write(run/'status.json',dict(status='complete',finished_at=now(),wall_seconds=time.perf_counter()-start,result=read(run/'result.json')['result']))
        print('COMPLETE '+read(run/'result.json')['result'],flush=True)
    except BaseException as e:
        if child is not None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
        write(run/'status.json',dict(status='stopped_or_failed',error=str(e),finished_at=now(),wall_seconds=time.perf_counter()-start))
        raise

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['check','plan','train']);p.add_argument('--execute',action='store_true');p.add_argument('--lock-held',action='store_true');a=p.parse_args()
    if a.action=='plan':print((ROOT/'plan.json').read_text());return
    if a.action=='check' or not a.execute:
        from preflight import check
        report=check();write(ROOT/'checks/preflight.json',report);print(report)
        if a.action=='train':print('Checks only. To start training yourself: ./start_nohup.sh')
        return
    train(a.lock_held)
if __name__=='__main__':main()
