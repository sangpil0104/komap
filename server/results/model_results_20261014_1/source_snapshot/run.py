"""Resume only inference/scoring/export; this executable has no training entrypoint."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,fcntl,os,shutil,signal,subprocess,sys,time
from tqdm import tqdm
from common import ROOT,GPUS,read,write,sha,now,devices

def execute(lock_held=False):
    if not lock_held:
        lock=(ROOT/'.run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from preflight import check
    report=check();hardware=devices()
    busy=[i for i,r in hardware.items() if r['memory_used_mib']>800 or r['utilization_percent']>10]
    if busy:raise RuntimeError(f'Requested GPUs are busy: {busy}. No inference started.')
    parent=Path(read(ROOT/'parent.json')['parent_run'])
    name=datetime.now(ZoneInfo('Asia/Seoul')).strftime('resume_%Y%m%d_%H%M%S_%f')
    run=ROOT/'outputs'/name;run.mkdir(parents=True);(run/'logs').mkdir();impl=run/'implementation';impl.mkdir()
    for p in ROOT.iterdir():
        if p.is_file() and p.suffix in ['.py','.sh','.json','.md','.diff']:shutil.copy2(p,impl/p.name)
    shutil.copytree(ROOT/'source',impl/'source');(impl/'data').symlink_to((ROOT/'data').resolve(),target_is_directory=True)
    (run/'model').mkdir();(run/'model/weights').symlink_to(parent/'model/weights',target_is_directory=True)
    shutil.copytree(parent/'stages',run/'stages')
    write(run/'preflight.json',report);write(run/'resume_provenance.json',read(ROOT/'parent.json'))
    shutil.copy2(ROOT/'checks/parent_audit.json',run/'parent_audit.json')
    with (run/'environment.freeze.txt').open('w') as f:subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
    write(ROOT/'latest_run.json',dict(run=str(run),started_at=now()));write(run/'status.json',dict(status='running',reused_training=33,new_training=0,started_at=now()))
    env=dict(os.environ,PYTHON=sys.executable);env['CUDA_VISIBLE_DEVICES']=','.join(hardware[i]['uuid'] for i in GPUS)
    child=None;t0=time.perf_counter()
    def stop(sig,frame):raise KeyboardInterrupt(f'Signal {sig}')
    for s in [signal.SIGTERM,signal.SIGINT]:signal.signal(s,stop)
    def call(command,name,estimate):
        nonlocal child
        log=run/'logs'/f'{name}.log';started=time.monotonic();last_line=None
        print(f'START {name}; log={log}',flush=True)
        with log.open('w') as stream:
            child=subprocess.Popen(list(map(str,command)),cwd=impl,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                elapsed=time.monotonic()-started
                print(f'[{name}] elapsed={elapsed:.0f}s estimated_remaining={max(0,estimate-elapsed):.0f}s (estimate)',flush=True)
                with log.open('rb') as f:
                    f.seek(max(0,log.stat().st_size-1500));lines=f.read().decode(errors='replace').replace('\r','\n').splitlines()
                if lines and lines[-1]!=last_line:print(lines[-1][-400:],flush=True);last_line=lines[-1]
                time.sleep(10)
        if child.returncode:
            print(log.read_text(errors='replace')[-7000:],flush=True)
            raise subprocess.CalledProcessError(child.returncode,command)
        print(f'DONE {name}; elapsed={time.monotonic()-started:.1f}s',flush=True)
    print(f'RUN={run}\nReuse all 33 completed training jobs / 13 weights; no retraining. GPUs={GPUS}',flush=True)
    try:
        with tqdm(total=4,desc='Resume evaluation',unit='step',dynamic_ncols=True) as progress:
            for split,estimate in [('valid',450),('test',250)]:
                call([sys.executable,'-B',impl/'timed_predict.py','--source',impl/'source','--weights',run/'model/weights',
                      '--images',ROOT/f'data/{split}/images','--output',run/f'pred_{split}','--gpus',env['CUDA_VISIBLE_DEVICES'],'--split',split],f'predict_{split}',estimate)
                progress.update(1)
            command=[sys.executable,'-B',impl/'source/tools/score_masks.py','--labels',run/'pred_valid/fusion/predicted_labels',
                     '--gt',ROOT/'data/valid/masks','--output',run/'pred_valid/metrics.json']
            start=time.perf_counter();call(command,'score_valid',5)
            write(run/'stages/score_valid.json',dict(name='score_valid',status='done',returncode=0,command=list(map(str,command)),
                  artifact=str(run/'pred_valid/metrics.json'),artifact_sha256=sha(run/'pred_valid/metrics.json'),wall_seconds=time.perf_counter()-start,finished_at=now()))
            progress.update(1)
            seconds=time.perf_counter()-t0;parent_seconds=read(parent/'status.json')['wall_seconds']
            write(run/'pipeline_completion.json',dict(ok=True,wall_seconds=seconds,parent_attempt_wall_seconds=parent_seconds,
                  total_active_wall_seconds=parent_seconds+seconds,reused_training_count=33,new_training_count=0,finished_at=now()))
            call([sys.executable,'-B',impl/'export_results.py','--run',run],'export_results',60)
            progress.update(1)
        write(run/'status.json',dict(status='complete',finished_at=now(),result=read(run/'result.json')['result'],wall_seconds=time.perf_counter()-t0,reused_training=33,new_training=0))
        print('COMPLETE '+read(run/'result.json')['result'],flush=True)
    except BaseException as error:
        if child is not None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        write(run/'status.json',dict(status='stopped_or_failed',error=str(error),finished_at=now(),wall_seconds=time.perf_counter()-t0))
        raise

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['check','plan','resume']);p.add_argument('--execute',action='store_true');p.add_argument('--lock-held',action='store_true');a=p.parse_args()
    if a.action=='plan':print((ROOT/'parent.json').read_text());return
    if a.action=='check' or not a.execute:
        from preflight import check
        result=check();write(ROOT/'checks/preflight.json',result);print(result);return
    execute(a.lock_held)
if __name__=='__main__':main()
