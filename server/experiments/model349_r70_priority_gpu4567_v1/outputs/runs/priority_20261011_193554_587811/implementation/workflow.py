"""User-started sequence: drain search -> r70 -> eight seed evaluations -> resume search."""
import argparse, fcntl, os, shutil, signal, subprocess, sys, time
from pathlib import Path
from tqdm import tqdm
from priority_common import (ROOT, REPO, R70, SEED, SEED_RUN, PREVIEW, GPUS, IDS, read, write,
    sha, stamp, environment, require_komap, records, verify_files, inventory, tail, lock_busy)
from pause_search import pause

def validate_run_path(path, experiment):
    path = Path(path).resolve()
    if path.parent != experiment/'outputs/runs' or not (path/'plan.json').is_file():
        raise ValueError(f'Not a real experiment run: {path}')
    if read(path/'plan.json') != read(experiment/'plan.json'):
        raise ValueError(f'Run plan mismatch: {path}')
    return path

def verify_source():
    verify_files(read(ROOT/'code_manifest.json'), ROOT)
    verify_files(read(ROOT/'pinned_inputs.json'))
    for experiment in [R70, SEED]:
        verify_files(read(experiment/'code_manifest.json'), experiment)
    validate_run_path(SEED_RUN, SEED)

def check():
    require_komap(); verify_source()
    for experiment in [R70, SEED]:
        subprocess.run([str(experiment/'run.sh'), 'check'], check=True, env=environment())
    preview = read(PREVIEW); previous={r['id']:r for r in preview['candidates']}
    assert preview['run'] == str(SEED_RUN) and preview['count'] == 20
    for index, identifier in enumerate(IDS):
        row = read(SEED_RUN/f'completed/{identifier}.json')
        assert row['seed'] == previous[identifier]['seed'] == 3407+index
        assert row['group'] == '336' and abs(row['image_miou']-previous[identifier]['REF_image_miou']) < 1e-12
        assert set(row['context']['probabilities']) == {'336'}
        assert sha(row['checkpoint']) == row['artifacts'][row['checkpoint']]
    info = dict(ok=True, training_started=False, processes_signalled=False, gpus=inventory(),
                seed_run=str(SEED_RUN), target_seeds=list(range(3407,3415)),
                completed_training=sum(r['kind']=='train' for r in records(SEED_RUN).values()),
                free_gib=shutil.disk_usage(ROOT).free/1024**3)
    if info['free_gib'] < 200: raise RuntimeError('Need 200 GiB free to preserve all models, inference and resumed search.')
    write(ROOT/f'checks/preflight_{stamp()}.json', info)
    print('PREFLIGHT PASS. No process stopped and no training started.', flush=True)
    return info

def update(work, **values):
    state = read(work/'state.json'); state.update(values); write(work/'state.json', state); return state

def new_work():
    work = ROOT/'outputs/runs'/f'priority_{stamp()}'; work.mkdir(parents=True)
    implementation = work/'implementation'; implementation.mkdir()
    for name in read(ROOT/'code_manifest.json'):
        source = ROOT/name; dest = implementation/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    shutil.copy2(ROOT/'code_manifest.json', implementation/'code_manifest.json')
    (work/'logs').mkdir(); (work/'evaluated').mkdir()
    write(work/'state.json', dict(phase='created', seed_run=str(SEED_RUN), r70_run=None,
                                gpus=GPUS, seeds=list(range(3407,3415)), created=stamp()))
    write(ROOT/'outputs/latest_run.json', dict(run=str(work)))
    return work

def stop_child(child, *, group=False):
    if child.poll() is not None and not group: return
    try:
        if group: os.killpg(child.pid, signal.SIGTERM)
        else: child.send_signal(signal.SIGTERM)
    except ProcessLookupError: return
    try: child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        # A coordinator has separate worker groups: never pretend cleanup succeeded.
        if not group: raise RuntimeError(f'Child coordinator PID {child.pid} still active; workflow stopped.')
        os.killpg(child.pid, signal.SIGKILL); child.wait()

def real_r70_latest():
    latest = R70/'outputs/latest_run.json'
    return validate_run_path(read(latest)['run'], R70) if latest.exists() else None

def run_child(command, work, phase, lock_fd, *, status_path=None, discover_r70=False, group=False):
    log = work/f'logs/{phase}_{stamp()}.log'
    write(work/f'{phase}_command.json', command)
    update(work, phase=phase, active_log=str(log), phase_started=stamp())
    print(f'STAGE {phase} | log={log}', flush=True)
    with log.open('x') as handle:
        child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, cwd=REPO,
            env=environment(), start_new_session=True, pass_fds=(lock_fd,))
        update(work, child_pid=child.pid); start=time.monotonic(); last=-1e9
        try:
            while child.poll() is None:
                if discover_r70:
                    saved = read(work/'state.json').get('r70_run')
                    if not saved:
                        latest=real_r70_latest()
                        if latest:
                            update(work, r70_run=str(latest)); status_path=latest/'status.json'
                    else: status_path=Path(saved)/'status.json'
                if time.monotonic()-last >= 20:
                    values=dict(phase_elapsed_seconds=time.monotonic()-start)
                    if status_path and status_path.exists():
                        current=read(status_path)
                        values.update(child_phase=current.get('phase'), training_completed=current.get('training_completed'),
                                      estimated_remaining_seconds=current.get('estimated_remaining_seconds'))
                    update(work, **values)
                    eta=values.get('estimated_remaining_seconds')
                    eta_text=f'{eta/3600:.2f}h' if eta is not None else 'see task tqdm'
                    print(f'[{phase}] elapsed={(time.monotonic()-start)/60:.1f}m | ETA≈{eta_text} | {tail(log)}', flush=True)
                    last=time.monotonic()
                time.sleep(1)
            if child.returncode: raise RuntimeError(f'{phase} exited {child.returncode}: {log}\n{tail(log)}')
        except BaseException:
            stop_child(child, group=group); raise
        finally:
            update(work, child_pid=None)

def evaluate_pool(work, lock_fd):
    pending=[]; done={}
    for identifier in IDS:
        path=work/f'evaluated/{identifier}.json'
        if path.exists():
            row=read(path); verify_files(row['files_sha256'], Path(row['attempt'])); done[identifier]=row
        else: pending.append(identifier)
    jobs={}; start=time.monotonic(); last=-1e9
    update(work, phase='evaluate_eight_seeds')
    with tqdm(total=8, initial=len(done), desc='Eight r70 seed combinations', unit='seed', mininterval=5) as progress:
        try:
            while pending or jobs:
                for identifier, job in list(jobs.items()):
                    code=job['process'].poll()
                    if code is None: continue
                    job['handle'].close()
                    if code: raise RuntimeError(f'{identifier} exited {code}: {job["log"]}\n{tail(job["log"])}')
                    row=read(work/f'evaluated/{identifier}.json')
                    row['scheduler_seconds']=time.monotonic()-job['start']
                    write(work/f'evaluated/{identifier}.json', row)
                    done[identifier]=row; del jobs[identifier]; progress.update(1)
                devices=inventory(); busy={r['gpu'] for r in jobs.values()}
                external=[REPO/f'experiments/{name}/outputs/search.lock' for name in
                    ['model349_lite20_seed100_gpu4567_v1','model349_seed100_gpu4567_v1','model349_lite12_seed100_gpu4567_v1']]
                blocked=any(lock_busy(p) for p in [*external, R70/'outputs/train.lock'])
                free=[] if blocked else [g for g in GPUS if g not in busy and devices[g]['memory']<=64 and devices[g]['utilization']==0]
                for gpu in free:
                    if not pending: break
                    identifier=pending.pop(0); log=work/f'logs/{identifier}_{stamp()}.log'; handle=log.open('x')
                    env=environment(); env.update(PRIORITY_USER_STARTED='1', CUDA_VISIBLE_DEVICES=devices[gpu]['uuid'],
                                                   KOMAP_PHYSICAL_GPU=str(gpu), PYTHONHASHSEED='0')
                    command=[sys.executable, '-B', '-u', str(work/'implementation/evaluate_seeds.py'),
                             '--work', str(work), '--task', identifier]
                    try:
                        child=subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env,
                            cwd=REPO, start_new_session=True, pass_fds=(lock_fd,))
                    except BaseException:
                        handle.close(); raise
                    jobs[identifier]=dict(process=child, gpu=gpu, log=log, handle=handle, start=time.monotonic())
                    print(f'START {identifier} GPU={gpu} | {log}', flush=True)
                if time.monotonic()-last>=20:
                    durations=[r['scheduler_seconds'] for r in done.values() if 'scheduler_seconds' in r]
                    estimate=max(1200, sum(durations)/len(durations) if durations else 0)
                    remaining=(len(pending)*estimate+sum(max(0,estimate-(time.monotonic()-j['start'])) for j in jobs.values()))/4
                    update(work, evaluations_completed=len(done), estimated_remaining_seconds=remaining,
                        active=[dict(task=k,gpu=j['gpu'],pid=j['process'].pid,log=str(j['log'])) for k,j in jobs.items()])
                    print(f'EVALUATION {len(done)}/8 | elapsed={(time.monotonic()-start)/60:.1f}m | work ETA≈{remaining/60:.1f}m (queue time excluded)', flush=True)
                    if blocked: print('Waiting for the existing GPU experiment lock to be released.', flush=True)
                    for key,j in jobs.items(): print(f'[{key} GPU {j["gpu"]}] {tail(j["log"])}', flush=True)
                    last=time.monotonic()
                time.sleep(5)
        finally:
            for job in jobs.values():
                stop_child(job['process'], group=True); job['handle'].close()
            update(work, active=[])

def sequence(work, lock_fd):
    verify_source()
    verify_files(read(work/'implementation/code_manifest.json'), work/'implementation')
    if not (work/'pause_complete.json').exists():
        update(work, phase='drain_seed_search'); pause(work)
    if not (work/'r70_complete.json').exists():
        state=read(work/'state.json')
        r70_run=validate_run_path(state['r70_run'], R70) if state.get('r70_run') else real_r70_latest()
        command=[str(R70/'run.sh'), 'train', '--execute']
        if r70_run:
            update(work, r70_run=str(r70_run)); command += ['--resume', str(r70_run)]
        if lock_busy(SEED/'outputs/search.lock'):
            raise RuntimeError('The saved seed search was restarted separately; stop it before resuming this workflow.')
        run_child(command, work, 'r70_train_and_report', lock_fd, discover_r70=True)
        state=read(work/'state.json'); r70_run=validate_run_path(state['r70_run'] or real_r70_latest(), R70)
        update(work, r70_run=str(r70_run))
        receipt=read(r70_run/'completed/export.json'); verify_files(receipt['artifacts'])
        assert read(r70_run/'status.json')['phase']=='complete'
        write(work/'r70_complete.json', dict(run=str(r70_run), result=receipt['result'], artifacts=receipt['artifacts']))
    else:
        verify_files(read(work/'r70_complete.json')['artifacts'])
    if not (work/'evaluation_export.json').exists():
        evaluate_pool(work, lock_fd)
        run_child([sys.executable,'-B','-u',str(work/'implementation/evaluate_seeds.py'), '--work',str(work),'--export'],
                  work,'eight_seed_report',lock_fd,group=True)
    else:
        receipt=read(work/'evaluation_export.json'); verify_files(receipt['files_sha256'],Path(receipt['result']))
    if not (work/'seed_resume_complete.json').exists():
        run_child([str(SEED/'run.sh'),'train','--execute','--resume',str(SEED_RUN)],
                  work,'resume_original_seed100',lock_fd,status_path=SEED_RUN/'status.json')
        done=records(SEED_RUN)
        assert sum(r['kind']=='train' for r in done.values())==100 and 'export' in done
        assert read(SEED_RUN/'status.json')['phase']=='complete'
        verify_files(done['export']['artifacts'])
        write(work/'seed_resume_complete.json',dict(run=str(SEED_RUN),result=done['export']['result'],completed=100))
    result=dict(r70=read(work/'r70_complete.json')['result'],
                eight_seeds=read(work/'evaluation_export.json')['result'],
                seed100=read(work/'seed_resume_complete.json')['result'])
    write(work/'results_index.json',result)
    (work/'summary.md').write_text('# 우선순위 실험 완료\n\n'+'\n'.join(f'- {key}: [{Path(value).name}]({value}/{Path(value).name}.md)' for key,value in result.items())+'\n')
    update(work,phase='complete',results=result,finished=stamp(),estimated_remaining_seconds=0)
    print('ALL FOUR STAGES COMPLETE\n'+str(result),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['check','plan','run','status'],nargs='?',default='check')
    p.add_argument('--execute',action='store_true'); p.add_argument('--resume',type=Path); a=p.parse_args()
    require_komap()
    if (a.execute or a.resume) and a.command!='run': p.error('--execute/--resume requires run')
    if a.command=='plan':
        print('1. Drain/save/stop pinned seed100\n2. r70 train + report\n3. Evaluate seeds 3407..3414 with r70 B4 + report\n4. Resume original seed100 to 100/100\nPhysical GPUs: 4,5,6,7. Only run --execute starts this sequence.'); return
    if a.command=='status':
        latest=ROOT/'outputs/latest_run.json'
        print(read(Path(read(latest)['run'])/'state.json') if latest.exists() else 'Not started.'); return
    if a.command=='check' or not a.execute: check(); return
    verify_source(); (ROOT/'outputs').mkdir(exist_ok=True)
    with (ROOT/'outputs/workflow.lock').open('a+') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('This priority workflow or its child is already running.')
        latest=ROOT/'outputs/latest_run.json'
        work=a.resume.resolve() if a.resume else Path(read(latest)['run']) if latest.exists() else new_work()
        if work.parent != ROOT/'outputs/runs': raise ValueError('Resume path must belong to this workflow.')
        state=read(work/'state.json')
        if state['seed_run']!=str(SEED_RUN): raise ValueError('Pinned seed run changed.')
        if state['phase']=='complete':
            print('Already complete:', read(work/'results_index.json')); return
        print(f'WORKFLOW_DIRECTORY={work}',flush=True)
        def interrupted(signum,frame): raise KeyboardInterrupt(f'signal {signum}')
        for signum in [signal.SIGTERM,signal.SIGINT]: signal.signal(signum,interrupted)
        try: sequence(work,lock.fileno())
        except BaseException as error:
            update(work,phase='failed_or_interrupted',error=str(error),interrupted=stamp()); raise

if __name__=='__main__': main()
