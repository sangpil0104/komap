"""Conditional startup-cost reduction after the preceding bounded experiment."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import fcntl, importlib.util, shutil, statistics, subprocess, time
from pathlib import Path
import yaml
ROOT=Path('/workspace/model349_compress/base12_singleworker_20261008')
PREVIOUS=Path('/workspace/model349_compress/base12_resident_packed_20261008')
UTIL=Path('/workspace/model349_compress/base12_followup_20261008')
spec=importlib.util.spec_from_file_location('previous',PREVIOUS/'run_experiment.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e);u=e.u;u.ROOT=ROOT

def main():
    while not any((PREVIOUS/n).exists() for n in ['completion.json','failure.json']):time.sleep(5)
    if (PREVIOUS/'failure.json').exists():
        u.write(ROOT/'skipped.json',{'reason':'preceding experiment failed; do not overlap or extend'});return
    done=u.read(PREVIOUS/'completion.json');assert done['base_and_weights_unchanged']
    raw=u.read(PREVIOUS/'raw_comparison.json')['packed_fast12']
    if not (raw['vs_base12']['quality_pass'] and raw['vs_lite15']['quality_pass']):
        u.write(ROOT/'skipped.json',{'reason':'preceding packed quality failed'});return
    rows=u.read(PREVIOUS/'timing_runs.json')
    values={k:[r['completion']['elapsed_seconds'] for r in rows if r['candidate']==k] for k in ['control12','packed_fast12']}
    assert all(len(v)==3 for v in values.values())
    reduction=100*(1-statistics.median(values['packed_fast12'])/statistics.median(values['control12']))
    quality=all(r['vs_base12']['quality_pass'] and r['vs_lite15']['quality_pass'] for r in rows)
    frames=[u.json.loads(x) for x in (PREVIOUS/'monitor.jsonl').read_text().splitlines()]
    uuid=next(line.split(',')[1].strip() for line in frames[0]['gpu'].splitlines() if line.startswith('7,'))
    interference=any(sum(uuid in line for line in f['processes'].splitlines())>1 for f in frames)
    u.write(ROOT/'preceding_gate.json',{'median_reduction_percent':reduction,'all_quality_pass':quality,'gpu7_interference':interference})
    if reduction>=10 and quality and not interference:
        u.write(ROOT/'skipped.json',{'reason':'preceding packed candidate already meets fixed time gate'});return
    locks=[]
    names=['base12_followup_20261008','base12_inputcache_20261008','base12_highflip4_20261008',PREVIOUS.name,ROOT.name]
    for p in [Path('/workspace/model349_compress/audit_20261008/gpu7.lock')]+[Path('/workspace/model349_compress')/n/'gpu7.lock' for n in names]:
        f=p.open('a')
        for attempt in range(12):
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if attempt==11:raise
                time.sleep(5)
        locks.append(f)
    if subprocess.run(['pgrep','-f','[i]nfer_lite.py|[i]nfer_audited.py|[i]nfer_batched.py|[i]nfer_cpu2.py'],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('Existing inference')
    u.idle();(ROOT/'runs').mkdir(exist_ok=False)
    frozen=u.snapshot();assert frozen==u.read(PREVIOUS/'frozen_before.json');u.write(ROOT/'frozen_before.json',frozen)
    control=u.build_bundle('control12')
    src=PREVIOUS/'bundles/packed_fast12';bundle=ROOT/'bundles/singleworker12';bundle.mkdir()
    shutil.copytree(src/'src',bundle/'src');shutil.copytree(src/'configs',bundle/'configs');shutil.copy2(src/'infer_lite.py',bundle/'infer_lite.py')
    manifest=u.read(src/'checkpoint_manifest.json')
    for row in manifest['checkpoints']:
        source=src/row['path'];assert u.sha(source)==row['sha256'] and source.stat().st_size==row['bytes']
        p=bundle/row['path'];p.parent.mkdir(parents=True,exist_ok=True);p.symlink_to(source)
        if (source.parent/'args.json').exists():shutil.copy2(source.parent/'args.json',p.parent/'args.json')
    u.write(bundle/'checkpoint_manifest.json',manifest)
    shutil.copy2(src/'lossless_storage_verification.json',bundle/'lossless_storage_verification.json')
    shutil.copy2(PREVIOUS/'canonical_lossless_storage_verification.json',ROOT/'canonical_lossless_storage_verification.json')
    p=bundle/'infer_lite.py';code=p.read_text()
    before='def worker(work, component):\n    import torch'
    after='''def worker(work, component):
    if component == 'all':
        import contextlib
        import torch
        config = json.loads((work / 'input_manifest.json').read_text())['config']
        physical_gpu = os.environ['CUDA_VISIBLE_DEVICES']
        uuid = subprocess.check_output(['nvidia-smi', '-i', physical_gpu, '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
        for name in active_components(config):
            processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True)
            if sum(uuid in line for line in processes.splitlines()) > 1:
                raise RuntimeError('Another compute process on our GPU; stopping our own worker')
            print('ALL_WORKER START', name, flush=True)
            with (work / 'logs' / (name + '.log')).open('x') as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    worker(work, name)
            if name == 'base':
                import tent_style_search
                tent_style_search._resident_model = None
                tent_style_search._resident_device = None
                tent_style_search._pristine_states.clear()
            gc.collect()
            torch.cuda.empty_cache()
            print('ALL_WORKER COMPLETE', name, flush=True)
        return
    import torch'''
    assert code.count(before)==1;code=code.replace(before,after)
    before='    pending = list(active_components(config))'
    after="    if len(gpus) != 1:\n        raise ValueError('single-worker recipe requires exactly one GPU')\n    pending = ['all']"
    assert code.count(before)==1;code=code.replace(before,after)
    code=code.replace("choices=COMPONENTS, help=argparse.SUPPRESS", "choices=COMPONENTS + ['all'], help=argparse.SUPPRESS")
    code=code.replace('with tqdm(total=len(active_components(config)),', 'with tqdm(total=len(pending),')
    code=code.replace("'training_started': False,\n               'tent_steps':", "'training_started': False, 'worker_mode': 'single_subprocess',\n               'tent_steps':")
    compile(code,str(p),'exec');p.write_text(code);e.code_hashes(bundle)
    monitor=(UTIL/'monitor.py').read_text().replace(str(UTIL),str(ROOT));(ROOT/'monitor.py').write_text(monitor)
    subprocess.Popen([u.PYTHON,'-B',str(ROOT/'monitor.py')],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    probe=(PREVIOUS/'probe.py').read_text().replace(str(PREVIOUS),str(ROOT)).replace('bundles/resident12','bundles/singleworker12')
    (ROOT/'probe.py').write_text(probe)
    with (ROOT/'probe.log').open('x') as log:subprocess.run([u.PYTHON,'-B','-u',str(ROOT/'probe.py')],env=dict(os.environ,CUDA_VISIBLE_DEVICES='7'),stdout=log,stderr=subprocess.STDOUT,check=True)
    run=u.execute(bundle,'singleworker12_valid20')
    comparison={'metrics':u.metrics(run),'completion':u.read(run/'completion.json'),'vs_base12':u.pair(u.metrics(u.OLD),u.metrics(run)),'vs_lite15':u.pair(u.metrics(u.OLD.parent/'cpu2_lite15'),u.metrics(run)),'checkpoints':manifest}
    u.write(ROOT/'raw_comparison.json',{'singleworker12':comparison})
    passed=comparison['vs_base12']['quality_pass'] and comparison['vs_lite15']['quality_pass']
    timings=[]
    if passed:
        for index,order in enumerate([['control12','singleworker12'],['singleworker12','control12'],['control12','singleworker12']],1):
            for key in order:
                run=u.execute(control if key=='control12' else bundle,f'timing_{index}_{key}')
                timings.append({'run':run.name,'candidate':key,'completion':u.read(run/'completion.json'),'vs_base12':u.pair(u.metrics(u.OLD),u.metrics(run)),'vs_lite15':u.pair(u.metrics(u.OLD.parent/'cpu2_lite15'),u.metrics(run))});u.write(ROOT/'timing_runs.json',timings)
        u.execute(bundle,'singleworker12_test10',test=True)
    after=u.snapshot();assert after==frozen;u.write(ROOT/'frozen_after.json',after)
    assert all(u.sha(src/r['path'])==r['sha256'] for r in manifest['checkpoints'])
    u.write(ROOT/'completion.json',{'diagnostics_completed':True,'base_and_weights_unchanged':True,'raw_valid_candidates':['singleworker12'],'time_candidate':'singleworker12' if passed else None,'test_candidates':['singleworker12'] if passed else [],'independent_generalization_verified':False,'new_training_started':False,'preceding_packed_checkpoints_unchanged':True})
    print('SESSION_COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as exc:u.write(ROOT/'failure.json',{'error':repr(exc),'complete':False});raise
