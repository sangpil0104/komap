"""Resume after schema-only ZIP audit failure; do not rerun completed candidates."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import fcntl, importlib.util, subprocess
from pathlib import Path
ROOT = Path('/workspace/model349_compress/base12_resident_packed_20261008')
spec = importlib.util.spec_from_file_location('experiment', ROOT / 'run_experiment.py')
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e); u = e.u
def main():
    locks=[]
    paths=['/workspace/model349_compress/audit_20261008/gpu7.lock']+[f'/workspace/model349_compress/{n}/gpu7.lock' for n in ['base12_followup_20261008','base12_inputcache_20261008','base12_highflip4_20261008',ROOT.name]]
    for path in paths:
        stream=open(path,'a'); fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB); locks.append(stream)
    if subprocess.run(['pgrep','-f','[i]nfer_lite.py|[i]nfer_audited.py|[i]nfer_batched.py|[i]nfer_cpu2.py'],stdout=subprocess.DEVNULL).returncode==0: raise RuntimeError('Existing inference')
    u.idle(); assert u.snapshot()==u.read(ROOT/'frozen_before.json')
    comparisons=u.read(ROOT/'raw_comparison.json')
    assert set(comparisons)=={'soup11','resident12','hramp12'}
    assert all(comparisons[k]['vs_base12']['quality_pass'] and comparisons[k]['vs_lite15']['quality_pass'] for k in ['resident12','hramp12'])
    assert not comparisons['soup11']['vs_base12']['quality_pass']
    failure=u.read(ROOT/'failure.json'); assert failure['error']=="KeyError('model')"
    (ROOT/'failure.json').rename(ROOT/'failure_storage_audit_1.json')
    subprocess.Popen([u.PYTHON,'-B',str(ROOT/'monitor.py')],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    packed=ROOT/'bundles/packed_fast12'
    e.deflate(packed); e.code_hashes(packed)
    run=u.execute(packed,'packed_fast12_valid20')
    comparisons['packed_fast12']={'metrics':u.metrics(run),'completion':u.read(run/'completion.json'),'vs_base12':u.pair(u.metrics(u.OLD),u.metrics(run)),'vs_lite15':u.pair(u.metrics(u.OLD.parent/'cpu2_lite15'),u.metrics(run)),'checkpoints':u.read(packed/'checkpoint_manifest.json')}
    u.write(ROOT/'raw_comparison.json',comparisons)
    passed=lambda k:k in comparisons and comparisons[k]['vs_base12']['quality_pass'] and comparisons[k]['vs_lite15']['quality_pass']
    selected=next((k for k in ['packed_fast11','packed_fast12','soup11','resident12','hramp12'] if passed(k)),None)
    u.write(ROOT/'timing_selection.json',{'candidate':selected,'selection':'original preregistered priority; schema-only audit fix'})
    timings=[]
    if selected:
        for index,order in enumerate([['control12',selected],[selected,'control12'],['control12',selected]],1):
            for key in order:
                run=u.execute(ROOT/'bundles'/key,f'timing_{index}_{key}')
                timings.append({'run':run.name,'candidate':key,'completion':u.read(run/'completion.json'),'vs_base12':u.pair(u.metrics(u.OLD),u.metrics(run)),'vs_lite15':u.pair(u.metrics(u.OLD.parent/'cpu2_lite15'),u.metrics(run))})
                u.write(ROOT/'timing_runs.json',timings)
        u.execute(ROOT/'bundles'/selected,selected+'_test10',test=True)
    after=u.snapshot();u.write(ROOT/'frozen_after.json',after);assert after==u.read(ROOT/'frozen_before.json')
    u.write(ROOT/'completion.json',{'diagnostics_completed':True,'base_and_weights_unchanged':True,'raw_valid_candidates':list(comparisons),'time_candidate':selected,'test_candidates':[selected] if selected else [],'independent_generalization_verified':False,'new_training_started':False,'storage_schema_failure_recovered':True})
    print('SESSION_COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as exc:
        u.write(ROOT/'failure.json',{'error':repr(exc),'complete':False});raise
