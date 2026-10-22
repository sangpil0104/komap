"""Resume after successful restore probe and pre-inference idle-check failure."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import fcntl,importlib.util,subprocess,time
from pathlib import Path
ROOT=Path('/workspace/model349_compress/base12_singleworker_20261008')
spec=importlib.util.spec_from_file_location('single',ROOT/'run_single.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s);u=s.u

def stable_idle():
 uuid=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();consecutive=0
 for _ in range(16):
  procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
  if any(uuid in line for line in procs.splitlines()):raise RuntimeError('Another compute process on GPU7; our next worker was not started')
  raw=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  mem,util=map(int,raw.strip().split(','));consecutive=consecutive+1 if mem<=64 and util==0 else 0
  if consecutive>=3:return
  time.sleep(2)
 raise RuntimeError('GPU7 failed 3 consecutive idle checks within 32s; no worker started')

def main():
 assert not (ROOT/'completion.json').exists()
 assert u.read(ROOT/'probe_result.json')
 assert list((ROOT/'runs').iterdir())==[], 'Resume only the pre-inference failure; no duplicate/partial runs'
 locks=[]
 names=['audit_20261008','base12_followup_20261008','base12_inputcache_20261008','base12_highflip4_20261008',s.PREVIOUS.name,ROOT.name]
 for n in names:
  f=(ROOT.parent/n/'gpu7.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 if subprocess.run(['pgrep','-f','[i]nfer_lite.py|[i]nfer_audited.py|[i]nfer_batched.py|[i]nfer_cpu2.py'],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('Existing inference')
 u.idle=stable_idle;u.idle();frozen=u.snapshot();assert frozen==u.read(ROOT/'frozen_before.json')
 assert (ROOT/'failure_idle_initial.json').exists() and not (ROOT/'failure.json').exists()
 bundles={k:ROOT/'bundles'/k for k in ['control12','singleworker12']}
 for bundle in bundles.values():
  manifest=u.read(bundle/'checkpoint_manifest.json')
  assert all(u.sha(bundle/r['path'])==r['sha256'] for r in manifest['checkpoints'])
 monitor=subprocess.Popen([u.PYTHON,'-B',str(ROOT/'monitor.py')],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
 run=u.execute(bundles['singleworker12'],'singleworker12_valid20');m=u.metrics(run)
 comparison={'metrics':m,'completion':u.read(run/'completion.json'),'vs_base12':u.pair(u.metrics(u.OLD),m),'vs_lite15':u.pair(u.metrics(u.OLD.parent/'cpu2_lite15'),m),'checkpoints':u.read(bundles['singleworker12']/'checkpoint_manifest.json')}
 u.write(ROOT/'raw_comparison.json',{'singleworker12':comparison});passed=comparison['vs_base12']['quality_pass'] and comparison['vs_lite15']['quality_pass'];timings=[]
 if passed:
  for index,order in enumerate([['control12','singleworker12'],['singleworker12','control12'],['control12','singleworker12']],1):
   for key in order:
    run=u.execute(bundles[key],f'timing_{index}_{key}');m=u.metrics(run)
    timings.append({'run':run.name,'candidate':key,'completion':u.read(run/'completion.json'),'vs_base12':u.pair(u.metrics(u.OLD),m),'vs_lite15':u.pair(u.metrics(u.OLD.parent/'cpu2_lite15'),m)});u.write(ROOT/'timing_runs.json',timings)
  u.execute(bundles['singleworker12'],'singleworker12_test10',test=True)
 after=u.snapshot();assert after==frozen;u.write(ROOT/'frozen_after.json',after)
 u.write(ROOT/'completion.json',{'diagnostics_completed':True,'base_and_weights_unchanged':True,'raw_valid_candidates':['singleworker12'],'time_candidate':'singleworker12' if passed else None,'test_candidates':['singleworker12'] if passed else [],'independent_generalization_verified':False,'new_training_started':False,'preceding_packed_checkpoints_unchanged':True,'resumed_after_pre_inference_failure':True})
 print('SESSION_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as exc:u.write(ROOT/'failure.json',{'error':repr(exc),'complete':False});raise
