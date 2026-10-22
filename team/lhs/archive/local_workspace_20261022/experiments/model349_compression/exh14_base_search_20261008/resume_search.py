"""Resume only fast14 probe/inference after fixed control12->control14 probe path."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import fcntl,importlib.util,subprocess
from pathlib import Path
ROOT=Path('/workspace/model349_compress/exh14_base_search_20261008')
spec=importlib.util.spec_from_file_location('search',ROOT/'run_search.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);u=m.u

def main():
 locks=[]
 for name in ['audit_20261008','base12_followup_20261008','base12_inputcache_20261008','base12_highflip4_20261008','base12_resident_packed_20261008','base12_singleworker_20261008','base12_partial_distill329_20261008',ROOT.name]:
  f=(ROOT.parent/name/'gpu7.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 if subprocess.run(['pgrep','-f','[i]nfer_lite.py|[r]un_search.py|[t]rain_pilot.py'],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('Existing inference/controller; no duplicate resume')
 assert not (ROOT/'completion.json').exists() and not (ROOT/'failure.json').exists()
 assert not (ROOT/'runs/fast14_valid20').exists()
 frozen=u.read(ROOT/'frozen_before.json');assert m.freeze()==frozen
 u.idle();control=ROOT/'bundles/control14';u.BASE=control;u.OLD=ROOT/'runs/control14_valid20';reference=u.metrics(u.OLD);results=u.read(ROOT/'raw_comparison.json');assert results['fp16_storage14']['vs_base14']['quality_pass'];bundles={k:ROOT/'bundles'/k for k in ['fp16_storage14','fast14']}
 for bundle in bundles.values():
  assert all(u.sha(bundle/r['path'])==r['sha256'] for r in u.read(bundle/'checkpoint_manifest.json')['checkpoints'])
 subprocess.Popen([u.PYTHON,'-B',str(ROOT/'monitor.py')],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
 with (ROOT/'probe_resumed.log').open('x') as log:subprocess.run([u.PYTHON,'-B',str(ROOT/'probe.py')],env=dict(os.environ,CUDA_VISIBLE_DEVICES='7'),stdout=log,stderr=subprocess.STDOUT,check=True)
 run=u.execute(bundles['fast14'],'fast14_valid20');metrics=u.metrics(run);results['fast14']={'metrics':metrics,'completion':u.read(run/'completion.json'),'vs_base14':u.pair(reference,metrics),'vs_base12':u.pair(u.metrics(ROOT.parent/'audit_20261008/runs/cpu2_fp16_12'),metrics),'checkpoints':u.read(bundles['fast14']/'checkpoint_manifest.json')};u.write(ROOT/'raw_comparison.json',results)
 selected=next(k for k in ['fast14','fp16_storage14'] if results[k]['vs_base14']['quality_pass']);timings=[]
 for index,order in enumerate([['control14',selected],[selected,'control14'],['control14',selected]],1):
  for key in order:
   run=u.execute(control if key=='control14' else bundles[key],f'timing_{index}_{key}');metrics=u.metrics(run);timings.append({'run':run.name,'candidate':key,'completion':u.read(run/'completion.json'),'vs_base14':u.pair(reference,metrics)});u.write(ROOT/'timing_runs.json',timings)
 u.execute(bundles[selected],selected+'_test10',test=True)
 after=m.freeze();assert after==frozen;u.write(ROOT/'frozen_after.json',after);u.write(ROOT/'completion.json',{'base_and_weights_unchanged':True,'raw_valid_candidates':list(results),'time_candidate':selected,'test_candidates':[selected],'reference':'user-selected base14','independent_generalization_verified':False,'new_training_started':False,'resumed_after_probe_path_error':True});print('SESSION_COMPLETE',flush=True)
 with (ROOT/'finalize.log').open('x') as log:subprocess.run([u.PYTHON,'-B',str(ROOT/'finalize.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':
 try:main()
 except Exception as exc:u.write(ROOT/'failure.json',{'error':repr(exc),'complete':False});raise
