"""Sequential GPU7 component replay; never claim whole-model timing."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import copy,fcntl,hashlib,importlib.util,json,shutil,subprocess,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent;SOURCE=PARENT/'Model349_exh14'
REF=PARENT/'exh14_base_search_20261008/runs/control14_valid20'
TTA=PARENT/'accuracy81_20261008';PY='/workspace/venv-komap/bin/python'
GT=Path('/workspace/komap_model1/data/valid/masks')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def load(p):return json.loads(p.read_text())
def live_tta():
 p=subprocess.run(['ps','-p','2158061','-o','args='],capture_output=True,text=True)
 if p.returncode==0:
  assert str(TTA/'run_remaining.py') in p.stdout, 'PID reused; refuse dependent execution'
  return True
 return False
def idle():
 for _ in range(3):
  mem,util=map(int,subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
  apps=subprocess.check_output(['nvidia-smi','-i','7','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
  assert mem<=64 and util==0 and not apps,(mem,util,apps)
  time.sleep(2)
def main():
 assert ROOT.name=='accuracy81_hrscale_20261008'
 write(ROOT/'source_lock.json',{p.name:sha(p) for p in ROOT.glob('*') if p.suffix in ['.py','.md']})
 while live_tta():
  write(ROOT/'queue_status.json',{'waiting_pid':2158061,'observed_live':True,'time':time.time(),'gpu_started':False});time.sleep(30)
 assert (TTA/'completion.json').exists() and load(TTA/'completion.json')['sequence_finished']
 if max(load(p)['image_miou'] for p in (TTA/'runs').glob('*/metrics.json'))>=.81:
  write(ROOT/'skipped.json',{'reason':'completed TTA reached81; collect/reproduce first'});return
 locks=[]
 for name in ['audit_20261008','base12_followup_20261008','base12_inputcache_20261008','base12_highflip4_20261008','base12_resident_packed_20261008','base12_singleworker_20261008','base12_partial_distill329_20261008','exh14_base_search_20261008','accuracy81_20261008',ROOT.name]:
  f=(PARENT/name/'gpu7.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 idle()
 frozen=load(TTA/'frozen_before.json');assert all(sha(SOURCE/p)==h for p,h in frozen.items())
 ref=load(REF/'input_manifest.json');expected=load(REF/'metrics.json');assert expected['count']==20
 manifest=load(PARENT/'exh14_base_search_20261008/bundles/control14/checkpoint_manifest.json')
 assert len(manifest['checkpoints'])==14 and set(ref['required_weights'])==set(r['path'] for r in manifest['checkpoints'])
 for row in ref['images']:
  assert sha(Path(row['path']))==row['sha256'],row['name']
 prob_hash={str(p.relative_to(REF)):sha(p) for p in (REF/'probabilities').rglob('*') if p.is_file()}
 assert len(list((REF/'probabilities/336').glob('*.npy')))==20
 bundle=ROOT/'bundle';shutil.copytree(SOURCE,bundle,ignore=shutil.ignore_patterns('weights','__pycache__','.git','results','search'))
 for row in manifest['checkpoints']:
  p=bundle/row['path'];p.parent.mkdir(parents=True,exist_ok=True);p.symlink_to(SOURCE/row['path'])
  args=(SOURCE/row['path']).parent/'args.json'
  if args.exists():shutil.copy2(args,p.parent/'args.json')
 write(bundle/'checkpoint_manifest.json',manifest)
 assert sha(bundle/ref['config']['calibration'])==ref['calibration_sha256']
 spec=importlib.util.spec_from_file_location('scale_inference',bundle/'infer_lite.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 import cv2
 cv2.setNumThreads(2)
 def setup(name,scale=None,replace336=False):
  run=ROOT/'runs'/name;run.mkdir(parents=True,exist_ok=False);(run/'logs').mkdir();(run/'probabilities').mkdir()
  m=copy.deepcopy(ref)
  if scale is not None:
   m['config']['inference']['hrnet']['input_scale']=scale
   h=m['config']['inference']['hrnet'];h['overlap']=round(round(h['patch']/scale)*170/256)
   assert round(h['patch']/scale)-h['overlap']>=50
  write(run/'input_manifest.json',m)
  for component in ['322','326','329','base','336','347','269']:
   if replace336 and component=='336':continue
   (run/'probabilities'/component).symlink_to(REF/'probabilities'/component,target_is_directory=True)
  return run
 def assemble(run,elapsed,kind):
  scores=module.assemble(run,GT)
  write(run/'completion.json',{'model':'Model349','images':20,'elapsed_seconds':elapsed,'elapsed_kind':kind,'training_started':False,'tent_steps':5,'gpus':['7'],'inference_sha256':sha(bundle/'infer_lite.py'),'mean_image_miou':float(np.mean([r['image_miou'] for r in scores])),'component_replay':True,'reference_run':str(REF)})
  return load(run/'metrics.json')
 baseline=setup('baseline_replay_valid20');start=time.monotonic();actual=assemble(baseline,0.,'assembly only')
 assert np.array_equal(np.array([x['class_iou'] for x in actual['images']]),np.array([x['class_iou'] for x in expected['images']])),'baseline replay differs'
 write(ROOT/'baseline_replay.json',{'exact_80_iou_match':True,'reference_miou':expected['image_miou'],'probability_hashes':prob_hash,'source_frozen':frozen,'assembly_seconds':time.monotonic()-start})
 for scale in [1.,3.]:
  idle();run=setup(f'scale{int(scale)}_valid20',scale,True);start=time.monotonic()
  with (ROOT/f'scale{int(scale)}.log').open('x') as log:
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',NO_ALBUMENTATIONS_UPDATE='1')
   proc=subprocess.Popen([PY,'-B','-u',str(bundle/'infer_lite.py'),'--_worker','336','--output',str(run)],cwd=bundle,env=env,stdout=log,stderr=subprocess.STDOUT)
   while proc.poll() is None:
    time.sleep(10)
    apps=subprocess.check_output(['nvidia-smi','-i','7','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    if len(apps)>1:
     proc.terminate();proc.wait(timeout=15);raise RuntimeError('Foreign GPU7 compute appeared; stopped our worker only')
   assert proc.returncode==0,proc.returncode
  duration=time.monotonic()-start
  assert len(list((run/'probabilities/336').glob('*.npy')))==20
  metrics=assemble(run,duration,'new HR336 component only; excludes replayed branches and fusion')
  print('SCORE',scale,100*metrics['image_miou'],flush=True)
 mixed=setup('mean123_valid20',replace336=True);(mixed/'probabilities/336').mkdir();start=time.monotonic()
 for row in ref['images']:
  name=row['name'];values=[np.load(REF/'probabilities/336'/f'{name}.npy').astype(np.float32)]+[np.load(ROOT/f'runs/scale{scale}_valid20/probabilities/336'/f'{name}.npy').astype(np.float32) for scale in [1,3]]
  assert all(v.shape==values[0].shape and np.isfinite(v).all() for v in values)
  np.save(mixed/'probabilities/336'/f'{name}.npy',(sum(values)/3).astype(np.float16))
 metrics=assemble(mixed,time.monotonic()-start,'CPU averaging only; excludes all GPU work')
 print('SCORE mean123',100*metrics['image_miou'],flush=True)
 assert all(sha(REF/p)==h for p,h in prob_hash.items()) and all(sha(SOURCE/p)==h for p,h in frozen.items())
 write(ROOT/'completion.json',{'sequence_finished':True,'original_source_unchanged':True,'reference_probabilities_unchanged':True,'metric':'image-wise four classes; empty union=0','time':time.time()})
if __name__=='__main__':
 try:main()
 except BaseException:
  write(ROOT/'failure.json',{'traceback':traceback.format_exc(),'time':time.time()});raise
