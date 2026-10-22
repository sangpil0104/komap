"""One locked GPU7: fresh teacher train70, native residual training, fresh valid20."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import fcntl,json,signal,subprocess,time,traceback,shutil
from fields_io import SOURCE,sha,write,prepare
ROOT=Path(__file__).resolve().parent;PARENT=ROOT.parent;PY='/workspace/venv-komap/bin/python'
def idle():
 while True:
  clear=True
  for _ in range(3):
   mem,util=map(int,subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','));apps=subprocess.check_output(['nvidia-smi','-i','7','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
   if mem>64 or util or apps:clear=False;break
   time.sleep(2)
  if clear:return
  write(ROOT/'queue_status.json',{'state':'waiting_gpu7_idle','memory_mib':mem,'util':util,'compute_pids':apps,'time':time.time()});time.sleep(30)
def run(args,name,cwd=ROOT):
 idle();env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',NO_ALBUMENTATIONS_UPDATE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 with (ROOT/(name+'.log')).open('x') as log:
  p=subprocess.Popen([PY,'-B','-u',*map(str,args)],cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write(ROOT/'active_child.json',{'pid':p.pid,'stage':name,'argv':list(map(str,args)),'time':time.time()})
  while p.poll() is None:
   time.sleep(10);apps=subprocess.check_output(['nvidia-smi','-i','7','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip().splitlines()
   if len(apps)>1:
    os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=20);raise RuntimeError('Foreign GPU7 compute appeared; stopped our child group only')
  assert p.returncode==0,(name,p.returncode)
def main():
 assert shutil.disk_usage(ROOT).free>15*1024**3
 for pid in ['2158061','2162847']:
  assert subprocess.run(['ps','-p',pid,'-o','args='],capture_output=True).returncode!=0,'prior GPU7 controller still live'
 assert json.loads((PARENT/'accuracy81_hrscale_20261008/completion.json').read_text())['original_source_unchanged']
 locks=[]
 for name in ['audit_20261008','base12_followup_20261008','base12_inputcache_20261008','base12_highflip4_20261008','base12_resident_packed_20261008','base12_singleworker_20261008','base12_partial_distill329_20261008','exh14_base_search_20261008','accuracy81_20261008','accuracy81_hrscale_20261008',ROOT.name]:
  f=(PARENT/name/'gpu7.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 frozen=json.loads((PARENT/'accuracy81_20261008/frozen_before.json').read_text());assert all(sha(SOURCE/p)==h for p,h in frozen.items());write(ROOT/'parent_frozen.json',frozen)
 write(ROOT/'prepared_source_lock.json',{p.name:sha(p) for p in ROOT.glob('*') if p.suffix in ['.py','.md']})
 (ROOT/'runs').mkdir();(ROOT/'features').mkdir();start=time.monotonic()
 def teacher(split,count):
  out=ROOT/'runs'/f'teacher_{split}{count}'
  run([SOURCE/'infer_lite.py','--input',f'/workspace/komap_model1/data/{split}/images','--output',out,'--gpus','7','--keep-probabilities','--gt',f'/workspace/komap_model1/data/{split}/masks'],f'teacher_{split}',SOURCE)
  assert json.loads((out/'completion.json').read_text())['images']==count
  return out
 teacher_train=teacher('train',70);prepare(teacher_train,ROOT/'features/train','train');run([ROOT/'train_residual.py'],'train_residual')
 receipt=json.loads((ROOT/'training/completion.json').read_text());assert receipt['epochs']==8 and receipt['optimizer_updates']==1120
 validation_start=time.monotonic();teacher_valid=teacher('valid',20);prepare(teacher_valid,ROOT/'features/valid','valid');run([ROOT/'apply_residual.py'],'apply_residual')
 p=ROOT/'runs/refined_valid20/completion.json';d=json.loads(p.read_text());d['refiner_evaluation_seconds']=d['elapsed_seconds'];d['elapsed_seconds']=time.monotonic()-validation_start;d['elapsed_kind']='whole fresh parent valid20 + field extraction + residual evaluation; excludes training';write(p,d)
 assert d['zero_refiner_parent_masks_exact'] and all(sha(SOURCE/p)==h for p,h in frozen.items())
 write(ROOT/'completion.json',{'sequence_finished':True,'parent_source_weights_unchanged':True,'elapsed_seconds_including_training':time.monotonic()-start,'mean_image_miou':d['mean_image_miou'],'training_valid_test_labels_excluded':True,'independent_generalization_verified':False,'time':time.time()})
 print('COMPLETE residual',100*d['mean_image_miou'],flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:write(ROOT/'failure.json',{'traceback':traceback.format_exc(),'time':time.time()});raise
