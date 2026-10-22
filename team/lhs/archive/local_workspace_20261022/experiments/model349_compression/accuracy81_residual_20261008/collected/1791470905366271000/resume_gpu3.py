"""Resume after completed teacher train70; same learning recipe on idle GPU3."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import fcntl,json,signal,subprocess,time,traceback,shutil
from fields_io import SOURCE,sha,write,prepare
ROOT=Path(__file__).resolve().parent;PARENT=ROOT.parent;PY='/workspace/venv-komap/bin/python';GPU='3'
def idle():
 while True:
  clear=True
  for _ in range(3):
   mem,util=map(int,subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
   apps=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
   if mem>64 or util or apps:clear=False;break
   time.sleep(2)
  if clear:return
  write(ROOT/'queue_status.json',{'state':'waiting_gpu3_idle','memory_mib':mem,'util':util,'compute_pids':apps,'time':time.time()});time.sleep(30)
def run(args,name,cwd=ROOT):
 idle();env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,KOMAP_PHYSICAL_GPU=GPU,NO_ALBUMENTATIONS_UPDATE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 with (ROOT/(name+'.log')).open('x') as log:
  p=subprocess.Popen([PY,'-B','-u',*map(str,args)],cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write(ROOT/'active_child.json',{'pid':p.pid,'stage':name,'argv':list(map(str,args)),'physical_gpu':GPU,'time':time.time()})
  while p.poll() is None:
   time.sleep(10)
   apps=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip().splitlines()
   if len(apps)>1:
    os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=20);raise RuntimeError('Foreign GPU3 compute appeared; stopped our created child group only')
  assert p.returncode==0,(name,p.returncode)
def main():
 assert shutil.disk_usage(ROOT).free>15*1024**3
 assert subprocess.run(['ps','-p','2167874','-o','args='],capture_output=True).returncode!=0
 migration=json.loads((ROOT/'gpu_migration.json').read_text());assert migration['old_controller_confirmed_stopped'] and migration['training_not_started']
 assert not (ROOT/'training').exists() and not (ROOT/'completion.json').exists() and not (ROOT/'failure.json').exists()
 locks=[]
 paths=set(PARENT.glob('*/gpu3.lock'))|{ROOT/'gpu3.lock',ROOT/'gpu7.lock'}
 for p in sorted(paths):
  f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 frozen=json.loads((ROOT/'parent_frozen.json').read_text());assert all(sha(SOURCE/p)==h for p,h in frozen.items())
 manifest=json.loads((ROOT/'features/train/manifest.json').read_text());assert manifest['parent_masks_exact'] and manifest['split']=='train' and len(manifest['images'])==70
 assert all(sha(ROOT/'features/train'/(x['name']+'.npz'))==x['features_sha256'] for x in manifest['images'])
 assert json.loads((ROOT/'runs/teacher_train70/completion.json').read_text())['images']==70
 lock=json.loads((ROOT/'preflight_source_lock.json').read_text());assert all(sha(ROOT/k)==h for k,h in lock.items())
 start=time.monotonic();run([ROOT/'train_residual.py'],'train_residual')
 receipt=json.loads((ROOT/'training/completion.json').read_text());assert receipt['epochs']==8 and receipt['optimizer_updates']==1120
 validation_start=time.monotonic();out=ROOT/'runs/teacher_valid20'
 run([SOURCE/'infer_lite.py','--input','/workspace/komap_model1/data/valid/images','--output',out,'--gpus',GPU,'--keep-probabilities','--gt','/workspace/komap_model1/data/valid/masks'],'teacher_valid',SOURCE)
 assert json.loads((out/'completion.json').read_text())['images']==20
 prepare(out,ROOT/'features/valid','valid');run([ROOT/'apply_residual.py'],'apply_residual')
 p=ROOT/'runs/refined_valid20/completion.json';d=json.loads(p.read_text());d['refiner_evaluation_seconds']=d['elapsed_seconds'];d['elapsed_seconds']=time.monotonic()-validation_start;d['elapsed_kind']='whole fresh parent valid20 + field extraction + residual evaluation; excludes training';write(p,d)
 assert d['zero_refiner_parent_masks_exact'] and all(sha(SOURCE/p)==h for p,h in frozen.items())
 write(ROOT/'completion.json',{'sequence_finished':True,'parent_source_weights_unchanged':True,'elapsed_seconds_resume_including_training':time.monotonic()-start,'mean_image_miou':d['mean_image_miou'],'training_valid_test_labels_excluded':True,'independent_generalization_verified':False,'physical_gpu':GPU,'train_teacher_reused_verified_complete':True,'time':time.time()})
 print('COMPLETE residual',100*d['mean_image_miou'],flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:write(ROOT/'failure.json',{'traceback':traceback.format_exc(),'time':time.time()});raise
