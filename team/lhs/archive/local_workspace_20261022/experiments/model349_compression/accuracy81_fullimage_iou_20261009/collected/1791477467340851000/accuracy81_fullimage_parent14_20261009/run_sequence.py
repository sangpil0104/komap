"""Train-only balanced CE residual; reuse verified parent fields on idle GPU3."""
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
 prior=PARENT/'accuracy81_residual_20261008'
 assert json.loads((prior/'completion.json').read_text())['parent_source_weights_unchanged']
 assert not (ROOT/'training').exists() and not (ROOT/'completion.json').exists()
 locks=[]
 for p in sorted(set(PARENT.glob('*/gpu3.lock'))|{ROOT/'gpu3.lock'}):
  f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 frozen=json.loads((prior/'parent_frozen.json').read_text());assert all(sha(SOURCE/p)==h for p,h in frozen.items());write(ROOT/'parent_frozen.json',frozen)
 lock=json.loads((ROOT/'preflight_source_lock.json').read_text());assert all(sha(ROOT/k)==h for k,h in lock.items())
 for pid in ['2186700','2187535']:assert subprocess.run(['ps','-p',pid],capture_output=True).returncode!=0
 source=PARENT/'accuracy81_residual_20261008'/'features/train'
 manifest=json.loads((source/'manifest.json').read_text());assert len(manifest['images'])==70
 assert all(sha(source/(x['name']+'.npz'))==x['features_sha256'] for x in manifest['images'])
 (ROOT/'features').mkdir();(ROOT/'features/train').symlink_to(source,target_is_directory=True)
 with (ROOT/'smoke_fullimage.log').open('x') as f:assert subprocess.run([PY,'-B',str(ROOT/'smoke_fullimage.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='')).returncode==0
 run([ROOT/'train_fullimage.py'],'train_fullimage')
 receipt=json.loads((ROOT/'training/completion.json').read_text());assert receipt['epochs']==8 and receipt['optimizer_updates']==560 and receipt['recipe']['native_full_image'] and not receipt['recipe']['crop']
 path=prior/'features/valid';valid=json.loads((path/'manifest.json').read_text());assert valid['parent_masks_exact'] and len(valid['images'])==20
 assert all(sha(path/(x['name']+'.npz'))==x['features_sha256'] for x in valid['images'])
 (ROOT/'features/valid').symlink_to(path,target_is_directory=True);(ROOT/'runs').mkdir();(ROOT/'runs/teacher_valid20').symlink_to(prior/'runs/teacher_valid20',target_is_directory=True)
 run([ROOT/'apply_residual.py'],'apply_residual')
 p=ROOT/'runs/refined_valid20/completion.json';d=json.loads(p.read_text());d['elapsed_kind']='refiner plus evaluation only; raw parent fields reused; not whole fresh pipeline';d['raw_parent_reused']=True;write(p,d)
 score=d['mean_image_miou'];assert d['zero_refiner_parent_masks_exact']
 assert all(sha(SOURCE/p)==h for p,h in frozen.items())
 write(ROOT/'completion.json',{'finished':True,'source_weights_unchanged':True,'mean_image_miou':score,'raw_parent_reused':True,'whole_fresh_pipeline':False,'training_valid_test_labels_excluded':True,'time':time.time()})
 print('COMPLETE native SoftIoU',100*score,flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:write(ROOT/'failure.json',{'traceback':traceback.format_exc(),'time':time.time()});raise
