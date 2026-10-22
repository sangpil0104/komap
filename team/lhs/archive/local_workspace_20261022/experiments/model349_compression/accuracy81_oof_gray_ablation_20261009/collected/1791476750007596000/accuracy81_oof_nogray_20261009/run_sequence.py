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
 prior=PARENT/'accuracy81_oof_refiner_20261009'
 assert json.loads((prior/'completion.json').read_text())['parent_source_weights_unchanged']
 assert not (ROOT/'training').exists() and not (ROOT/'completion.json').exists()
 locks=[]
 for p in sorted(set(PARENT.glob('*/gpu3.lock'))|{ROOT/'gpu3.lock'}):
  f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 frozen=json.loads((prior/'parent_frozen.json').read_text());assert all(sha(SOURCE/p)==h for p,h in frozen.items());write(ROOT/'parent_frozen.json',frozen)
 lock=json.loads((ROOT/'preflight_source_lock.json').read_text());assert all(sha(ROOT/k)==h for k,h in lock.items())
 path=prior/'features/train';manifest=json.loads((path/'manifest.json').read_text());assert len(manifest['images'])==70 and manifest['source_network_components_group_oof'] and manifest['original_fixed_oof_metric_reproduced']
 assert all(sha(path/(x['name']+'.npz'))==x['features_sha256'] for x in manifest['images'])
 (ROOT/'features').mkdir();(ROOT/'features/train').symlink_to(path,target_is_directory=True)
 from gray_input import MODE
 import numpy as np
 from gray_input import prepare_gray
 x=np.linspace(.2,.6,320*340,dtype=np.float32).reshape(320,340);result=prepare_gray(x)
 assert result.dtype==np.float32 and np.isfinite(result).all() and result.min()>=0 and result.max()<=1
 assert np.array_equal(prepare_gray(np.ones((8,8),np.float32)*.2),np.ones((8,8),np.float32)*.5)
 if MODE=='zscore':assert np.allclose(result,prepare_gray(.4*x+.2),rtol=0,atol=1e-6)
 else:assert np.array_equal(result,np.full_like(x,.5))
 write(ROOT/'gray_smoke.json',{'mode':MODE,'synthetic_full_image_normalization_passed':True,'real_gt_used':False,'source_sha256':sha(ROOT/'gray_input.py')})
 run([ROOT/'train_residual.py'],'train_residual')
 receipt=json.loads((ROOT/'training/completion.json').read_text());assert receipt['epochs']==8 and receipt['optimizer_updates']==1120 and receipt['recipe']['gray_input_mode']==MODE
 idle();os.environ['CUDA_VISIBLE_DEVICES']=GPU;os.environ['KOMAP_PHYSICAL_GPU']=GPU
 import matched_eval
 matched_eval.main(acquire_locks=False)
 assert all(sha(SOURCE/p)==h for p,h in frozen.items())
 write(ROOT/'completion.json',{'finished':True,'source_weights_unchanged':True,'gray_mode':MODE,'raw_parent_reused':True,'whole_fresh_pipeline':False,'matched_scores_percent':json.loads((ROOT/'matched_source/completion.json').read_text())['scores_percent'],'time':time.time()})
 print('COMPLETE gray',MODE,flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:write(ROOT/'failure.json',{'traceback':traceback.format_exc(),'time':time.time()});raise
