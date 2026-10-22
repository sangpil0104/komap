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
 assert not (ROOT/'completion.json').exists() and not (ROOT/'gpu_completion.json').exists()
 for pid in ['2188415','2188645']:assert subprocess.run(['ps','-p',pid],capture_output=True).returncode!=0
 locks=[]
 for p in sorted(set(PARENT.glob('*/gpu3.lock'))|{ROOT/'gpu3.lock'}):
  f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 frozen=json.loads((PARENT/'accuracy81_residual_20261008/parent_frozen.json').read_text());assert all(sha(SOURCE/p)==h for p,h in frozen.items());write(ROOT/'parent_frozen.json',frozen)
 lock=json.loads((ROOT/'preflight_source_lock.json').read_text());assert all(sha(ROOT/k)==h for k,h in lock.items())
 prior=PARENT/'accuracy81_residual_20261008/runs/teacher_valid20'
 cached={str(p):sha(p) for p in (prior/'probabilities').rglob('*') if p.is_file()};cached[str(prior/'input_manifest.json')]=sha(prior/'input_manifest.json');write(ROOT/'cached_source_before.json',cached)
 run([ROOT/'predict_moments.py'],'predict_moments')
 assert json.loads((ROOT/'gpu_completion.json').read_text())['baseline_probability_maps_exact']
 with (ROOT/'replay_moments.log').open('x') as f:
  child=subprocess.Popen([PY,'-B','-u',str(ROOT/'replay_moments.py')],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=f,stderr=subprocess.STDOUT)
  write(ROOT/'active_child.json',{'pid':child.pid,'stage':'cpu_replay','time':time.time()});assert child.wait()==0,'CPU replay failed'
 assert all(sha(Path(p))==h for p,h in cached.items()) and all(sha(SOURCE/p)==h for p,h in frozen.items())
 write(ROOT/'completion.json',{'finished':True,'parent_source_weights_unchanged':True,'cached_sources_unchanged':True,'baseline80_iou_exact':True,'checkpoint_count':14,'checkpoint_bytes':sum((SOURCE/p).stat().st_size for p in frozen if p.startswith('weights/')),'raw_unchanged_components_reused':True,'whole_fresh_pipeline':False,'candidate_scores_percent':{k:100*json.loads((ROOT/'runs'/k/'completion.json').read_text())['mean_image_miou'] for k in ['baseline','alpha025','alpha1']},'time':time.time()})
 print('COMPLETE HR336 moments',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:write(ROOT/'failure.json',{'traceback':traceback.format_exc(),'time':time.time()});raise
