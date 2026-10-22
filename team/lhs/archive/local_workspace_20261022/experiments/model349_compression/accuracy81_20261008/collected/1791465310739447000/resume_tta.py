"""Resume only absent components after confirmed terminal GPU-occupancy failure."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import hashlib,json,subprocess,sys,time,fcntl,importlib.util,traceback
import numpy as np
ROOT=Path('/workspace/model349_compress/accuracy81_20261008');PY='/workspace/venv-komap/bin/python'
def write(p,v):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wait_idle():
 while True:
  readings=[]
  for _ in range(3):
   mem,util=map(int,subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
   apps=subprocess.check_output(['nvidia-smi','-i','7','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
   readings.append({'memory_mib':mem,'util':util,'compute_pids':apps})
   if mem>64 or util or apps:break
   time.sleep(2)
  if len(readings)==3 and all(x['memory_mib']<=64 and not x['util'] and not x['compute_pids'] for x in readings):return
  write(ROOT/'resume_wait.json',{'state':'waiting_foreign_gpu7','time':time.time(),'readings':readings})
  time.sleep(30)
def main():
 old=2141791
 if Path(f'/proc/{old}/cmdline').exists():
  args=Path(f'/proc/{old}/cmdline').read_bytes().replace(b'\x00',b' ').decode();assert str(ROOT) not in args,'old controller live'
 assert (ROOT/'failure_gpu_occupied_initial.json').is_file()
 wait_idle();locks=[]
 for name in ['audit_20261008','base12_followup_20261008','base12_inputcache_20261008','base12_highflip4_20261008','base12_resident_packed_20261008','base12_singleworker_20261008','base12_partial_distill329_20261008','exh14_base_search_20261008',ROOT.name]:
  f=(ROOT.parent/name/'gpu7.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 source=ROOT.parent/'Model349_exh14';frozen=json.loads((ROOT/'frozen_before.json').read_text());assert all(sha(source/p)==h for p,h in frozen.items())
 bundle=ROOT/'bundles/hrflip4_14';work=ROOT/'runs/hrflip4_14_valid20';manifest=json.loads((work/'input_manifest.json').read_text());names=[r['name'] for r in manifest['images']];assert len(set(names))==20
 hashes={}
 for c in ['322','326','329','base','336']:
  d=work/'probabilities'/c;receipt=json.loads((d/'complete.json').read_text());assert receipt['images']==20 and receipt['component']==c
  suffix='.pkl' if c=='base' else '.npy';assert {p.stem for p in d.glob('*'+suffix)}==set(names)
  for p in d.glob('*'+suffix):
   hashes[str(p.relative_to(work))]=sha(p)
   if suffix=='.npy':
    arr=np.load(p);assert arr.shape[0]==4 and np.isfinite(arr).all() and arr.min()>=0 and arr.max()<=1
 write(ROOT/'partial_verified_before_resume.json',{'complete_components':['322','326','329','base','336'],'probability_hashes':hashes,'time':time.time()})
 start=time.monotonic()
 for c in ['347','269']:
  d=work/'probabilities'/c;assert not d.exists(),'unexpected partial pending component; manual audit required'
  wait_idle();env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',NO_ALBUMENTATIONS_UPDATE='1')
  with (work/'logs'/f'{c}_resumed.log').open('x') as log:
   print('RESUME_COMPONENT',c,flush=True)
   subprocess.run([PY,'-B','-u',str(bundle/'infer_lite.py'),'--_worker',c,'--output',str(work)],cwd=bundle,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  assert json.loads((d/'complete.json').read_text())['images']==20
 os.chdir(bundle);sys.path.insert(0,str(bundle/'src'));sys.path.insert(0,str(bundle))
 spec=importlib.util.spec_from_file_location('resumed_infer',bundle/'infer_lite.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 import torch,cv2
 torch.set_num_threads(2);cv2.setNumThreads(2)
 scores=m.assemble(work,Path('/workspace/komap_model1/data/valid/masks'))
 assert all(sha(work/p)==h for p,h in hashes.items()),'completed output changed'
 assert all(sha(source/p)==h for p,h in frozen.items()),'original source changed'
 score=float(np.mean([r['image_miou'] for r in scores]));write(work/'completion.json',{'model':'Model349','images':20,'mean_image_miou':score,'elapsed_seconds':time.monotonic()-start,'elapsed_time_kind':'resume segment only; full run interrupted by foreign GPU occupancy, not comparable timing','resumed':True,'completed_components_preserved':True,'inference_sha256':sha(bundle/'infer_lite.py'),'gpus':['7'],'tent_steps':5,'training_started':False})
 write(ROOT/'hrflip4_14_verified_source.json',{'source_unchanged':True,'image_miou':score,'count':20,'resumed':True});print('SCORE hrflip4_14',score*100,flush=True)
 for f in locks:f.close()
 if score>=.81:write(ROOT/'completion.json',{'sequence_finished':True,'source_unchanged':True,'time':time.time()})
 else:subprocess.run([PY,'-B','-u',str(ROOT/'run_remaining.py')],cwd=ROOT,check=True)
if __name__=='__main__':
 try:main()
 except BaseException:write(ROOT/'resume_failure.json',{'traceback':traceback.format_exc(),'time':time.time()});raise
