"""Boundary-retained fine tuning against an already-completed paired control."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import json,hashlib,subprocess,time,fcntl,shutil,traceback
import yaml
ROOT=Path('/workspace/model349_compress/accuracy81_hrboundary_20261008');TTA=ROOT.parent/'accuracy81_20261008';SOURCE=ROOT.parent/'Model349_exh14';PY='/workspace/venv-komap/bin/python'
def write(p,v):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def idle():
 while True:
  clear=True
  for _ in range(3):
   value=subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
   mem,util=map(int,value.strip().split(','));apps=subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
   if mem>64 or util or apps:clear=False;break
   time.sleep(2)
  if clear:return
  write(ROOT/'queue_status.json',{'state':'waiting_for_gpu3','memory_mib':mem,'util':util,'compute_pids':apps,'time':time.time()});time.sleep(30)
def main():
 previous=ROOT.parent/'accuracy81_hrlovasz_gpu3_20261008'
 done=json.loads((previous/'completion.json').read_text());assert done['paired_sequence_complete'] and done['source_unchanged']
 p=subprocess.run(['ps','-p','2157571','-o','args='],capture_output=True,text=True);assert p.returncode!=0,'previous controller still live; do not duplicate'
 assert shutil.disk_usage(ROOT).free>20*1024**3,'storage budget insufficient'
 locks=[]
 for name in ['accuracy81_hrlovasz_gpu3_20261008', ROOT.name]:
  f=(ROOT.parent/name/'gpu3.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 frozen=json.loads((TTA/'frozen_before.json').read_text());assert all(sha(SOURCE/p)==h for p,h in frozen.items());write(ROOT/'frozen_before.json',frozen)
 manifest=json.loads((ROOT.parent/'exh14_base_search_20261008/bundles/control14/checkpoint_manifest.json').read_text());hr='weights/model336_hrnet_best/last.pt'
 for arm in ['boundary']:
  idle();env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',NO_ALBUMENTATIONS_UPDATE='1')
  with (ROOT/(arm+'_train.log')).open('x') as log:
   print('TRAIN',arm,flush=True);subprocess.run([PY,'-B','-u',str(ROOT/'train.py'),'--arm',arm],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  receipt=json.loads((ROOT/'training'/arm/'completion.json').read_text());final=Path(receipt['final_checkpoint']);assert sha(final)==receipt['final_sha256']
  bundle=ROOT/'bundles'/arm;shutil.copytree(SOURCE,bundle,ignore=shutil.ignore_patterns('weights','__pycache__','.git','results','search'))
  rows=[]
  for row in manifest['checkpoints']:
   row=dict(row);p=bundle/row['path'];p.parent.mkdir(parents=True,exist_ok=True);target=final if row['path']==hr else SOURCE/row['path'];p.symlink_to(target)
   if row['path']==hr:row['sha256']=sha(final);row['bytes']=final.stat().st_size
   args=(SOURCE/row['path']).parent/'args.json'
   if args.exists():shutil.copy2(args,p.parent/'args.json')
   rows.append(row)
  updated=dict(manifest,checkpoints=rows);write(bundle/'checkpoint_manifest.json',updated)
  config=bundle/'configs/model349_lite.yaml';cfg=yaml.safe_load(config.read_text());cfg['validation_image_miou']=None;config.write_text(yaml.safe_dump(cfg,sort_keys=False))
  idle();output=ROOT/'runs'/(arm+'_valid20');output.parent.mkdir(exist_ok=True)
  with (ROOT/(arm+'_valid.log')).open('x') as log:
   print('VALID',arm,flush=True);subprocess.run([PY,'-B','-u',str(bundle/'infer_lite.py'),'--input','/workspace/komap_model1/data/valid/images','--output',str(output),'--gpus','3','--keep-probabilities','--gt','/workspace/komap_model1/data/valid/masks'],cwd=bundle,stdout=log,stderr=subprocess.STDOUT,check=True)
  score=json.loads((output/'metrics.json').read_text())['image_miou'];print('RAW_SCORE',arm,score*100,flush=True)
  assert all(sha(SOURCE/p)==h for p,h in frozen.items()),'original source changed'
 write(ROOT/'completion.json',{'boundary_sequence_complete':True,'source_unchanged':True,'time':time.time(),'independent_generalization_verified':False})
if __name__=='__main__':
 try:main()
 except BaseException:write(ROOT/'failure.json',{'traceback':traceback.format_exc(),'time':time.time()});raise
