"""Locked GPU3 stage1 full raw valid20/test10 inference; no training."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import fcntl,hashlib,json,subprocess,time,shutil,signal,traceback
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'Model349_exh14'
DATA=Path('/workspace/komap_model1')
PY='/workspace/venv-komap/bin/python'
GPU='3'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def idle():
 deadline=time.monotonic()+3*3600
 while time.monotonic()<deadline:
  clear=True
  for _ in range(3):
   mem,util=map(int,subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
   apps=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
   if mem>64 or util or apps:clear=False;break
   time.sleep(2)
  if clear:return
  write(ROOT/'queue_status.json',{'state':'waiting_gpu3_idle','memory_mib':mem,'util':util,'apps':apps,'time':time.time()});time.sleep(30)
 raise TimeoutError('GPU3 idle wait exceeded 3h; no automatic restart')
def run(args,name,gpu=False,cwd=ROOT):
 if gpu:idle()
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU if gpu else '',NO_ALBUMENTATIONS_UPDATE='1')
 with (ROOT/f'{name}.log').open('x') as log:
  child=subprocess.Popen([PY,'-B','-u',*map(str,args)],cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write(ROOT/'active_child.json',{'pid':child.pid,'stage':name,'gpu':GPU if gpu else None,'time':time.time()})
  start=time.monotonic()
  while child.poll() is None:
   time.sleep(5)
   if gpu:
    apps=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    if len(apps)>1 and child.poll() is None:
     os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=20);raise RuntimeError('Foreign GPU3 compute appeared; stopped our group only')
  assert child.returncode==0,(name,child.returncode)
  write(ROOT/f'{name}_receipt.json',{'elapsed_seconds':time.monotonic()-start,'exit_code':0,'gpu':GPU if gpu else None})
def main():
 assert not (ROOT/'completion.json').exists() and not (ROOT/'failure.json').exists()
 locks=[]
 for path in sorted(set(ROOT.parent.glob('*/gpu3.lock'))|{ROOT/'gpu3.lock'}):
  f=path.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 assert shutil.disk_usage(ROOT).free>8*1024**3
 bundle=ROOT/'Model349_exh14'
 shutil.copytree(SOURCE,bundle,ignore=shutil.ignore_patterns('weights','results','search','__pycache__','.git'))
 (bundle/'weights').symlink_to(SOURCE/'weights',target_is_directory=True)
 frozen={}
 parent=json.loads((SOURCE/'checkpoint_manifest.json').read_text())
 for row in parent['checkpoints']:
  p=SOURCE/row['path'];assert sha(p)==row['sha256'];frozen[str(p)]=row['sha256']
 assets=ROOT/'assets'
 for row in json.loads((ROOT/'config/auxiliary_assets.json').read_text()):
  rel=Path(row['path']);family=rel.parts[0]
  src=DATA/'outputs'/('model329_fullstats_width4000_uniform_group_oof' if family=='model329' else 'model392_widthnorm')/Path(*rel.parts[1:])
  assert sha(src)==row['sha256'];frozen[str(src)]=row['sha256'];dst=assets/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(src)
 for row in parent['checkpoints']:
  p=SOURCE/row['path'];args=p.parent/'args.json'
  if args.exists():frozen[str(args)]=sha(args)
 for path in SOURCE.rglob('*'):
  if path.is_file() and ('src' in path.parts or path.suffix in ['.py','.yaml']) and '__pycache__' not in path.parts:frozen[str(path)]=sha(path)
 write(ROOT/'parent_frozen.json',frozen)
 assert len(list((DATA/'data/test/images').glob('*')))==10
 for split,count in [('valid',20),('test',10)]:
  out=ROOT/'runs';out.mkdir(exist_ok=True);images=DATA/f'data/{split}/images'
  assert len(list(images.glob('*')))==count
  release=out/f'release_{split}';expert=out/f'expert_{split}';fusion=out/f'stage1_{split}'
  run([bundle/'infer_lite.py','--input',images,'--output',release,'--gpus',GPU],f'release_{split}',True,bundle)
  run([ROOT/'infer_model401.py','--input',images,'--assets',assets,'--output',expert,'--device','cuda'],f'expert_{split}',True)
  run([ROOT/'infer_stage1.py','--release',release,'--expert',expert,'--output',fusion],f'fusion_{split}')
  if split=='valid':
   for name,path in [('release',release),('stage1',fusion)]:
    run([ROOT/'score_masks.py','--labels',path/'predicted_labels','--gt',DATA/'data/valid/masks','--output',path/'metrics.json'],f'score_{name}_valid')
  assert all(sha(Path(p))==h for p,h in frozen.items())
 write(ROOT/'completion.json',{'finished':True,'whole_fresh_pipeline':True,'valid_images':20,'test_images':10,'test_gt_read':False,'training_started':False,'parent_weights_unchanged':True,'valid_percent':100*json.loads((ROOT/'runs/stage1_valid/metrics.json').read_text())['image_miou'],'base_changed':False,'time':time.time()})
if __name__=='__main__':
 try:
  write(ROOT/'controller.json',{'pid':os.getpid(),'time':time.time(),'gpu':GPU})
  main()
 except BaseException:write(ROOT/'failure.json',{'traceback':traceback.format_exc(),'time':time.time()});raise
