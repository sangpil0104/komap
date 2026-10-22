import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import hashlib,json,subprocess,sys,time,fcntl,shutil,traceback
import yaml
ROOT=Path('/workspace/model349_compress/accuracy81_20261008');SOURCE=ROOT.parent/'Model349_exh14';PY='/workspace/venv-komap/bin/python'
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def idle():
 for _ in range(3):
  value=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  mem,util=map(int,value.strip().split(','));assert mem<=64 and util==0,value
  apps=subprocess.check_output(['nvidia-smi','-i','7','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip();assert not apps,apps
  time.sleep(2)
def main():
 locks=[]
 for name in ['audit_20261008','base12_followup_20261008','base12_inputcache_20261008','base12_highflip4_20261008','base12_resident_packed_20261008','base12_singleworker_20261008','base12_partial_distill329_20261008','exh14_base_search_20261008',ROOT.name]:
  f=(ROOT.parent/name/'gpu7.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 idle()
 manifest=json.loads((ROOT.parent/'exh14_base_search_20261008/bundles/control14/checkpoint_manifest.json').read_text())
 frozen={r['path']:sha(SOURCE/r['path']) for r in manifest['checkpoints']}
 assert all(frozen[r['path']]==r['sha256'] for r in manifest['checkpoints'])
 frozen.update({str(p.relative_to(SOURCE)):sha(p) for p in SOURCE.rglob('*.py') if '__pycache__' not in str(p)})
 frozen.update({str(p.relative_to(SOURCE)):sha(p) for p in (SOURCE/'configs').glob('*') if p.is_file()})
 assert frozen==json.loads((ROOT/'frozen_before.json').read_text())
 for name,b4,hr in [('b4d4_14',True,False),('combined_tta14',True,True)]:
  idle();bundle=ROOT/'bundles'/name
  shutil.copytree(SOURCE,bundle,ignore=shutil.ignore_patterns('weights','__pycache__','.git','results','search'))
  for row in manifest['checkpoints']:
   p=bundle/row['path'];p.parent.mkdir(parents=True,exist_ok=True);p.symlink_to(SOURCE/row['path'])
   args=(SOURCE/row['path']).parent/'args.json'
   if args.exists():shutil.copy2(args,p.parent/'args.json')
  write(bundle/'checkpoint_manifest.json',manifest)
  p=bundle/'infer_lite.py';code=p.read_text()
  before='predict_probs(members, load_gray(row[\'path\']), tta=False, device=str(device))'
  assert code.count(before)==1
  code=code.replace(before,"predict_probs(members, load_gray(row['path']), tta=config['inference'].get('b4_tta_enabled', False), device=str(device))")
  before="batch_size=h['batch_size'], tta=False, input_scale=h['input_scale']"
  assert code.count(before)==1
  code=code.replace(before,"batch_size=h['batch_size'], tta=h.get('enabled', False), input_scale=h['input_scale']")
  compile(code,str(p),'exec');p.write_text(code)
  p=bundle/'configs/model349_lite.yaml';config=yaml.safe_load(p.read_text());config['inference']['b4_tta_enabled']=b4;config['inference']['hrnet']['enabled']=hr;config['inference']['hrnet']['tta']='flip4';p.write_text(yaml.safe_dump(config,sort_keys=False))
  write(bundle/'accuracy_recipe.json',{'b4_tta_enabled':b4,'b4_views':8 if b4 else 1,'hrnet_tta_enabled':hr,'hrnet_views':4 if hr else 1,'inference_sha256':sha(bundle/'infer_lite.py'),'config_sha256':sha(p),'original_checkpoint_sha256':{r['path']:frozen[r['path']] for r in manifest['checkpoints']}})
  out=ROOT/'runs'/(name+'_valid20');out.parent.mkdir(exist_ok=True)
  with (ROOT/(name+'.log')).open('x') as log:
   print('START',name,flush=True)
   subprocess.run([PY,'-B','-u',str(bundle/'infer_lite.py'),'--input','/workspace/komap_model1/data/valid/images','--output',str(out),'--gpus','7','--keep-probabilities','--gt','/workspace/komap_model1/data/valid/masks'],cwd=bundle,stdout=log,stderr=subprocess.STDOUT,check=True)
  metrics=json.loads((out/'metrics.json').read_text());print('SCORE',name,metrics['image_miou']*100,flush=True)
  assert all(sha(SOURCE/p)==h for p,h in frozen.items()),'original source changed'
  write(ROOT/(name+'_verified_source.json'),{'source_unchanged':True,'image_miou':metrics['image_miou'],'count':metrics['count']})
  if metrics['image_miou']>=.81:break
 write(ROOT/'completion.json',{'sequence_finished':True,'source_unchanged':True,'time':time.time()})
if __name__=='__main__':
 try:main()
 except BaseException:
  write(ROOT/'remaining_failure.json',{'traceback':traceback.format_exc(),'time':time.time()});raise
