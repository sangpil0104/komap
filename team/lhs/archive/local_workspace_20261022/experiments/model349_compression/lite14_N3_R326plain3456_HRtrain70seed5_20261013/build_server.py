from pathlib import Path
import json,shutil,hashlib
K=Path.home()/'komap';ROOT=Path(__file__).resolve().parent;OLD=K/'lite15_N3_R326plain3456_adopted_20261013'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def main():
 source=K/'Model349_lite20_b4ws_bundle/Model349_fusion805'
 for d in ['refiner','config']:
  shutil.copytree(source/d,ROOT/'parent_package'/d,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pt','*.pth','*.npy','*.npz'))
 if (ROOT/'build_completion.json').exists():return
 manifest=json.load(open(ROOT/'selected_assets.json'));assert manifest['count']==14
 parent=ROOT/'parent_package';r=parent/'Model349_exh14';r.mkdir(parents=True,exist_ok=True)
 shutil.copy2(OLD/'parent_package/infer_fusion.py',parent/'infer_fusion.py')
 oldr=OLD/'parent_package/Model349_exh14'
 for d in ['src','configs','search','oof_design']:
  shutil.copytree(oldr/d,r/d,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pt','*.pth','*.npz','*.npy'))
 for f in ['infer_lite.py','requirements.lock.txt','environment.yaml']:
  shutil.copy2(oldr/f,r/f)
 receipts={}
 for row in manifest['checkpoints']+manifest['metadata']:
  relative=row['path'];dest=(r if row['namespace']=='R' else ROOT/'assets')/relative
  src=(oldr if row['namespace']=='R' else OLD/'assets')/relative
  if relative=='model392_train70/seed5/last.pt':src=K/'codex_HR392_train70_seed345_lite15_20261013/training/seed5/last.pt'
  assert sha(src)==row['sha256'],str(src);dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():shutil.copy2(src,dest)
  assert sha(dest)==row['sha256'];receipts[str(src)]=row['sha256']
 weights=list(ROOT.rglob('*.pt'));assert len(weights)==14,(len(weights),weights)
 for p,h in receipts.items():assert sha(Path(p))==h
 (ROOT/'build_completion.json').write_text(json.dumps(dict(checkpoint_count=14,checkpoint_bytes=manifest['bytes'],standalone_assets=True,no_symlinked_checkpoints=not any(p.is_symlink() for p in weights),original_files_sha_unchanged=True),indent=2)+'\n')
 print('LITE14_PACKAGE_BUILT',manifest['bytes'],flush=True)
if __name__=='__main__':main()
