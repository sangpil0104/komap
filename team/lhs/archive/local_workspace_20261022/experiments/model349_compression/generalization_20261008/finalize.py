"""Collect the fixed diagnostics once both GPU runs finish; no additional inference."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
import json,subprocess,tarfile,time,hashlib
from pathlib import Path
ROOT=Path('/workspace/model349_compress/generalization_20261008');OLD=ROOT.parent/'audit_20261008';PY='/workspace/venv-komap/bin/python'
while not (ROOT/'test10_complete.json').exists():
 if subprocess.run(['pgrep','-f','[r]un_test10.py'],stdout=subprocess.DEVNULL).returncode!=0:raise RuntimeError('Test queue ended without completion; investigate logs')
 time.sleep(5)
assert (ROOT/'data_exposure_audit.json').exists() and (ROOT/'valid_conditional_stability.json').exists()
for script in ['compare_test_masks.py','generate_report.py']:
 print('FINAL',script,flush=True);subprocess.run([PY,'-B',str(ROOT/script)],cwd=ROOT,check=True)
# Include exact immutable inference settings/code/manifests for the two candidates.
for name in ['cpu2_fp16_12','cpu2_fp16_10']:
 import shutil
 dst=ROOT/'frozen_bundle_metadata'/name;dst.mkdir(parents=True,exist_ok=False);src=OLD/'bundles'/name
 for relative in ['infer_lite.py','checkpoint_manifest.json','configs/model349_lite.yaml','configs/calibration.json']:
  p=dst/relative;p.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(src/relative,p)
(ROOT/'completion.json').write_text(json.dumps({'scope':'Exposure audit, selected-valid conditional stability and unlabelled test10 agreement','diagnostics_completed':True,'independent_generalization_verified':False,'new_training_started':False,'fixed_weights_and_configs_unchanged':True,'utc_unix':time.time()},indent=2)+'\n')
paths=[]
for p in ROOT.rglob('*'):
 if p.is_file() and p.name!='results_archive.tar.gz' and 'probabilities' not in p.parts and '__pycache__' not in p.parts:paths.append(p)
with tarfile.open(ROOT/'results_archive.tar.gz','w:gz') as tar:
 for p in sorted(paths):tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
hashv=hashlib.sha256((ROOT/'results_archive.tar.gz').read_bytes()).hexdigest();print('ARCHIVE',len(paths),'sha256',hashv,flush=True)
