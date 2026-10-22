"""Wait for both fixed benchmark phases, then verify and archive results (CPU only)."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',NO_ALBUMENTATIONS_UPDATE='1')
import json, subprocess, time
from pathlib import Path
ROOT=Path('/workspace/model349_compress/audit_20261008'); PYTHON='/workspace/venv-komap/bin/python'
while not (ROOT/'hramp_complete.json').exists():
 if subprocess.run(['pgrep','-f','[r]un_hramp.py'],stdout=subprocess.DEVNULL).returncode!=0:
  raise RuntimeError('HRAMP coordinator exited without completion; investigate preserved logs')
 time.sleep(10)
assert (ROOT/'batched_complete.json').exists()
for script,extra in [('analyze_runs.py',['--root',str(ROOT),'--gt','/workspace/komap_model1/data/valid/masks']),('summarize_timings.py',[]),('verify_and_package.py',[]),('generate_report.py',[]),('collect_results.py',[])]:
 print('START FINAL CHECK',script,flush=True)
 subprocess.run([PYTHON,'-B',str(ROOT/script),*extra],cwd=ROOT,check=True)
(ROOT/'audit_complete.json').write_text(json.dumps({'status':'completed','unix':time.time(),'gpu_jobs_completed':True,'cpu_verification_completed':True})+'\n')
print('AUDIT COMPLETE',flush=True)
