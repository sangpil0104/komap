"""Frozen 12/10 test10 inference; unlabelled output audit only."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NO_ALBUMENTATIONS_UPDATE='1')
import fcntl,hashlib,json,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/workspace/model349_compress/generalization_20261008'); OLD=ROOT.parent/'audit_20261008'; PY='/workspace/venv-komap/bin/python'
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def cmd(args):return subprocess.check_output(args,text=True)
def snapshot():
 return {'utc':datetime.now(timezone.utc).isoformat(),'gpu':cmd(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv']),'compute_apps':cmd(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv']),'processes':cmd(['ps','-eo','pid,ppid,args']),'storage':cmd(['df','-h','/workspace'])}
def idle():
 assert subprocess.run(['pgrep','-f','[i]nfer_lite.py|[i]nfer_audited.py'],stdout=subprocess.DEVNULL).returncode!=0,'Existing inference detected'
 for _ in range(2):
  raw=cmd(['nvidia-smi','-i','7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'])
  mem,util=map(int,raw.strip().split(','));assert mem<=64 and util==0,raw
  time.sleep(2)
ROOT.mkdir(exist_ok=True);(ROOT/'runs').mkdir(exist_ok=True)
lock=(OLD/'gpu7.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
write(ROOT/'server_before.json',snapshot())
before={};names=['cpu2_fp16_12','cpu2_fp16_10']
for name in names:
 b=OLD/'bundles'/name;manifest=json.loads((b/'checkpoint_manifest.json').read_text())
 rows=[]
 for row in manifest['checkpoints']:
  p=b/row['path'];actual=sha(p);assert actual==row['sha256'];rows.append({'path':row['path'],'sha256':actual,'bytes':p.stat().st_size})
 before[name]={'infer_sha256':sha(b/'infer_lite.py'),'config_sha256':sha(b/'configs/model349_lite.yaml'),'calibration_sha256':sha(b/'configs/calibration.json'),'weights':rows}
write(ROOT/'frozen_candidates_before.json',before)
for name in names:
 idle();b=OLD/'bundles'/name;out=ROOT/'runs'/f'{name}_test10';assert not out.exists()
 args=[PY,'-B','-u',str(b/'infer_lite.py'),'--input','/workspace/komap_model1/data/test/images','--output',str(out),'--gpus','7','--keep-probabilities']
 started=time.time()
 with (ROOT/f'{name}_test10.log').open('w') as f:subprocess.run(args,cwd=b,stdout=f,stderr=subprocess.STDOUT,check=True)
 write(out/'run_audit.json',{'started_unix':started,'finished_unix':time.time(),'command':args,'bundle':str(b),'ground_truth_available':False})
 print('FINISHED',name,json.loads((out/'completion.json').read_text()),flush=True)
for name in names:
 b=OLD/'bundles'/name;v=before[name]
 assert v['infer_sha256']==sha(b/'infer_lite.py') and v['config_sha256']==sha(b/'configs/model349_lite.yaml') and v['calibration_sha256']==sha(b/'configs/calibration.json')
 for row in v['weights']:assert row['sha256']==sha(b/row['path'])
idle();write(ROOT/'server_after.json',snapshot());write(ROOT/'test10_complete.json',{'status':'completed','candidates':names,'weights_and_configs_unchanged':True,'training_started':False,'test_score_claimed':False})
