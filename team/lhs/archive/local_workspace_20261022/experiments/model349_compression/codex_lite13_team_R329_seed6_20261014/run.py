import json,os,time,sys,hashlib,subprocess,fcntl,traceback,zipfile
from pathlib import Path
D=Path(__file__).resolve().parent;PY='/home/sangpil/miniforge3/envs/komap/bin/python'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(n,v):
 p=D/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def call(cmd,log):
 with (D/log).open('a') as f:
  p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);dump('active_child.json',{'pid':p.pid,'command':cmd});assert p.wait()==0,cmd
def idle():return not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
lock=(D/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (D/'completion.json').exists():sys.exit(0)
os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NO_ALBUMENTATIONS_UPDATE='1')
try:
 while not (D/'input.zip').exists() or (D/'input.zip').stat().st_size!=1469971650:
  dump('status.json',{'phase':'waiting_archive_transfer','pid':os.getpid(),'expected_bytes':1469971650});time.sleep(5)
 time.sleep(2)
 dump('status.json',{'phase':'archive_verification','pid':os.getpid()})
 proof=json.load(open(D/'archive_proof.json'));assert sha(D/'input.zip')==proof['zip_sha256']
 extracted=D/'package'
 if not extracted.exists():
  with zipfile.ZipFile(D/'input.zip') as z:
   for i in z.infolist():assert not i.filename.startswith('/') and '..' not in Path(i.filename).parts
   z.extractall(extracted)
 for r in proof['checkpoints']:assert sha(extracted/'R329_seed_search'/r['path'])==r['sha256']
 call([PY,str(D/'score.py'),'preflight'],'preflight.log')
 jobs=[]
 for seed in range(1557,1563):
  tag=f'R329_seed{seed}';job={'tag':tag,'family':'R329_folds134','seed':seed,'component':'329','weights':str(extracted/f'R329_seed_search/seed{seed}'),'work':str(D/'inference'/tag)};jobs.append(job)
  if (D/'scores'/f'{tag}.json').exists():continue
  while not idle():dump('status.json',{'phase':'waiting_gpu','seed':seed,'pid':os.getpid()});time.sleep(10)
  dump('status.json',{'phase':'valid_inference','seed':seed,'pid':os.getpid()})
  jf=D/'active_probe.json';jf.write_text(json.dumps(job));call([PY,str(D/'probe.py'),str(jf)],'inference.log')
  dump('status.json',{'phase':'scoring','seed':seed,'pid':os.getpid()});call([PY,str(D/'score.py'),str(jf)],'evaluation.log')
 dump('jobs.json',jobs)
 for r in proof['checkpoints']:assert sha(extracted/'R329_seed_search'/r['path'])==r['sha256']
 dump('completion.json',{'normal_exit':True,'new_training_used':False,'weights_sha_unchanged':True,'seeds':6,'baseline':'lite13','release_replaced':False,'test_GT_used':False,'finished':time.time()})
 dump('status.json',{'phase':'complete','pid':os.getpid()})
except Exception:
 dump('failure.json',{'traceback':traceback.format_exc()});raise
