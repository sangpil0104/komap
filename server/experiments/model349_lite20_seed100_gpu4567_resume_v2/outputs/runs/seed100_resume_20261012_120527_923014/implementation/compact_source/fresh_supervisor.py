from pathlib import Path
import subprocess,os,json,fcntl,time,hashlib
r=Path('/workspace/model349_compress/fusion805_compression_round2_20261010');fresh=r/'fresh23';locks=[];children=[]
for gpu in [6,7]:
 h=(r/f'gpu{gpu}.lock').open('w');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
for _ in range(3):
 for gpu in [6,7]:
  val=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True);mem,util=map(int,val.strip().split(','));assert mem<100 and util==0,val
 time.sleep(2)
source_files=[r/'infer_expert.py',r/'selected_config.json',r/'selected_assets.json',r/'fresh_postprocess.py']
frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
(fresh/'source_sha256.json').write_text(json.dumps(frozen,indent=2))
for shard,gpu in enumerate([6,7]):
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
 log=(fresh/f'shard{shard}.log').open('w')
 command=['/workspace/venv-komap/bin/python',str(r/'infer_expert.py'),'--parent-package','/workspace/model349_compress/accuracy81_stage1_fresh_20261009','--assets','/workspace/model349_compress/accuracy81_stage1_fresh_20261009/assets','--input',str(fresh/f'input{shard}'),'--output',str(fresh/f'expert{shard}'),'--config',str(r/'selected_config.json')]
 child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT);children.append(child)
(fresh/'running.json').write_text(json.dumps({'pids':[c.pid for c in children]}))
codes=[c.wait() for c in children];assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen.items());(fresh/'process_completion.json').write_text(json.dumps({'exit_codes':codes}));assert codes==[0,0],codes
