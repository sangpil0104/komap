from pathlib import Path
import subprocess,os,json,fcntl,time,hashlib
r=Path('/workspace/model349_compress/fusion805_compression_round2_20261010');fresh=r/'fresh23_gpu0';py='/workspace/venv-komap/bin/python';locks=[]
try:
 for p in sorted(set(Path('/workspace/model349_compress').glob('*/gpu0.lock'))|{r/'gpu0.lock'}):
  h=p.open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
 for _ in range(3):
  value=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True);mem,util=map(int,value.strip().split(','));assert mem<=400 and util==0,value;time.sleep(2)
 files=[r/'infer_expert.py',r/'selected_config.json',r/'selected_assets.json',r/'fresh_postprocess_gpu0.py',Path(__file__)]
 frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};(fresh/'source_sha256.json').write_text(json.dumps(frozen,indent=2));codes=[]
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
 for shard in [0,1]:
  command=[py,str(r/'infer_expert.py'),'--parent-package','/workspace/model349_compress/accuracy81_stage1_fresh_20261009','--assets','/workspace/model349_compress/accuracy81_stage1_fresh_20261009/assets','--input',str(fresh/f'input{shard}'),'--output',str(fresh/f'expert{shard}'),'--config',str(r/'selected_config.json')]
  with (fresh/f'shard{shard}.log').open('w') as log:
   child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
   (fresh/'running.json').write_text(json.dumps({'supervisor_pid':os.getpid(),'child_pid':child.pid,'shard':shard,'physical_gpu':0,'started_unix':time.time()}));code=child.wait();codes.append(code);assert code==0,(shard,code)
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen.items())
 (fresh/'process_completion.json').write_text(json.dumps({'exit_codes':codes,'physical_gpu':0}))
 subprocess.run([py,str(r/'fresh_postprocess_gpu0.py')],check=True,env=env)
 (fresh/'pipeline_completion.json').write_text(json.dumps({'finished':True,'physical_gpu':0,'time_unix':time.time()}))
except Exception as exc:
 (fresh/'failure.json').write_text(json.dumps({'error':repr(exc),'time_unix':time.time()}));raise
