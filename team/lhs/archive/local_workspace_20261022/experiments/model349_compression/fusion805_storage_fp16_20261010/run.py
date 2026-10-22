from pathlib import Path
import json,subprocess,os,time,fcntl,hashlib,traceback
ROOT=Path(__file__).resolve().parent;PY='/workspace/venv-komap/bin/python';locks=[]
def write(n,d):(ROOT/n).write_text(json.dumps(d,indent=2)+'\n')
def gpu():
 raw=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True);mem,util=map(int,raw.strip().split(','));return mem,util
try:
 while not (ROOT/'build_completion.json').exists():
  if not Path('/proc/2432143').exists():raise RuntimeError('Asset build stopped without completion receipt; inspect build.log')
  write('status.json',{'phase':'waiting_asset_build','time':time.time()});time.sleep(5)
 idle=0
 while True:
  mem,util=gpu();idle=idle+1 if mem<400 and util==0 else 0;write('status.json',{'phase':'waiting_gpu0_idle','memory_MiB':mem,'utilization':util,'idle_observations':idle,'time':time.time()})
  if idle>=3:
   try:
    for p in sorted(set(ROOT.parent.glob('*/gpu0.lock'))|{ROOT/'gpu0.lock'}):
     h=p.open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
    if gpu()[0]>=400 or gpu()[1]!=0:raise BlockingIOError('GPU changed during lock acquisition')
    break
   except BlockingIOError:
    for h in locks:h.close()
    locks=[];idle=0
  time.sleep(15)
 out=ROOT/'validation';out.mkdir(exist_ok=False);cache=ROOT.parent/'fusion805_compression_round2_20261010/fresh23';(out/'input_sha256.json').write_bytes((cache/'input_sha256.json').read_bytes())
 for shard in [0,1]:(out/f'input{shard}').symlink_to(cache/f'input{shard}',target_is_directory=True)
 files=[ROOT/'infer_expert.py',ROOT/'selected_config.json',ROOT/'selected_assets.json',ROOT/'postprocess.py'];frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};(out/'source_sha256.json').write_text(json.dumps(frozen,indent=2))
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');codes=[]
 for shard in [0,1]:
  command=[PY,str(ROOT/'infer_expert.py'),'--parent-package','/workspace/model349_compress/accuracy81_stage1_fresh_20261009','--assets',str(ROOT/'assets'),'--input',str(out/f'input{shard}'),'--output',str(out/f'expert{shard}'),'--config',str(ROOT/'selected_config.json')]
  with (out/f'shard{shard}.log').open('w') as log:
   child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT);write('status.json',{'phase':'fresh_E_inference','gpu':0,'child_pid':child.pid,'shard':shard,'time':time.time()});code=child.wait();codes.append(code);assert code==0,(shard,code)
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen.items());(out/'process_completion.json').write_text(json.dumps({'exit_codes':codes,'physical_gpu':0}))
 write('status.json',{'phase':'fusion_and_independent_scoring','time':time.time()});subprocess.run([PY,str(ROOT/'postprocess.py')],check=True,env=env)
 write('pipeline_completion.json',{'finished':True,'time':time.time()});write('status.json',{'phase':'complete','time':time.time()})
except BaseException:write('pipeline_failure.json',{'traceback':traceback.format_exc(),'time':time.time()});raise
finally:
 for h in locks:h.close()
