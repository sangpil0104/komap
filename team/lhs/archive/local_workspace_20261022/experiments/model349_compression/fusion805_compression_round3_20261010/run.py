from pathlib import Path
import subprocess,fcntl,os,time,json,traceback
ROOT=Path(__file__).resolve().parent;PY='/workspace/venv-komap/bin/python';locks=[]
def write(name,value):(ROOT/name).write_text(json.dumps(value,indent=2)+'\n')
try:
 for p in sorted(set(ROOT.parent.glob('*/gpu7.lock'))|{ROOT/'gpu7.lock'}):
  h=p.open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
 for _ in range(3):
  value=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True);mem,util=map(int,value.strip().split(','));assert mem<128 and util==0,value;time.sleep(2)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 for shard in [0,1]:
  with (ROOT/f'replay_shard{shard}.log').open('w') as log:
   child=subprocess.Popen([PY,str(ROOT/'replay.py'),'--shard',str(shard)],env=env,stdout=log,stderr=subprocess.STDOUT);write('running.json',{'supervisor_pid':os.getpid(),'child_pid':child.pid,'shard':shard,'gpu':7,'time':time.time()});assert child.wait()==0
 receipts=[json.loads((ROOT/f'replay_shard{s}_completion.json').read_text()) for s in [0,1]];assert receipts[0]['configs']==receipts[1]['configs'] and receipts[0]['names']==receipts[1]['names'];rows=[x for d in receipts for x in d['rows']];assert len(rows)==20 and len({x['name'] for x in rows})==20
 write('replay_completion.json',{'configs':receipts[0]['configs'],'names':receipts[0]['names'],'rows':rows,'gt_read':False,'test_read':False})
 with (ROOT/'score.log').open('w') as log:subprocess.run([PY,str(ROOT/'score.py')],check=True,stdout=log,stderr=subprocess.STDOUT,env=env)
 write('pipeline_completion.json',{'finished':True,'time':time.time()})
except BaseException:write('pipeline_failure.json',{'traceback':traceback.format_exc(),'time':time.time()});raise
finally:
 for h in locks:h.close()
