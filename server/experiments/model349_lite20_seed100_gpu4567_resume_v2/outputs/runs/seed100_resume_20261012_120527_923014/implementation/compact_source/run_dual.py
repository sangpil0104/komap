"""Wait for existing raw workers; verify receipts; run two replay shards then score."""
from pathlib import Path
import json,os,time,subprocess,fcntl,traceback
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'dual_seed';PY='/workspace/venv-komap/bin/python'
children=[];locks=[];handles=[]
def write(name,value):(OUT/name).write_text(json.dumps(value,indent=2)+'\n')
def main():
 assert not (OUT/'pipeline_completion.json').exists() and not (OUT/'pipeline_failure.json').exists()
 while not all((ROOT/f'seed1_maps/shard{i}/completion.json').exists() for i in [0,1]):
  if any(ROOT.glob('seed1_failure_*.txt')):raise RuntimeError('Existing seed1 worker failed; see preserved receipt')
  write('pipeline_status.json',{'phase':'waiting_existing_seed1_inference','updated_at':time.time()});time.sleep(5)
 receipts=[json.loads((ROOT/f'seed1_maps/shard{i}/completion.json').read_text()) for i in [0,1]]
 rows=[r for receipt in receipts for r in receipt['rows']]
 assert len(rows)==20 and len({r['name'] for r in rows})==20
 assert all(r['family_changed_values']==0 and r['max_abs_difference']==0 for r in rows)
 for gpu in [6,7]:
  handle=(ROOT/f'gpu{gpu}.lock').open('a');deadline=time.time()+120
  while True:
   try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);break
   except BlockingIOError:
    if time.time()>deadline:raise RuntimeError(f'GPU{gpu} lock busy after raw completion')
    time.sleep(2)
  locks.append(handle)
 for _ in range(3):
  raw=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  parsed=[[int(x.strip()) for x in line.split(',')] for line in raw.strip().splitlines()]
  assert all(memory<128 and util==0 for _,memory,util in parsed),raw
  time.sleep(1)
 for shard,gpu in [(0,6),(1,7)]:
  handle=(OUT/f'replay_shard{shard}.log').open('w');handles.append(handle);env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
  child=subprocess.Popen([PY,str(OUT/'replay.py'),'--shard',str(shard)],env=env,stdout=handle,stderr=subprocess.STDOUT);children.append(child)
 write('pipeline_status.json',{'phase':'dual_seed_replay','pids':[c.pid for c in children],'updated_at':time.time()})
 while any(c.poll() is None for c in children):
  if any(c.poll() not in [None,0] for c in children):raise RuntimeError('Replay shard failed')
  time.sleep(3)
 assert all(c.returncode==0 for c in children)
 receipts=[json.loads((OUT/f'replay_shard{i}_completion.json').read_text()) for i in [0,1]]
 assert receipts[0]['names']==receipts[1]['names'] and receipts[0]['configs']==receipts[1]['configs']
 rows=[r for receipt in receipts for r in receipt['rows']];assert len(rows)==20 and len({r['name'] for r in rows})==20
 frozen={}
 for receipt in receipts:frozen.update(receipt['input_sha256'])
 merged={'names':receipts[0]['names'],'configs':receipts[0]['configs'],'rows':rows,'input_sha256':frozen,'gt_read':False,'test_read':False,'reference25_exact':True,'reference27_exact':True,'hr_family_parity_exact':True,'gpu_replay':True,'whole_model_raw_inference':False}
 write('replay_completion.json',merged)
 with (OUT/'score.log').open('w') as handle:subprocess.run([PY,str(OUT/'score.py')],stdout=handle,stderr=subprocess.STDOUT,check=True)
 write('pipeline_completion.json',{'phase':'complete','seed1_raw_images':20,'replay_images':20,'parent_raw_gt_read':False,'test_read':False,'finished_at':time.time()})
 write('pipeline_status.json',{'phase':'complete','updated_at':time.time()})
if __name__=='__main__':
 try:main()
 except BaseException:
  for child in children:
   if child.poll() is None:child.terminate()
  for child in children:
   try:child.wait(timeout=30)
   except subprocess.TimeoutExpired:child.kill();child.wait()
  write('pipeline_failure.json',{'traceback':traceback.format_exc(),'time':time.time()});raise
 finally:
  for h in handles+locks:h.close()
