import os,fcntl,json,time,subprocess,signal,traceback,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def write(name,r):(ROOT/name).write_text(json.dumps(r,indent=2)+'\n')
def main():
 recover='--recover-evaluation' in sys.argv
 assert not (ROOT/'completion.json').exists() and (recover or not (ROOT/'training').exists())
 locks=[]
 gpu_uuid=subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 for path in sorted(set(ROOT.parent.glob('*/gpu3.lock'))|{ROOT/'gpu3.lock'}):
  handle=path.open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
 for _ in range(3):
  mem,util=map(int,subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
  apps=subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
  assert mem<=64 and util==0 and not apps,'GPU3 not idle; no training launched'
  time.sleep(2)
 write('source_lock.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')})
 with (ROOT/'train.log').open('x') as log:
  child=subprocess.Popen(['/workspace/venv-komap/bin/python','-B','-u',str(ROOT/'train.py'),*(['--recover-evaluation'] if recover else [])],env=dict(os.environ,CUDA_VISIBLE_DEVICES='3'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write('active_child.json',{'pid':child.pid,'gpu':'3','stage':'grouped_meta_oof_train','time':time.time()})
  while child.poll() is None:
   time.sleep(5)
   all_apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).strip().splitlines()
   apps=[line.split(',')[1].strip() for line in all_apps if line.split(',')[0].strip()==gpu_uuid]
   identity=ROOT/'gpu_identity.json'
   own_pid=json.loads(identity.read_text())['nvml_pid'] if identity.exists() else None
   if len(apps)>1 or (own_pid is not None and any(pid.strip()!=own_pid for pid in apps)):
    write('foreign_detection.json',{'child_pid':child.pid,'gpu_uuid':gpu_uuid,'gpu_pids':apps,'all_gpu_processes':all_apps,'time':time.time()});os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=20);raise RuntimeError('Foreign GPU process appeared; stopped our child only')
  write('process_receipt.json',{'pid':child.pid,'exit_code':child.returncode,'time':time.time()})
  assert child.returncode==0,'Training failed; preserve failure output'
if __name__=='__main__':
 try:main()
 except BaseException:write('controller_failure.json',{'traceback':traceback.format_exc()});raise
