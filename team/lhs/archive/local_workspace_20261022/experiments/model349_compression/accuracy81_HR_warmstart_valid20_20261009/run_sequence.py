import os, sys, json, hashlib, subprocess, time, fcntl, signal, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PY='/workspace/venv-komap/bin/python'
DATA=Path('/workspace/komap_model1/data')
def write(name,value): (ROOT/name).write_text(json.dumps(value,indent=2)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert not (ROOT/'started.json').exists(), 'Existing run: inspect, do not duplicate'
 parents=[ROOT.parent/('accuracy81_HR_texture_pair_20261009' if f==1 else f'accuracy81_HR_warmstart_fold{f}_20261009') for f in range(1,5)]
 assert all(json.loads((p/'completion.json').read_text())['all_stages_completed'] for p in parents)
 inputs=[p/'original/last.pt' for p in parents]+sorted((DATA/'valid/images').glob('*_image.png'))+sorted((DATA/'valid/masks').glob('*_mask.png'))
 assert len(inputs)==44
 frozen={str(p):sha(p) for p in inputs}
 write('input_lock.json',{'sha256':frozen})
 locks=[]
 for p in sorted(set(ROOT.parent.glob('*/gpu3.lock'))|{ROOT/'gpu3.lock'}):
  h=p.open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
 uuid=subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 for _ in range(3):
  mem,util=map(int,subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
  assert mem<=64 and util==0 and not subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  time.sleep(2)
 write('source_lock.json',{str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py')})
 write('started.json',{'time':time.time(),'protocol':json.loads((ROOT/'protocol.json').read_text())})
 stages=[]
 for fold,parent in enumerate(parents,1):
  stages.append((f'fold{fold}_eval','original',['evaluate_image_mean.py','--checkpoint',str(parent/'original/last.pt'),'--data-root',str(DATA),'--split','valid','--output-dir',str(ROOT/f'fold{fold}_eval'),'--device','cuda:0','--overlap','170','--batch-size','2','--tta-mode','flip4']))
 for stage,arm,args in stages:
  (ROOT/'gpu_identity.json').unlink(missing_ok=True)
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',KOMAP_HR_INPUT_MODE=arm,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
  with (ROOT/(stage+'.log')).open('x') as log:
   child=subprocess.Popen([PY,'-B','-u',str(ROOT/'bootstrap.py'),*args],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   write('active_child.json',{'pid':child.pid,'stage':stage,'time':time.time()})
   while child.poll() is None:
    time.sleep(5)
    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).strip().splitlines()
    apps=[l.split(',')[1].strip() for l in lines if l.split(',')[0].strip()==uuid]
    own=json.loads((ROOT/'gpu_identity.json').read_text())['nvml_pid'] if (ROOT/'gpu_identity.json').exists() else None
    if len(apps)>1 or (own and any(p!=own for p in apps)):
     os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=20);raise RuntimeError('Foreign GPU process detected; stopped only our child')
   write(stage+'_receipt.json',{'exit_code':child.returncode,'pid':child.pid,'time':time.time()})
   assert child.returncode==0, stage+' failed'
 assert all(sha(Path(p))==v for p,v in frozen.items())
 write('completion.json',{'all_stages_completed':True,'inputs_unchanged':True,'official_valid_test_used':True,'time':time.time()})
if __name__=='__main__':
 try: main()
 except BaseException: write('controller_failure.json',{'traceback':traceback.format_exc()});raise
