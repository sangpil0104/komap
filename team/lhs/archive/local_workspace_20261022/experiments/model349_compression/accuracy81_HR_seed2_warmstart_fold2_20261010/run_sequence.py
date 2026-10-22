import os, sys, json, hashlib, subprocess, time, fcntl, signal, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PY='/workspace/venv-komap/bin/python'
DATA=Path('/workspace/komap_model1/data_model382_seed2_fold2')
CK=Path('/workspace/komap_model1/outputs/model392_widthnorm/seed2/fold2/last.pt')
def write(name,value): (ROOT/name).write_text(json.dumps(value,indent=2)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert not (ROOT/'started.json').exists(), 'Existing run: inspect, do not duplicate'
 splits={s:sorted(DATA.joinpath(s,'images').glob('*_image.png')) for s in ['train','valid']}
 assert all(splits.values())
 groups={s:{p.name.split('_')[0] for p in paths} for s,paths in splits.items()}
 assert not groups['train']&groups['valid'], 'Specimen leakage'
 inputs=[CK]+[p for s in splits for p in splits[s]]+[p.parent.parent/'masks'/p.name.replace('_image.png','_mask.png') for s in splits for p in splits[s]]
 frozen={str(p):sha(p) for p in inputs}
 write('input_lock.json',{'sha256':frozen,'split_counts':{s:len(v) for s,v in splits.items()},'groups':{s:sorted(v) for s,v in groups.items()}})
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
 for arm in ['original']:
  args=['train.py','--data-root',str(DATA),'--output-dir',str(ROOT/arm),'--device','cuda:0','--architecture','UnetPlusPlus','--encoder-name','tu-hrnet_w48','--patch-size','512','--input-scale','2','--normalize-width','--samples-per-image','12','--batch-size','1','--accumulation-steps','8','--num-workers','0','--epochs','4','--lr','3e-5','--encoder-lr-mult','.2','--weight-decay','1e-4','--init-checkpoint',str(CK),'--boundary-head-weight','1','--seed','819349','--rng-mode','per_item','--val-every','0','--no-amp']
  stages.append((arm+'_train',arm,args))
 for arm in ['original']:
  stages.append((arm+'_eval',arm,['evaluate_image_mean.py','--checkpoint',str(ROOT/arm/'last.pt'),'--data-root',str(DATA),'--split','valid','--output-dir',str(ROOT/(arm+'_eval')),'--device','cuda:0','--overlap','170','--batch-size','2','--tta-mode','flip4']))
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
 write('completion.json',{'all_stages_completed':True,'inputs_unchanged':True,'official_valid_test_used':False,'time':time.time()})
if __name__=='__main__':
 try: main()
 except BaseException: write('controller_failure.json',{'traceback':traceback.format_exc()});raise
