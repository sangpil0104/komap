import os, sys, json, time, subprocess, hashlib, traceback, fcntl
from pathlib import Path
D=Path(__file__).resolve().parent
K=Path('/home/sangpil/komap'); PY='/home/sangpil/miniforge3/envs/komap/bin/python'
PKG=K/'Model349_lite20_b4ws_bundle/Model349_fusion805'; ES=K/'eseed'
def dump(n,v):
 p=D/n; t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(v,indent=2)); t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def call(cmd,log,cwd=None,env=None):
 with log.open('a') as f: subprocess.run(cmd,cwd=cwd,env=env or os.environ.copy(),stdout=f,stderr=subprocess.STDOUT,check=True)
def idle():
 rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 return not rows
lock=(D/'run.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (D/'completion.json').exists():sys.exit(0)
os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',NO_ALBUMENTATIONS_UPDATE='1',PYTHONUNBUFFERED='1')
try:
 files=set((K/'data/train').rglob('*.png'))|set((K/'data/valid').rglob('*.png'))
 files |= set((K/'lite13_R322share326_20261014/assets').rglob('*.pt'))
 files |= {K/f'weights/all/assets/model329/fold{f}/last.pt' for f in (1,3)}
 inputs={str(p):sha(p) for p in sorted(files)}
 dump('input_sha.json',inputs)
 dump('status.json',{'phase':'preflight','pid':os.getpid(),'time':time.time()})
 # Baseline scoring snapshot is isolated from other users' source/output paths.
 call([PY,str(D/'evaluate.py'),'preflight'],D/'preflight.log')
 jobs=[]
 for seed in (7001,8001,9001):
  out=D/f'training/hr_seed{seed}'; out.mkdir(parents=True,exist_ok=True)
  cmd=[PY,str(PKG/'model397_reproduction_bundle/model392/train.py'),'--data-root',str(ES/'splits/model392_cv/seed1_fold4'),'--output-dir',str(out),'--device','cuda:0','--architecture','UnetPlusPlus','--encoder-name','tu-hrnet_w48','--encoder-weights','imagenet','--patch-size','512','--input-scale','2.0','--normalize-width','--samples-per-image','12','--batch-size','4','--accumulation-steps','2','--num-workers','4','--epochs','70','--lr','3e-4','--encoder-lr-mult','1.0','--weight-decay','1e-4','--boundary-head-weight','1.0','--seed',str(seed),'--rng-mode','legacy','--val-every','0']
  jobs.append((f'hr_{seed}','hr',out,cmd,None))
 for s in (7,8):
  for fold in (1,3):
   out=D/f'training/ws{s}/fold{fold}';out.mkdir(parents=True,exist_ok=True)
   cmd=[PY,str(PKG/'b4ws/train_bnfreeze.py'),'--data',str(ES/'splits/model329_cv'),'--train_splits',f'train_fold{fold}','--out',str(out),'--arch','unet','--encoder','efficientnet-b4','--norm','image_global','--input','zcdf3','--crop','1536','--scale','1','--target_width','4000','--sample','uniform','--bs','2','--iters','2000','--lr','3e-5','--wd','1e-2','--dice_w','1','--cldice_w','0','--seed',str(819349+1000*s),'--workers','4','--no_valid_eval','--deterministic_workers','--init_checkpoint',str(K/f'weights/all/assets/model329/fold{fold}/last.pt')]
   jobs.append((f'ws{s}_f{fold}','b4',out,cmd,str(PKG/'b4ws')))
 dump('jobs.json',[{'tag':t,'kind':kind,'output':str(out),'command':cmd} for t,kind,out,cmd,cwd in jobs])
 for index,(tag,kind,out,cmd,cwd) in enumerate(jobs):
  if not (out/'training_completion.json').exists():
   if (out/'last.pt').exists():
    if kind=='hr' and (out/'train_state.pt').exists() and (out/'resume_authorized.json').exists():
     if '--resume' not in cmd:cmd=cmd+['--resume']
    else:raise RuntimeError(f'Partial/unreceipted checkpoint {out}; no duplicate training permitted')
   while not idle():
    dump('status.json',{'phase':'waiting_gpu','job':tag,'index':index,'pid':os.getpid()});time.sleep(10)
   dump('status.json',{'phase':'training','job':tag,'index':index,'jobs':len(jobs),'pid':os.getpid(),'started':time.time()})
   env=os.environ.copy()
   if kind=='b4':env['PYTHONPATH']=str(PKG/'Model349_exh14/oof_design/runtime')
   call(cmd,out/'train.log',cwd,env)
   dump(str(out.relative_to(D)/'training_completion.json'),{'checkpoint_sha':sha(out/'last.pt'),'finished':time.time()})
  while not idle():time.sleep(10)
  dump('status.json',{'phase':'valid_inference','job':tag,'index':index,'pid':os.getpid()})
  jobfile=D/'active_probe.json';jobfile.write_text(json.dumps([[tag,kind,str(out if kind=='b4' else out/'last.pt')]]))
  env=os.environ.copy();env['PROBE_JOBS']=str(jobfile)
  call([PY,str(D/'probe_e.py'),str(PKG),str(K/'weights/all/assets'),str(K/'data/valid/images'),str(D/'probabilities')],D/'inference.log',env=env)
 dump('status.json',{'phase':'scoring','pid':os.getpid()})
 call([PY,str(D/'evaluate.py'),'all'],D/'evaluation.log')
 changed=[p for p,h in inputs.items() if sha(Path(p))!=h]
 assert not changed,changed
 dump('completion.json',{'normal_exit':True,'input_sha_unchanged':True,'new_training_jobs':len(jobs),'baseline':'lite13','test_gt_used':False,'release_replaced':False,'finished':time.time()})
 dump('status.json',{'phase':'complete','pid':os.getpid()})
except Exception:
 dump('failure.json',{'traceback':traceback.format_exc(),'time':time.time()});raise
