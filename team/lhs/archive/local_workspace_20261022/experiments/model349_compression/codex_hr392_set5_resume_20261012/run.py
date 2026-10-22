import os,sys,json,time,subprocess,shutil,hashlib,traceback
from pathlib import Path
K=Path.home()/'komap';O=K/'codex_hr392_set5_resume_20261012';E=K/'codex_r326_ema_seed4_20261012';PY=sys.executable
P=K/'Model349_lite20_b4ws_bundle/Model349_fusion805';A=K/'lite16_b413_adopted_20261012'
def dump(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def run(cmd,log,env):
 with (O/log).open('a') as f:
  p=subprocess.Popen(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,env=env)
  dump('active_child.json',{'pid':p.pid,'stage':log});assert p.wait()==0,log

def main():
 if (O/'completion.json').exists():return
 while not ((E/'completion.json').exists() and (E/'lite16_evaluation_completion.json').exists()):
  if (E/'failure.json').exists() or (E/'lite16_evaluation_failure.json').exists():raise RuntimeError('326 prerequisite failed; no automatic start')
  dump('queue_status.json',{'state':'waiting R326 EMA and lite16 evaluation','pid':os.getpid(),'time':time.time()});time.sleep(20)
 stable=0
 while stable<3:
  ids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines()
  stable=0 if any(x.strip() for x in ids) else stable+1
  if stable<3:time.sleep(20)
 import torch
 src=K/'eseed/hr5/seed2/fold4';out=O/'training';out.mkdir(exist_ok=True)
 parent={n:sha(src/n) for n in ['train_state.pt','last.pt','args.json']}
 if not (O/'training_completion.json').exists():
  assert not (out/'train_state.pt').exists(),'Partial retry requires explicit inspection'
  state=torch.load(src/'train_state.pt',map_location='cpu',weights_only=False);assert state['epoch']==31
  args=json.loads((src/'args.json').read_text());assert args['seed']==5002 and args['epochs']==70
  shutil.copy2(src/'train_state.pt',out/'train_state.pt')
  dump('parent_receipt.json',{'sha256':parent,'epoch':31,'args':args})
  args['output_dir']=str(out)
  cmd=[PY,'-u',P/'model397_reproduction_bundle/model392/train.py']
  for k,v in args.items():
   if k=='target_width' or v is None:continue
   flag='--'+k.replace('_','-')
   if isinstance(v,bool):
    if v:cmd.append(flag)
   else:cmd.extend([flag,str(v)])
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4')
  dump('queue_status.json',{'state':'training epoch32-70','pid':os.getpid()})
  run(cmd,'train.log',env)
  final=torch.load(out/'train_state.pt',map_location='cpu',weights_only=False);assert final['epoch']==70
  dump('training_completion.json',{'epoch':70,'sha256':sha(out/'last.pt')})
 # Keep original assets intact; overlay the completed set5 four HR checkpoints.
 assets=O/'assets';assets.mkdir(exist_ok=True)
 for name in ['model329_ws','refiner']:
  dest=assets/name
  if not dest.exists():dest.symlink_to(K/'weights/all/assets'/name,target_is_directory=True)
 for s,f in [(1,2),(1,4),(2,1),(2,4)]:
  dest=assets/f'model392/seed{s}/fold{f}';dest.mkdir(parents=True,exist_ok=True)
  source=out if (s,f)==(2,4) else K/f'eseed/hr5/seed{s}/fold{f}'
  ck=dest/'last.pt'
  if not ck.exists():ck.symlink_to(source/'last.pt')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4')
 dump('queue_status.json',{'state':'lite16 set5 fresh E/F valid20','pid':os.getpid()})
 run([PY,A/'infer_compact.py','--config',A/'selected_config_lite16_b413.json','--parent-package',P,'--assets',assets,'--input',K/'data/valid/images','--output',O/'valid','--gpus','0','--release',K/'l18h/e2e_valid/release','--allow-retrained'],'eval.log',env)
 import numpy as np
 from PIL import Image
 manifest=json.loads((K/'l18h/e2e_valid/release/input_manifest.json').read_text());pal=np.array(manifest['config']['palette'],dtype=np.uint8);rows={}
 for r in manifest['images']:
  n=r['name'];gt_rgb=np.array(Image.open(K/f'data/valid/masks/{n}_mask.png').convert('RGB'));gt=np.full(gt_rgb.shape[:2],255,np.uint8)
  for c,col in enumerate(pal):gt[np.all(gt_rgb==col,axis=-1)]=c
  assert (gt<4).all()
  pred=np.array(Image.open(O/f'valid/fusion/predicted_labels/{n}.png'))
  assert (pred<4).all()
  ious=[float(((pred==c)&(gt==c)).sum()/u) if (u:=((pred==c)|(gt==c)).sum()) else 0. for c in range(4)]
  rows[n]={'class_iou':ious,'miou':float(np.mean(ious))}
 for n,h in parent.items():assert sha(src/n)==h
 score=float(np.mean([v['miou'] for v in rows.values()])*100)
 dump('result.json',{'lite16_set5_percent':score,'baseline_percent':80.55464328764197,'delta_pp':score-80.55464328764197,'per_image':rows,'all_hr_models_epochs':70,'original_parent_unchanged':True,'valid_development_score':True,'release_replaced':False})
 dump('completion.json',{'time':time.time(),'result':'result.json'});dump('queue_status.json',{'state':'complete'})
if __name__=='__main__':
 try:main()
 except Exception:dump('failure.json',{'error':traceback.format_exc(),'time':time.time()});raise
