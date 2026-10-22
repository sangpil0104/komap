"""Fixed-budget, train70-only grouped meta-OOF Eutectic context correction."""
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8')
from pathlib import Path
import json, hashlib, random, time, subprocess, traceback, sys
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent
FIELDS=ROOT.parent/'accuracy81_oof_refiner_20261009/features/train'
STEPS=400
RECOVER="--recover-evaluation" in sys.argv
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,r): p.write_text(json.dumps(r,indent=2,default=lambda x:x.item())+'\n')
def labels(x):
 priority=np.array([2,0,1,3],np.uint8)
 return priority[x[priority].argmax(0)]
def score(p,y):
 out=[]
 for c in range(4):
  a=p==c;b=y==c;u=np.count_nonzero(a|b)
  out.append(float(np.count_nonzero(a&b)/u) if u else 0.)
 return out
class Context(nn.Module):
 def __init__(self):
  super().__init__();layers=[];channels=5
  for dilation in [1,2,4,8]:
   layers.extend([nn.Conv2d(channels,16,3,padding=dilation,dilation=dilation),nn.SiLU()]);channels=16
  layers.append(nn.Conv2d(16,1,1));self.net=nn.Sequential(*layers)
  nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward(self,logits,gray):
  centered=logits-logits.mean(1,keepdim=True)
  delta=self.net(torch.cat([centered,gray],1)).tanh()
  return torch.cat([logits[:,:3],logits[:,3:4]+delta],1),delta
def foreign_check():
 apps=subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip().splitlines()
 assert len(apps)<=1,'Foreign GPU compute appeared; abort our process'
def main():
 assert not (ROOT/'completion.json').exists() and (RECOVER or not (ROOT/'training').exists())
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
 apps=subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip();assert not apps,'GPU acquired before our context initialization'
 context_hold=torch.zeros(1,device='cuda');torch.cuda.synchronize()
 own_apps=subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip().splitlines();assert len(own_apps)==1,'GPU context exclusivity failed'
 write(ROOT/'gpu_identity.json',{'namespace_pid':os.getpid(),'nvml_pid':own_apps[0].strip(),'gpu':'3','registration_method':'No apps before CUDA initialization; exactly one app after'})
 manifest=json.loads((FIELDS/'manifest.json').read_text())
 assert manifest['split']=='train' and len(manifest['images'])==70 and manifest['source_network_components_group_oof']
 frozen={};data=[];groups={}
 for row in manifest['images']:
  p=FIELDS/(row['name']+'.npz');assert sha(p)==row['features_sha256'];frozen[str(p)]=row['features_sha256']
  with np.load(p) as z:l=z['logits'];g=z['gray'];y=z['target']
  assert l.shape==(4,*y.shape) and g.shape==y.shape and np.isfinite(l).all() and np.isin(y,range(4)).all()
  mean=g.mean(dtype=np.float64);std=g.std(dtype=np.float64);g=np.clip((g-mean)/max(std,1e-6),-3,3).astype(np.float32)/3
  group=row['name'].split('_')[0];assert groups.get(group,row['fold'])==row['fold'];groups[group]=row['fold']
  data.append((row,l,g,y,np.flatnonzero(y==3)))
 out=ROOT/'training';out.mkdir(exist_ok=RECOVER);baseline=[];candidate=[];records=[];start=time.monotonic()
 for fold in range(1,5):
  seed=81900+fold;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  rng=np.random.default_rng(seed);training=[x for x in data if x[0]['fold']!=fold];held=[x for x in data if x[0]['fold']==fold]
  model=Context().cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
  if RECOVER:model.load_state_dict(torch.load(out/f'fold{fold}.pt',map_location='cpu',weights_only=False)['state_dict'])
  for step in range(0 if RECOVER else STEPS):
   batch=[]
   for _ in range(4):
    row,l,g,y,coords=training[int(rng.integers(len(training)))];h,w=y.shape;p=192
    if len(coords) and rng.random()<.5:
     cy,cx=divmod(int(coords[rng.integers(len(coords))]),w);top=int(np.clip(cy-rng.integers(p),0,h-p));left=int(np.clip(cx-rng.integers(p),0,w-p))
    else:top=int(rng.integers(h-p+1));left=int(rng.integers(w-p+1))
    a=l[:,top:top+p,left:left+p];b=g[None,top:top+p,left:left+p];c=y[top:top+p,left:left+p]
    k=int(rng.integers(4));a=np.rot90(a,k,axes=(-2,-1));b=np.rot90(b,k,axes=(-2,-1));c=np.rot90(c,k)
    batch.append((a,b,c))
   a=torch.tensor(np.stack([v[0] for v in batch]),device='cuda');b=torch.tensor(np.stack([v[1] for v in batch]),device='cuda');c=torch.tensor(np.stack([v[2] for v in batch]),device='cuda',dtype=torch.long)
   pred,delta=model(a,b);binary=pred[:,3]-torch.logsumexp(pred[:,:3],dim=1)
   ce=-nn.functional.log_softmax(pred,dim=1).gather(1,c[:,None]).mean()
   loss=.5*ce+.5*nn.functional.binary_cross_entropy_with_logits(binary,(c==3).float())+.05*delta.square().mean()
   assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
   nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
   if step%50==0:foreign_check();print(json.dumps({'fold':fold,'step':step,'loss':loss.item(),'seconds':time.monotonic()-start}),flush=True)
  if not RECOVER:torch.save({'state_dict':model.cpu().state_dict(),'fold':fold,'seed':seed,'steps':STEPS},out/f'fold{fold}.pt')
  model.cuda().eval()
  with torch.inference_mode():
   for row,l,g,y,_ in held:
    before=score(labels(l),y)
    pred,_=model(torch.tensor(l[None],device='cuda'),torch.tensor(g[None,None],device='cuda'))
    after=score(labels(pred[0].cpu().numpy()),y);baseline.append(before);candidate.append(after)
    records.append({'name':row['name'],'fold':fold,'before':before,'after':after})
  del model;torch.cuda.empty_cache();foreign_check()
 assert len(records)==70 and all(sha(Path(p))==h for p,h in frozen.items())
 fold_delta=[100*np.mean([np.mean(x['after'])-np.mean(x['before']) for x in records if x['fold']==f]) for f in range(1,5)]
 base=float(np.mean(baseline)*100);value=float(np.mean(candidate)*100)
 assert abs(base-79.9858620080774)<1e-8,('Baseline parity failed',base)
 checks={'gain_at_least_0_10pp':value-base>=.1,'positive_in_3_folds':sum(x>0 for x in fold_delta)>=3,'worst_fold_at_least_minus_0_10pp':min(fold_delta)>=-.1}
 result={'baseline_percent':base,'candidate_meta_oof_percent':value,'delta_pp':value-base,'fold_delta_pp':fold_delta,'checks':checks,'screen_pass':all(checks.values()),'records':records,'feature_sha256':frozen,'valid_read':False,'test_read':False,'parent_features_unchanged':True,'independent_generalization_verified':False,'limitation':'Meta-OOF reuses existing base OOF; not nested base retraining. Some meta-training features were generated by base models trained with held-fold specimens.','elapsed_seconds':time.monotonic()-start}
 write(ROOT/'result.json',result);write(ROOT/'completion.json',{'finished':True,'folds':4,'updates':4*STEPS,'screen_pass':result['screen_pass']});print(json.dumps({k:result[k] for k in ['baseline_percent','candidate_meta_oof_percent','delta_pp','screen_pass']},indent=2))
if __name__=='__main__':
 try:main()
 except BaseException:write(ROOT/'failure.json',{'traceback':traceback.format_exc()});raise
