import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import json,random,time,subprocess
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from residual_model import Residual
from fields_io import sha,write
ROOT=Path(__file__).resolve().parent;SEED=81349
class Crops(Dataset):
 def __init__(self,root):
  self.epoch=0;self.rows=[];self.manifest=json.loads((root/'manifest.json').read_text());assert self.manifest['split']=='train' and len(self.manifest['images'])==70
  for row in self.manifest['images']:
   p=root/(row['name']+'.npz');assert sha(p)==row['features_sha256']
   with np.load(p) as z:logits=z['logits'];gray=z['gray'];target=z['target']
   assert logits.dtype==np.float32 and logits.shape==(4,*target.shape) and gray.shape==target.shape and np.isfinite(logits).all() and set(np.unique(target))<=set(range(4))
   coords={c:np.flatnonzero(target==c) for c in [0,3]};self.rows.append((logits,gray,target,coords))
 def __len__(self):return 70*16
 def __getitem__(self,index):
  logits,gray,target,coords=self.rows[index//16];rng=np.random.default_rng(SEED+self.epoch*100000+index);h,w=target.shape;p=256;assert min(h,w)>=p
  active=[c for c in coords if len(coords[c])]
  if active and rng.random()<.5:
   cls=active[rng.integers(len(active))];center=coords[cls][rng.integers(len(coords[cls]))];cy,cx=divmod(int(center),w);y=int(np.clip(cy-rng.integers(p),0,h-p));x=int(np.clip(cx-rng.integers(p),0,w-p))
  else:y=int(rng.integers(h-p+1));x=int(rng.integers(w-p+1))
  a=logits[:,y:y+p,x:x+p];b=gray[None,y:y+p,x:x+p];c=target[y:y+p,x:x+p]
  if rng.random()<.5:a=a[...,::-1];b=b[...,::-1];c=c[...,::-1]
  if rng.random()<.5:a=a[...,::-1,:];b=b[...,::-1,:];c=c[::-1,:]
  k=int(rng.integers(4));a=np.rot90(a,k,axes=(-2,-1));b=np.rot90(b,k,axes=(-2,-1));c=np.rot90(c,k)
  return torch.from_numpy(np.ascontiguousarray(a)),torch.from_numpy(np.ascontiguousarray(b)),torch.from_numpy(np.ascontiguousarray(c)).long()
def objective(logits,target,delta,class_weights=None):
 p=logits.softmax(1);gt=torch.nn.functional.one_hot(target,4).permute(0,3,1,2).float();inter=(p*gt).sum((0,2,3));den=(p+gt).sum((0,2,3));dice=1-((2*inter+1e-6)/(den+1e-6)).mean()
 nll=-torch.nn.functional.log_softmax(logits,dim=1).gather(1,target[:,None]).squeeze(1)
 if class_weights is None:ce=nll.mean()
 else:
  weights=class_weights[target];ce=(nll*weights).sum()/weights.sum()
 return .5*ce+.5*dice+.05*delta.square().mean()
def main():
 torch.set_num_threads(2);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
 out=ROOT/'training';out.mkdir(exist_ok=False);ds=Crops(ROOT/'features/train');generator=torch.Generator().manual_seed(SEED);loader=DataLoader(ds,batch_size=8,shuffle=True,num_workers=0,generator=generator,pin_memory=True)
 frequencies=np.mean([np.bincount(target.ravel(),minlength=4)/target.size for _,_,target,_ in ds.rows],axis=0)
 weights=1/np.sqrt(frequencies+1e-6);weights/=weights.mean();class_weights=torch.tensor(weights,dtype=torch.float32,device='cuda')
 assert len(loader)==140;model=Residual().cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=8)
 recipe={'seed':SEED,'epochs':8,'optimizer_updates_planned':1120,'parameters':11876,'train_features_manifest_sha256':sha(ROOT/'features/train/manifest.json'),'code_sha256':{p.name:sha(p) for p in ROOT.glob('*.py')},'valid_read':False,'test_read':False,'class_frequencies_mean_train_image':frequencies.tolist(),'ce_class_weights':weights.tolist(),'class_weight_rule':'inverse sqrt(mean image train70 class fraction + 1e-6), mean normalized'};write(out/'recipe.json',recipe)
 start=time.monotonic();updates=0;rows=[]
 for epoch in range(1,9):
  ds.epoch=epoch;model.train();total=0.
  for i,(logits,gray,target) in enumerate(loader):
   assert time.monotonic()-start<1200,'training cap exceeded';logits=logits.cuda();gray=gray.cuda();target=target.cuda();opt.zero_grad(set_to_none=True)
   predicted,delta=model(logits,gray);loss=objective(predicted,target,delta,class_weights);assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
   assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));opt.step();updates+=1;total+=loss.item()
   if i%20==0:
    apps=subprocess.check_output(['nvidia-smi','-i',os.environ.get('KOMAP_PHYSICAL_GPU','7'),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    if len(apps)>1:torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'minibatch':i},out/'interrupted_foreign.pt');raise RuntimeError('Foreign compute appeared on selected GPU; stop our training only')
    print(json.dumps({'epoch':epoch,'minibatch':i,'minibatches':140,'updates':updates,'loss':total/(i+1),'seconds':time.monotonic()-start}),flush=True)
  scheduler.step();torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'seed':SEED,'epoch':epoch,'parameters':11876},out/f'epoch{epoch}.pt')
  torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'updates':updates,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'loader_rng':generator.get_state()},out/'train_state.pt')
  rows.append({'epoch':epoch,'loss':total/len(loader),'updates':updates});write(out/'progress.json',rows)
 assert updates==1120;final=out/'epoch8.pt';write(out/'completion.json',{'epochs':8,'optimizer_updates':updates,'elapsed_seconds':time.monotonic()-start,'final_checkpoint':str(final),'final_sha256':sha(final),'checkpoint_bytes':final.stat().st_size,'recipe':recipe,'valid_read':False,'test_read':False})
if __name__=='__main__':main()
