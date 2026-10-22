import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import json,random,time,subprocess
import numpy as np
import torch
from residual_model import Residual
from fields_io import sha,write
ROOT=Path(__file__).resolve().parent;SEED=81349
def soft_iou(prob,target):
 gt=torch.nn.functional.one_hot(target,4).permute(0,3,1,2).float()
 intersection=(prob*gt).sum((2,3));union=prob.sum((2,3))+gt.sum((2,3))-intersection
 return intersection/union.clamp_min(1e-6)
def objective(logits,target,delta):
 ce=-torch.nn.functional.log_softmax(logits,dim=1).gather(1,target[:,None]).mean()
 return .25*ce+.75*(1-soft_iou(logits.softmax(1),target).mean())+.05*delta.square().mean()
def main():
 torch.set_num_threads(2);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
 manifest=json.loads((ROOT/'features/train/manifest.json').read_text());assert manifest['split']=='train' and len(manifest['images'])==70
 assert len({x['name'] for x in manifest['images']})==70;rows=[]
 for row in sorted(manifest['images'],key=lambda x:x['name']):
  p=ROOT/'features/train'/(row['name']+'.npz');assert sha(p)==row['features_sha256']
  with np.load(p,allow_pickle=False) as z:lp=z['logits'];gray=z['gray'];gt=z['target']
  assert lp.dtype==np.float32 and lp.shape==(4,*gt.shape) and gray.shape==gt.shape and np.isfinite(lp).all() and set(np.unique(gt))<=set(range(4))
  rows.append((row['name'],lp,gray,gt))
 out=ROOT/'training';out.mkdir(exist_ok=False);model=Residual().cuda()
 opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=8)
 recipe={'seed':SEED,'epochs':8,'updates_planned':560,'batch_size':1,'native_full_image':True,'resize':False,'crop':False,'parameters':11876,'loss':'0.25CE+0.75(1-mean_image4classSoftIoU)+0.05mean(delta2)','train_features_manifest_sha256':sha(ROOT/'features/train/manifest.json'),'source_network_oof':manifest.get('source_network_components_group_oof',False),'source_sha256':{p.name:sha(p) for p in ROOT.glob('*.py')},'valid_read':False,'test_read':False}
 write(out/'recipe.json',recipe);updates=0;history=[];start=time.monotonic()
 for epoch in range(1,9):
  model.train();total=0.;order=np.random.default_rng(SEED+epoch).permutation(70)
  for position,index in enumerate(order):
   assert time.monotonic()-start<1200,'native training cap exceeded'
   name,lp,gray,gt=rows[index];rng=np.random.default_rng(SEED+epoch*100000+int(index))
   a=lp;b=gray[None];c=gt
   if rng.random()<.5:a=a[...,::-1];b=b[...,::-1];c=c[...,::-1]
   if rng.random()<.5:a=a[...,::-1,:];b=b[...,::-1,:];c=c[::-1,:]
   k=int(rng.integers(4));a=np.rot90(a,k,axes=(-2,-1));b=np.rot90(b,k,axes=(-2,-1));c=np.rot90(c,k)
   logits=torch.from_numpy(np.ascontiguousarray(a[None])).cuda();guide=torch.from_numpy(np.ascontiguousarray(b[None])).cuda();target=torch.from_numpy(np.ascontiguousarray(c[None])).long().cuda()
   opt.zero_grad(set_to_none=True);value,delta=model(logits,guide)
   if updates==0:assert torch.equal(value,logits) and torch.count_nonzero(delta)==0
   loss=objective(value,target,delta);assert torch.isfinite(loss);loss.backward()
   assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
   assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));opt.step();updates+=1;total+=loss.item()
   if position%10==0:
    apps=subprocess.check_output(['nvidia-smi','-i',os.environ.get('KOMAP_PHYSICAL_GPU','3'),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip().splitlines();assert len(apps)<=1,'foreign GPU compute; exit our training only'
    print(json.dumps({'epoch':epoch,'image_position':position,'image':name,'updates':updates,'shape':list(gt.shape),'loss':total/(position+1),'seconds':time.monotonic()-start}),flush=True)
  scheduler.step();torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'seed':SEED,'epoch':epoch,'parameters':11876},out/f'epoch{epoch}.pt')
  torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'updates':updates,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},out/'train_state.pt')
  history.append({'epoch':epoch,'updates':updates,'loss':total/70});write(out/'progress.json',history)
 assert updates==560;final=out/'epoch8.pt';write(out/'completion.json',{'epochs':8,'optimizer_updates':updates,'elapsed_seconds':time.monotonic()-start,'final_checkpoint':str(final),'final_sha256':sha(final),'checkpoint_bytes':final.stat().st_size,'recipe':recipe,'valid_read':False,'test_read':False})
if __name__=='__main__':main()
