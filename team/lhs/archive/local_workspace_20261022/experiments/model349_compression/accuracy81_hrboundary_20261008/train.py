import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',NO_ALBUMENTATIONS_UPDATE='1')
import argparse,hashlib,json,random,sys,time,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parent;SOURCE=ROOT.parent/'Model349_exh14'
sys.path.insert(0,str(SOURCE/'src'))
from komap_seg.model import build_model
from training_dataset import KomapPatchDataset
from loss import objective
from training_boundary import BoundaryHeadLoss, forward_with_boundary
boundary_loss=BoundaryHeadLoss()

def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class PairedDataset(KomapPatchDataset):
 epoch=0
 def __getitem__(self,index):
  seed=81336+self.epoch*100000+index
  self.rng.seed(seed);random.seed(seed);np.random.seed(seed)
  self.transform.set_random_seed(seed)
  return super().__getitem__(index)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['boundary'],required=True);a=ap.parse_args()
 torch.set_num_threads(2);seed=81336;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 out=ROOT/'training'/a.arm;out.mkdir(parents=True,exist_ok=False)
 data=Path('/workspace/komap_model1/data/train');inputs=sorted(data.glob('images/*_image.png'))+sorted(data.glob('masks/*_mask.png'));assert len(inputs)==140
 ckpt=SOURCE/'weights/model336_hrnet_best/last.pt';parent_sha=sha(ckpt)
 receipts={'arm':a.arm,'train_files':{str(p):sha(p) for p in inputs},'train_images':70,'parent_sha256':parent_sha,'epochs':4,'seed':seed,'valid_read_during_training':False,'test_read_during_training':False,'script_sha256':sha(Path(__file__)),'loss_sha256':sha(ROOT/'loss.py'),'dataset_sha256':sha(ROOT/'training_dataset.py'),'palette_sha256':sha(ROOT/'training_palette.py'),'boundary_sha256':sha(ROOT/'training_boundary.py'),'boundary_weight':1.,'segmentation_objective':'0.5CE+0.5Dice'}
 write(out/'recipe.json',receipts)
 ds=PairedDataset(data/'images',data/'masks',patch_size=512,samples_per_image=12,input_scale=2.,seed=seed);assert len(ds.stems)==70
 generator=torch.Generator().manual_seed(seed)
 loader=DataLoader(ds,batch_size=2,shuffle=True,num_workers=0,drop_last=True,pin_memory=True,generator=generator)
 saved=torch.load(ckpt,map_location='cpu',weights_only=False);model=build_model(**saved['model_config'],encoder_weights=None);model.load_state_dict(saved['state_dict']);model.cuda()
 enc=list(model.encoder.parameters());encids={id(p) for p in enc};dec=[p for p in model.parameters() if id(p) not in encids and p.requires_grad]
 opt=torch.optim.AdamW([{'params':enc,'lr':5e-6},{'params':dec,'lr':2e-5}],weight_decay=1e-4)
 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=4);scaler=torch.amp.GradScaler('cuda',init_scale=1024)
 start=time.monotonic();updates=0;records=[]
 for epoch in range(1,5):
  ds.epoch=epoch;model.train()
  for m in model.modules():
   if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
  opt.zero_grad(set_to_none=True);total=0.;epoch_start=time.monotonic()
  for i,(image,target) in enumerate(loader):
   assert time.monotonic()-start<2400,'arm budget exceeded'
   torch.manual_seed(seed+epoch*100000+i);torch.cuda.manual_seed_all(seed+epoch*100000+i)
   image=image.cuda(non_blocking=True);target=target.cuda(non_blocking=True)
   with torch.autocast('cuda',dtype=torch.float16):logits,boundary=forward_with_boundary(model,image)
   value=objective(logits,target,'control')+boundary_loss(boundary,target);assert torch.isfinite(value),'nonfinite loss'
   scaler.scale(value/4).backward();total+=value.item()
   if (i+1)%4==0 or i+1==len(loader):
    scaler.unscale_(opt)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),'nonfinite gradient'
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
    scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True);updates+=1
   if i%40==0:
    apps=subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    if len(apps)>1:
     torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),'epoch':epoch,'minibatch':i,'updates':updates},out/'interrupted_foreign_gpu.pt')
     raise RuntimeError('Foreign GPU3 compute appeared; stop only our training')
    print(json.dumps({'arm':a.arm,'epoch':epoch,'minibatch':i,'minibatches':len(loader),'loss':total/(i+1),'updates':updates,'seconds':time.monotonic()-start}),flush=True)
  scheduler.step();model.eval()
  state={k:v.detach().cpu() for k,v in model.state_dict().items()};payload=dict(saved,state_dict=state)
  torch.save(payload,out/f'epoch{epoch}.pt')
  torch.save({'model':state,'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'updates':updates,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'loader_rng':generator.get_state()},out/'train_state.pt')
  row={'epoch':epoch,'loss':total/len(loader),'seconds':time.monotonic()-epoch_start,'updates':updates};records.append(row);write(out/'progress.json',records)
 final=out/'epoch4.pt';assert sha(ckpt)==parent_sha
 write(out/'completion.json',{'arm':a.arm,'epochs':4,'optimizer_updates':updates,'elapsed_seconds':time.monotonic()-start,'final_checkpoint':str(final),'final_sha256':sha(final),'parent_unchanged':True,'valid_read':False,'test_read':False,'recipe':receipts})
if __name__=='__main__':main()
