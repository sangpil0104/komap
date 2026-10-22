"""Bounded fixed-step hard/soft pilot using only train70, frozen deployed329 teacher."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
import argparse,collections,hashlib,json,re,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
ROOT=Path('/workspace/model349_compress/base12_partial_distill329_20261008')
BASE=Path('/workspace/model349_compress/audit_20261008/bundles/cpu2_fp16_12')
DATA=Path('/workspace/komap_model1/data/train')
sys.path.insert(0,str(BASE/'src'));import b4_zcdf as b4
import cv2
cv2.setNumThreads(2);torch.set_num_threads(2)
PALETTE=np.array([[153,127,76],[25,76,153],[76,178,76],[204,204,204]],np.uint8)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def paths():return [BASE/'weights/model329_fullstats_width4000_uniform_group_oof'/f'fold{i}' for i in range(1,5)]
def mean_state():
 states=[torch.load(p/'last.pt',map_location='cpu',weights_only=False) for p in paths()]
 assert all(s.keys()==states[0].keys() for s in states)
 out={}
 for k,a in states[0].items():
  assert all(s[k].shape==a.shape and s[k].dtype==a.dtype for s in states)
  out[k]=sum(s[k].float()*.25 for s in states) if a.is_floating_point() else torch.stack([s[k] for s in states]).max(0).values
 return out
class Sampler:
 def __init__(self):
  self.files=sorted((DATA/'images').glob('*_image.png'));assert len(self.files)==70
  self.cache=collections.OrderedDict();self.rng=np.random.default_rng(20261008);self.rows=[]
  for p in self.files:
   q=DATA/'masks'/p.name.replace('_image.png','_mask.png');assert q.is_file()
   self.rows.append({'name':p.stem.removesuffix('_image'),'group':re.match(r'A\d+',p.name)[0],'image_sha256':sha(p),'mask_sha256':sha(q),'image_path':str(p),'mask_path':str(q)})
 def load(self,index):
  if index not in self.cache:
   p=self.files[index];gray=b4.load_gray(p);scale=4000/gray.shape[1];gray=b4.resize_gray(gray,scale)
   mask=np.array(Image.open(DATA/'masks'/p.name.replace('_image.png','_mask.png')).convert('RGB'));label=np.full(mask.shape[:2],255,np.uint8)
   for cid,color in enumerate(PALETTE):label[np.all(mask==color,axis=-1)]=cid
   assert (label<4).all();label=cv2.resize(label,(gray.shape[1],gray.shape[0]),interpolation=cv2.INTER_NEAREST)
   values=gray.astype(np.float32)/255;stats=(float(values.mean()),float(values.std()));cdf=(np.cumsum(np.bincount(gray.reshape(-1),minlength=256))/gray.size).astype(np.float32)
   self.cache[index]=(gray,label,stats,cdf)
   if len(self.cache)>4:self.cache.popitem(last=False)
  else:self.cache.move_to_end(index)
  return self.cache[index]
 def sample(self):
  index=int(self.rng.integers(70));gray,label,stats,cdf=self.load(index);h,w=gray.shape;assert min(h,w)>=512
  y=int(self.rng.integers(h-512+1));x=int(self.rng.integers(w-512+1));g=gray[y:y+512,x:x+512];m=label[y:y+512,x:x+512];k=int(self.rng.integers(8))
  if k>=4:g=np.fliplr(g);m=np.fliplr(m)
  g=np.rot90(g,k%4);m=np.rot90(m,k%4);inp=b4.make_input(g,'image_global','zcdf3',None,stats,cdf)
  return torch.from_numpy(np.ascontiguousarray(inp))[None],torch.from_numpy(np.ascontiguousarray(m)).long()[None],{'index':index,'name':self.rows[index]['name'],'x':x,'y':y,'d4':k}
def tensor_hash(model):
 h=hashlib.sha256()
 for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def check_gpu():
 uuid=os.environ['PILOT_GPU_UUID'];pids=os.popen('nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader').read()
 if sum(uuid in line for line in pids.splitlines())>1:raise RuntimeError('Foreign compute on GPU7; stopping our training')
def main(mode):
 assert mode in ['hard','distill'];assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7';torch.manual_seed(20261008);torch.cuda.manual_seed_all(20261008)
 folder=ROOT/'training'/mode;folder.mkdir(parents=True,exist_ok=False);sampler=Sampler();write(folder/'training_manifest.json',{'split':'train70 only','valid_used_for_training_or_epoch_selection':False,'rows':sampler.rows})
 args=json.loads((paths()[0]/'args.json').read_text());assert all(json.loads((p/'args.json').read_text())==args for p in paths())
 student=b4.build_model(args['arch'],args['encoder'],pretrained=False).cuda();student.load_state_dict(mean_state(),strict=True);student.train()
 teachers=[]
 if mode=='distill':
  for p in paths():
   model=b4.load_member(str(p),'cuda')['model'];model.eval()
   for param in model.parameters():param.requires_grad_(False)
   teachers.append(model)
 teacher_hashes=[tensor_hash(model) for model in teachers]
 # Pair the student's dropout/drop-connect RNG despite extra teacher construction.
 torch.manual_seed(20261008);torch.cuda.manual_seed_all(20261008)
 opt=torch.optim.AdamW(student.parameters(),lr=1e-5,weight_decay=1e-4);scaler=torch.amp.GradScaler('cuda',init_scale=1024)
 start=time.monotonic();history=[];crop_rows=[];completed=False;successful=0
 for step in range(1,401):
  if step==1 or step%25==0:check_gpu()
  if time.monotonic()-start>1800:break
  x,y,info=sampler.sample();crop_rows.append(info);x=x.cuda();y=y.cuda();opt.zero_grad(set_to_none=True)
  target=None
  if teachers:
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
    p=sum(model(x).float().softmax(1) for model in teachers)/4;target=p.clamp_min(1e-7).pow(.5);target=target/target.sum(1,keepdim=True)
  with torch.autocast('cuda',dtype=torch.float16):
   logits=student(x);ce=F.cross_entropy(logits,y);prob=logits.float().softmax(1);onehot=F.one_hot(y,4).permute(0,3,1,2).float();inter=(prob*onehot).sum((0,2,3));den=(prob+onehot).sum((0,2,3));dice=1-((2*inter+1)/(den+1)).mean();kl=F.kl_div(F.log_softmax(logits.float()/2,dim=1),target,reduction='none').sum(1).mean() if target is not None else logits.new_zeros((),dtype=torch.float32);loss=ce+dice+.2*4*kl
  assert torch.isfinite(loss), 'Nonfinite loss';scaler.scale(loss).backward();scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(student.parameters(),1.)
  if not torch.isfinite(norm):raise RuntimeError('Nonfinite unscaled gradient; no silent skipped updates')
  old_scale=scaler.get_scale();scaler.step(opt);scaler.update();assert scaler.get_scale()>=old_scale,'GradScaler skipped update';successful+=1
  if step==5:assert [tensor_hash(model) for model in teachers]==teacher_hashes
  if step<=5 or step%25==0:
   row={'step':step,'ce':float(ce.detach()),'dice':float(dice.detach()),'kl':float(kl.detach()),'loss':float(loss.detach()),'grad_norm':float(norm),'seconds':time.monotonic()-start};history.append(row);write(folder/'history.json',history);print(json.dumps(row),flush=True)
  if step%100==0:
   torch.save({'model':student.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'step':step,'sampler_rng':sampler.rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'crop_rows':crop_rows},folder/'resume.pt')
  if step==400:completed=True
 assert [tensor_hash(model) for model in teachers]==teacher_hashes
 state={k:v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu() for k,v in student.state_dict().items()};torch.save(state,folder/'last.pt');write(folder/'args.json',args);write(folder/'crop_sequence.json',crop_rows)
 summary={'mode':mode,'completed':completed,'successful_updates':successful,'configured_updates':400,'epoch_selection':False,'teacher_state_unchanged':True,'teacher_sha256':{str(p/'last.pt'):sha(p/'last.pt') for p in paths()},'checkpoint_sha256':sha(folder/'last.pt'),'checkpoint_bytes':(folder/'last.pt').stat().st_size,'elapsed_seconds':time.monotonic()-start,'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(),'independent_generalization_verified':False};write(folder/'summary.json',summary);print(json.dumps(summary),flush=True)
 if not completed:raise RuntimeError('Fixed trial hit 30 minute limit; partial checkpoint retained')
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['hard','distill'],required=True);args=parser.parse_args();main(args.mode)
