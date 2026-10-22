"""Portable image-only inference for a frozen subset of auxiliary HR folds."""
from pathlib import Path
import argparse,json,hashlib,sys,time

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-package',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
 whole=json.loads(a.config.read_text());config=whole['expert'];weights=config['family_weights'];folds={int(k):v for k,v in config['hr_folds'].items()}
 assert weights[0]==.5 and abs(sum(weights)-1)<1e-12
 for seed in [1,2]:assert len(folds[seed])==len(set(folds[seed])) and all(f in [1,2,3,4] for f in folds[seed]) and bool(folds[seed])==bool(weights[seed])
 assert not a.output.exists()
 import numpy as np
 import torch
 sys.path.insert(0,str(a.parent_package/'model397_reproduction_bundle/model329'));sys.path.insert(0,str(a.parent_package/'model397_reproduction_bundle/model392'))
 from common import load_member,load_gray,predict_probs
 from komap_seg.checkpoint import load_model_and_preprocessing,resolve_preprocessing
 from komap_seg.dataset import KomapFullDataset,effective_input_scale,scaled_overlap
 from komap_seg.inference import predict_probs_tta
 manifest=json.loads((a.config.parent/whole['assets_manifest_file']).read_text());manifest=manifest['checkpoints']+manifest['metadata'];expected={row['path']:row['sha256'] for row in manifest};frozen={}
 required=[f'model329/fold{f}/{filename}' for f in [1,2,3,4] for filename in ['last.pt','args.json']]+[f'model392/seed{s}/fold{f}/last.pt' for s in [1,2] for f in folds[s]]
 for relative in required:
  path=a.assets/relative;frozen[str(path)]=sha(path);assert frozen[str(path)]==expected[relative],relative
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;device=torch.device(a.device)
 b4=[load_member(a.assets/f'model329/fold{f}',a.device,'last.pt') for f in [1,2,3,4]];hr=[]
 for seed in [1,2]:
  for fold in folds[seed]:
   model,saved=load_model_and_preprocessing(a.assets/f'model392/seed{seed}/fold{fold}/last.pt',device);prep=resolve_preprocessing(saved)
   assert prep['patch_size']==512 and prep['input_scale']==2. and prep['normalize_width'] and prep['target_width']>0
   hr.append((seed,model,prep))
 assert all(parameter.dtype==torch.float32 for member in b4 for parameter in member['model'].parameters())
 assert all(parameter.dtype==torch.float32 for _,model,_ in hr for parameter in model.parameters())
 dataset=KomapFullDataset(a.input,masks_dir=None);input_sha={str(x['image']):sha(x['image']) for x in dataset.pairs};a.output.mkdir(parents=True);rows=[]
 for i,pair in enumerate(dataset.pairs):
  compute_start=time.perf_counter();sample=dataset[i];name=pair['stem'];shape=(sample['height'],sample['width'])
  b=predict_probs(b4,load_gray(pair['image']),tta=True,device=a.device).cpu().numpy().astype(np.float16).astype(np.float32)
  sums={s:np.zeros((4,*shape),dtype=np.float32) for s in [1,2] if folds[s]}
  for seed,model,prep in hr:
   scale=effective_input_scale(prep['input_scale'],prep['target_width'],sample['width']);overlap=scaled_overlap(512,2.,170,scale)
   probs=predict_probs_tta(model,sample['image'],patch_size=512,overlap=overlap,batch_size=2,tta=True,input_scale=scale,device=device,tta_mode='flip4')
   sums[seed]+=probs.numpy().astype(np.float32)*(1./len(folds[seed]))
  value=weights[0]*b
  for seed in [1,2]:
   if folds[seed]:value=value+weights[seed]*sums[seed].astype(np.float16).astype(np.float32)
  value=value[[1,2,0,3]];compute_seconds=time.perf_counter()-compute_start;assert np.isfinite(value).all();path=a.output/f'{name}.npy';np.save(path,value);rows.append({'name':name,'sha256':sha(path),'HR_forwards':4,'compute_seconds':compute_seconds})
  print(name,flush=True)
 assert all(sha(p)==h for p,h in frozen.items())
 assert all(sha(p)==h for p,h in input_sha.items())
 (a.output/'completion.json').write_text(json.dumps({'images':rows,'config':config,'input_checkpoint_sha256':frozen,'input_image_sha256':input_sha,'inference_source_sha256':sha(__file__),'gt_read':False,'fresh_expert_inference':True},indent=2)+'\n')
if __name__=='__main__':main()
