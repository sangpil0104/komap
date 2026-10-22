"""Portable image-only inference for a frozen subset of auxiliary HR folds."""
from pathlib import Path
import argparse,json,hashlib,sys

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-package',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--allow-retrained',action='store_true');a=p.parse_args()
 full_config=json.loads(a.config.read_text());config=full_config['expert'];replacement=config['replacement'];b4dir=config.get('b4_dir','b4_w4000');assert b4dir in ('b4_w4000','E1_2_b4','b4_w4000_full70');weights=config['family_weights'];folds={int(k):v for k,v in config['hr_folds'].items()}
 overrides={int(k):v for k,v in config.get('hr_group_overrides',{}).items()};assert set(overrides)<=set([1,2])
 b4ids=config.get("b4_fold_ids",[1,2,3,4]);assert b4ids and len(set(b4ids))==len(b4ids) and all(f in [1,2,3,4] for f in b4ids)
 assert 0<weights[0]<=1 and all(w>=0 for w in weights) and abs(sum(weights)-1)<1e-12
 for seed in [1,2]:assert len(folds[seed])==len(set(folds[seed])) and all(f in [1,2,3,4] for f in folds[seed]) and bool(folds[seed])==bool(weights[seed])
 assert not a.output.exists()
 import numpy as np
 import torch
 reproduction=Path(__file__).resolve().parent
 assert (reproduction/'b4/common.py').is_file(),str(reproduction/'b4/common.py')
 assert (reproduction/'hrnet/komap_seg').is_dir(),str(reproduction/'hrnet/komap_seg')
 sys.path.insert(0,str(reproduction/'b4'))
 sys.path.insert(0,str(reproduction/'hrnet'))
 from common import load_member,load_gray,predict_probs
 from komap_seg.checkpoint import load_model_and_preprocessing,resolve_preprocessing
 from komap_seg.dataset import KomapFullDataset,effective_input_scale,scaled_overlap
 from komap_seg.inference import predict_probs_tta
 runtime_config=Path(__file__).resolve().parents[1]/'config'
 manifest=json.loads((runtime_config/'e_weights_sha256.json').read_text());expected={row['path']:row['sha256'] for row in manifest};frozen={}
 if b4dir=='E1_2_b4':expected.update({row['path']:row['sha256'] for row in json.loads((runtime_config/'e1_2_b4_sha256.json').read_text())})
 required=[f'{b4dir}/fold{f}/{filename}' for f in b4ids for filename in ['last.pt','args.json']]+[f'E4_hrnet/seed{s}/fold{f}/last.pt' for s in [1,2] if s not in overrides for f in folds[s]]
 required.extend(v['checkpoint'] for v in overrides.values())
 old=f"E4_hrnet/seed{replacement['seed']}/fold{replacement['fold']}/last.pt";required.remove(old);required.extend([replacement['checkpoint'],replacement['args']])
 expected.update({row['path']:row['sha256'] for row in json.loads((a.config.parent/full_config['assets_manifest_file']).read_text())['checkpoints']+json.loads((a.config.parent/full_config['assets_manifest_file']).read_text())['metadata'] if row['namespace']=='E'})
 for relative in required:
  path=a.assets/relative;frozen[str(path)]=sha(path)
  if a.allow_retrained:continue  # retrained weights: SHA recorded in completion.json, not checked
  assert frozen[str(path)]==expected[relative],relative
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;device=torch.device(a.device)
 b4=[load_member(a.assets/f'{b4dir}/fold{f}',a.device,'last.pt') for f in b4ids];hr=[]
 for seed in [1,2]:
  selected=[None] if seed in overrides else folds[seed]
  for fold in selected:
   is_n3=(seed,fold)==(replacement['seed'],replacement['fold'])
   ck=a.assets/(overrides[seed]['checkpoint'] if seed in overrides else replacement['checkpoint'] if is_n3 else f'E4_hrnet/seed{seed}/fold{fold}/last.pt')
   model,saved=load_model_and_preprocessing(ck,device);prep=resolve_preprocessing(saved)
   assert prep['patch_size']==512 and prep['input_scale']==2.
   if is_n3:assert not prep['normalize_width']
   else:assert prep['normalize_width'] and prep['target_width']>0
   hr.append((seed,model,prep,is_n3,1 if seed in overrides else len(folds[seed])))
 dataset=KomapFullDataset(a.input,masks_dir=None);input_sha={str(x['image']):sha(x['image']) for x in dataset.pairs};a.output.mkdir(parents=True);rows=[]
 from zcdf_input import prepare
 encoded=prepare(a.input,a.output/'n3_encoded_input');n3dataset=KomapFullDataset(encoded,masks_dir=None)
 assert [p['stem'] for p in n3dataset.pairs]==[p['stem'] for p in dataset.pairs]
 for i,pair in enumerate(dataset.pairs):
  sample=dataset[i];name=pair['stem'];shape=(sample['height'],sample['width'])
  b=predict_probs(b4,load_gray(pair['image']),tta=True,device=a.device).cpu().numpy().astype(np.float16).astype(np.float32)
  sums={s:np.zeros((4,*shape),dtype=np.float32) for s in [1,2] if folds[s]}
  for seed,model,prep,is_n3,divisor in hr:
   input_sample=n3dataset[i] if is_n3 else sample
   scale=effective_input_scale(prep['input_scale'],prep['target_width'],sample['width']);overlap=scaled_overlap(512,2.,170,scale)
   probs=predict_probs_tta(model,input_sample['image'],patch_size=512,overlap=overlap,batch_size=2,tta=not is_n3,input_scale=scale,device=device,tta_mode='flip4')
   value=probs.numpy().astype(np.float32)
   if is_n3:value=value.astype(np.float16).astype(np.float32)
   sums[seed]+=value*(1./divisor)
  value=weights[0]*b
  for seed in [1,2]:
   if folds[seed]:value=value+weights[seed]*sums[seed].astype(np.float16).astype(np.float32)
  value=value[[1,2,0,3]];assert np.isfinite(value).all();path=a.output/f'{name}.npy';np.save(path,value);rows.append({'name':name,'sha256':sha(path)})
  print(name,flush=True)
 assert all(sha(p)==h for p,h in frozen.items())
 assert all(sha(p)==h for p,h in input_sha.items())
 (a.output/'completion.json').write_text(json.dumps({'images':rows,'config':config,'input_checkpoint_sha256':frozen,'input_image_sha256':input_sha,'inference_source_sha256':sha(__file__),'gt_read':False,'fresh_expert_inference':True,'allow_retrained':a.allow_retrained},indent=2)+'\n')
if __name__=='__main__':main()
