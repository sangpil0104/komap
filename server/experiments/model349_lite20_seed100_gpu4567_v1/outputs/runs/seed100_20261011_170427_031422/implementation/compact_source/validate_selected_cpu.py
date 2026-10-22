"""Recompute the selected configuration on CPU and compare GPU labels."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import json,sys,hashlib,time
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parent;REMOTE=Path('/workspace/model349_compress');OLD=REMOTE/'fusion805_compression_20261010'
REF=REMOTE/'accuracy81_oof_zgray_20261009';DATA=Path('/workspace/komap_model1')
B4=DATA/'outputs/model329_fullstats_width4000_uniform_group_oof/valid_ensemble/probabilities.npz'
BASE=REMOTE/'accuracy81_candidate_oracle_20261009/publication805_v1/canonical_labels'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def labels(p):return np.array([2,0,1,3],dtype=np.uint8)[p[[2,0,1,3]].argmax(0)]
def main():
 result=json.loads((ROOT/'dual_seed/result.json').read_text());receipt=json.loads((ROOT/'dual_seed/replay_completion.json').read_text());key=result['results']['stage2']['selected'];cfg=next(c for c in receipt['configs'] if c['key']==key)
 out=ROOT/'selected_cpu';assert not out.exists()
 for stage in [1,2]:
  for kind in ['predicted_labels','predicted_masks']:(out/f'stage{stage}/{kind}').mkdir(parents=True)
 sys.path.insert(0,str(REF));from residual_model import Residual;from gray_input import prepare_gray
 training=json.loads((REF/'training/completion.json').read_text());checkpoint=Path(training['final_checkpoint']);assert sha(checkpoint)==training['final_sha256']
 torch.set_num_threads(2);model=Residual().eval();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict'])
 palette=np.array([(25,76,153),(76,178,76),(153,127,76),(204,204,204)],dtype=np.uint8);rows=[];start=time.time()
 with np.load(B4,allow_pickle=False) as archive:
  for name in receipt['names']:
   prob=cfg['weights'][0]*archive[name].astype(np.float32)
   for seed in [1,2]:
    keep=cfg[f'hr{seed}']
    if not keep:continue
    parent=ROOT/'seed1_maps' if seed==1 else OLD/'fold_maps';shard=next(parent/f'shard{i}' for i in [0,1] if (parent/f'shard{i}/{name}_fold1.npy').exists())
    part=sum(np.load(shard/f'{name}_fold{f}.npy',allow_pickle=False).astype(np.float32)*(1./len(keep)) for f in keep).astype(np.float16).astype(np.float32);prob=prob+cfg['weights'][seed]*part
   prob=prob[[1,2,0,3]];image=next(p for p in (DATA/'data/valid/images').iterdir() if p.stem.removesuffix('_image')==name)
   gray=prepare_gray(np.asarray(Image.open(image).convert('L'),dtype=np.float32)/255.)
   mode=cfg['R_mode']
   parent_path=BASE/f'{name}.png' if mode=='canonical14' else ROOT/f"r_prune_v2/runs/{ {'cache14':'r_replay25','no347':'r_no347','no322':'r_no322','no322_347':'r_no322_347'}[mode] }/release/predicted_labels/{name}.png"
   parent=np.asarray(Image.open(parent_path))
   with torch.no_grad():value,delta=model(torch.from_numpy(np.log(np.maximum(prob,1e-7))[None]),torch.from_numpy(gray[None,None]))
   assert torch.isfinite(value).all() and delta.abs().max()<=1
   expert=labels(prob);refined=labels(value.numpy()[0]);s1=np.where(np.isin(expert,[0,2]),expert,parent).astype(np.uint8);s2=np.where(refined==2,refined,s1).astype(np.uint8)
   difference={}
   for stage,pred in [(1,s1),(2,s2)]:
    difference[f'stage{stage}']=int(np.count_nonzero(pred!=np.asarray(Image.open(ROOT/f'dual_seed/runs/{key}/stage{stage}/predicted_labels/{name}.png'))))
    dest=out/f'stage{stage}';Image.fromarray(pred).save(dest/f'predicted_labels/{name}.png');Image.fromarray(palette[pred]).save(dest/f'predicted_masks/{name}_mask.png')
   rows.append({'name':name,'cpu_vs_gpu_changed_pixels':difference,'input_sha256':sha(image)});print(name,difference,flush=True)
 (out/'completion.json').write_text(json.dumps({'selected_config':cfg,'images':rows,'cpu_gpu_exact':all(all(v==0 for v in r['cpu_vs_gpu_changed_pixels'].values()) for r in rows),'gt_read':False,'test_read':False,'whole_model_raw_inference':False,'elapsed_seconds':time.time()-start},indent=2)+'\n')
if __name__=='__main__':main()
