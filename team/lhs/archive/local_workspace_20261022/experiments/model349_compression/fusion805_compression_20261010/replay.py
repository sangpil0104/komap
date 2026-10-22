"""GT-free family-pruning replay; F recomputed for every changed input."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parent
REMOTE=Path('/workspace/model349_compress')
REF=REMOTE/'accuracy81_oof_zgray_20261009'
HERE=REMOTE/'accuracy81_candidate_oracle_20261009'
PACK=HERE/'publication805_v1/Model349_fusion805'
BASE=HERE/'publication805_v1/canonical_labels'
DATA=Path('/workspace/komap_model1')
SRC=DATA/'outputs/model401_model397_blend_valid_once'
B4=DATA/'outputs/model329_fullstats_width4000_uniform_group_oof/valid_ensemble/probabilities.npz'
GRID={'control':[.5,.25,.25],'seed1':[.5,.5,0],'seed2':[.5,0,.5],'b4':[1,0,0],'hr':[0,.5,.5],'hr1':[0,1,0],'hr2':[0,0,1]}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def labels(p):return np.array([2,0,1,3],dtype=np.uint8)[p[[2,0,1,3]].argmax(0)]
def main():
 assert not (ROOT/'runs').exists()
 sys.path.insert(0,str(REF))
 from residual_model import Residual
 from gray_input import prepare_gray
 import yaml
 palette=np.asarray(yaml.safe_load((PACK/'Model349_exh14/configs/model349_lite.yaml').read_text())['palette'],dtype=np.uint8)
 receipt=json.loads((REF/'training/completion.json').read_text());checkpoint=Path(receipt['final_checkpoint'])
 frozen={str(checkpoint):sha(checkpoint),str(B4):sha(B4)}
 assert frozen[str(checkpoint)]==receipt['final_sha256']
 assert frozen[str(B4)]=='80ebfddad10abb1bb5f1d7af8472f477fd5b6435b3c7fef0bfa51e21e1314b05'
 manifest=json.loads((SRC/'inference_manifest.json').read_text())
 for seed in [1,2]:
  for row in manifest[f'hrnet_seed{seed}_mean_maps']:
   path=DATA/row['path'];frozen[str(path)]=sha(path);assert frozen[str(path)]==row['sha256']
 torch.set_num_threads(2);model=Residual().eval();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict'])
 names=[r['name'] for r in json.loads((PACK/'config/input_images.json').read_text())];rows=[];started=time.time()
 for key in GRID:
  for stage in [1,2]:
   for kind in ['predicted_labels','predicted_masks']:(ROOT/f'runs/{key}/stage{stage}/{kind}').mkdir(parents=True)
 with np.load(B4,allow_pickle=False) as archive:
  assert set(archive.files)==set(names)
  for name in names:
   parts=[archive[name].astype(np.float32)]
   for seed in [1,2]:
    with np.load(SRC/f'hrnet_seed{seed}_mean_valid/{name}.npz',allow_pickle=False) as z:parts.append(z[name].astype(np.float32))
   image=DATA/f'data/valid/images/{name}_image.png'
   if not image.exists():image=next(p for p in (DATA/'data/valid/images').iterdir() if p.stem.removesuffix('_image')==name)
   gray=prepare_gray(np.asarray(Image.open(image).convert('L'),dtype=np.float32)/255.)
   parent=np.asarray(Image.open(BASE/f'{name}.png'));frozen[str(image)]=sha(image);frozen[str(BASE/f'{name}.png')]=sha(BASE/f'{name}.png')
   for key,weights in GRID.items():
    prob=sum(w*x for w,x in zip(weights,parts) if w)[[1,2,0,3]]
    assert prob.shape==(4,*parent.shape) and np.isfinite(prob).all()
    with torch.no_grad():value,delta=model(torch.from_numpy(np.log(np.maximum(prob,1e-7))[None]),torch.from_numpy(gray[None,None]))
    expert=labels(prob);refined=labels(value.numpy()[0]);s1=np.where(np.isin(expert,[0,2]),expert,parent).astype(np.uint8);s2=np.where(refined==2,refined,s1).astype(np.uint8)
    if key=='control':
     assert np.array_equal(s2,np.asarray(Image.open(HERE/f'publication805_v1/portable_valid20/predicted_labels/{name}.png'))),name
    for stage,pred in [(1,s1),(2,s2)]:
     dest=ROOT/f'runs/{key}/stage{stage}'
     Image.fromarray(pred).save(dest/f'predicted_labels/{name}.png');Image.fromarray(palette[pred]).save(dest/f'predicted_masks/{name}_mask.png')
   rows.append({'name':name});print(len(rows),name,round(time.time()-started,1),flush=True)
 assert all(sha(p)==h for p,h in frozen.items())
 (ROOT/'replay_completion.json').write_text(json.dumps({'names':names,'grid':GRID,'gt_read':False,'test_read':False,'gpu_used':False,'fresh_raw_inference':False,'control_stage2_exact':True,'input_sha256':frozen,'elapsed_seconds':time.time()-started},indent=2)+'\n')
if __name__=='__main__':
 import traceback
 try:main()
 except BaseException:(ROOT/'failure.json').write_text(traceback.format_exc());raise
