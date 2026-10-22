"""Frozen 25-setting confidence gate screen on fusion805. CPU only."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image

ROOT=Path('/workspace/model349_compress')
HERE=ROOT/'accuracy81_candidate_oracle_20261009'
PACK=HERE/'publication805_v1/Model349_fusion805'
BASE=HERE/'publication805_v1/canonical_labels'
REF=ROOT/'accuracy81_oof_zgray_20261009/matched_source/refined/predicted_labels'
DATA=Path('/workspace/komap_model1')
SRC=DATA/'outputs/model401_model397_blend_valid_once'
B4=DATA/'outputs/model329_fullstats_width4000_uniform_group_oof/valid_ensemble/probabilities.npz'
OUT=HERE/'confidence_screen'
assert not OUT.exists()
OUT.mkdir()
GRID=[(a,b) for a in [0,.025,.05,.1,.2] for b in [0,.025,.05,.1,.2]]
names=[x['name'] for x in json.loads((PACK/'config/input_images.json').read_text())]
import yaml
palette=np.asarray(yaml.safe_load((PACK/'Model349_exh14/configs/model349_lite.yaml').read_text())['palette'],dtype=np.uint8)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert sha(B4)=='80ebfddad10abb1bb5f1d7af8472f477fd5b6435b3c7fef0bfa51e21e1314b05'
manifest=json.loads((SRC/'inference_manifest.json').read_text())
for seed in [1,2]:
 for row in manifest[f'hrnet_seed{seed}_mean_maps']:assert sha(DATA/row['path'])==row['sha256']
def score(pred,target):
 vals=[]
 for c in range(4):
  x=pred==c;y=target==c;u=np.count_nonzero(x|y);vals.append(float(np.count_nonzero(x&y)/u) if u else 0.)
 return vals
def getprob(name,archive):
 parts=[archive[name].astype(np.float32)]
 for seed in [1,2]:
  with np.load(SRC/f'hrnet_seed{seed}_mean_valid/{name}.npz') as z:parts.append(z[name].astype(np.float32))
 return (.5*parts[0]+.25*parts[1]+.25*parts[2])[[1,2,0,3]]
def compose(parent,refined,prob,t0,t2):
 expert=np.array([2,0,1,3],dtype=np.uint8)[prob[[2,0,1,3]].argmax(0)]
 margin=np.sort(prob,axis=0)[-1]-np.sort(prob,axis=0)[-2]
 use=((expert==0)&(margin>=t0))|((expert==2)&(margin>=t2))
 return np.where(refined==2,refined,np.where(use,expert,parent)).astype(np.uint8)
scores=np.zeros((len(GRID),len(names),4))
with np.load(B4) as archive:
 for j,name in enumerate(names):
  prob=getprob(name,archive);parent=np.asarray(Image.open(BASE/f'{name}.png'));refined=np.asarray(Image.open(REF/f'{name}.png'))
  rgb=np.asarray(Image.open(DATA/f'data/valid/masks/{name}_mask.png').convert('RGB'));target=np.full(rgb.shape[:2],255,dtype=np.uint8)
  for c,color in enumerate(palette):target[np.all(rgb==color,axis=-1)]=c
  assert np.all(target<4)
  for i,(a,b) in enumerate(GRID):scores[i,j]=score(compose(parent,refined,prob,a,b),target)
  assert np.array_equal(compose(parent,refined,prob,0,0),np.asarray(Image.open(HERE/f'publication805_v1/portable_valid20/predicted_labels/{name}.png')))
  print(j+1,'/20',flush=True)
assert abs(scores[0].mean()*100-80.50068909458149)<1e-10
means=scores.mean(axis=(1,2));best=int(means.argmax());cv=[]
for j,name in enumerate(names):
 train=np.array([n.split('_')[0]!=name.split('_')[0] for n in names]);chosen=int(scores[:,train].mean(axis=(1,2)).argmax());cv.append(float(scores[chosen,j].mean()))
result={'grid':GRID,'per_setting_percent':(means*100).tolist(),'best_thresholds':GRID[best],'best_percent':float(means[best]*100),'baseline_percent':float(means[0]*100),'delta_pp':float((means[best]-means[0])*100),'conditional_stage1_stage2_fixed_specimen_leave_one_out_percent':float(np.mean(cv)*100),'class_iou':scores.tolist(),'names':names,'cpu_only':True,'fresh_parent_raw_inference':False,'repeated_valid_selection':True,'test_used':False,'independent_generalization_verified':False}
(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
run=OUT/'candidate';(run/'predicted_labels').mkdir(parents=True);(run/'predicted_masks').mkdir()
with np.load(B4) as archive:
 for name in names:
  pred=compose(np.asarray(Image.open(BASE/f'{name}.png')),np.asarray(Image.open(REF/f'{name}.png')),getprob(name,archive),*GRID[best])
  Image.fromarray(pred).save(run/f'predicted_labels/{name}.png');Image.fromarray(palette[pred]).save(run/f'predicted_masks/{name}_mask.png')
(OUT/'completion.json').write_text(json.dumps({'images':20,'best_percent':result['best_percent'],'candidate_generated_without_gt':True},indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['best_thresholds','best_percent','delta_pp','conditional_stage1_stage2_fixed_specimen_leave_one_out_percent']},indent=2),flush=True)
