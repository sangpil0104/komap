"""Reuse frozen probability caches; CPU-only nine-setting fusion screen."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import sys, json, pickle, hashlib
import numpy as np
from PIL import Image
ROOT=Path('/workspace/model349_compress')
HERE=ROOT/'accuracy81_candidate_oracle_20261009'
BUNDLE=ROOT/'Model349_exh14'
sys.path.insert(0,str(BUNDLE))
import infer_lite as infer
import model335_exhaustive_b4_route_search as route
PARENT=ROOT/'exh14_base_search_20261008/runs/control14_valid20'
OUTPUTS=Path('/workspace/komap_model1/outputs')
M401=OUTPUTS/'model401_model397_blend_valid_once'
B4=OUTPUTS/'model329_fullstats_width4000_uniform_group_oof/valid_ensemble/probabilities.npz'
GT=Path('/workspace/komap_model1/data/valid/masks')
GRID=[(0,0),(.25,.25),(.5,.5),(.75,.75),(1,1),(1,0),(0,1),(.5,1),(1,.5)]
OUT=HERE/'probability_fusion_runs'
assert not OUT.exists()
OUT.mkdir(parents=True)
manifest=json.loads((PARENT/'input_manifest.json').read_text())
config=manifest['config'];palette=np.asarray(config['palette'],dtype=np.uint8)
expected=json.loads((PARENT/'metrics.json').read_text())['image_miou']

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert sha(B4)=='80ebfddad10abb1bb5f1d7af8472f477fd5b6435b3c7fef0bfa51e21e1314b05'

def ious(pred,target):
 vals=[]
 for c in range(4):
  a=pred==c;b=target==c;u=np.count_nonzero(a|b)
  vals.append(float(np.count_nonzero(a&b)/u) if u else 0.)
 return vals

def loadrow(name):
 with (PARENT/'probabilities/base'/f'{name}.pkl').open('rb') as f:item=pickle.load(f)
 sources={c:np.load(PARENT/'probabilities'/c/f'{name}.npy').astype(np.float32) for c in infer.active_components(config) if c!='base'}
 return item,sources

def expert(name,archive):
 arrays=[archive[name].astype(np.float32)]
 for seed in [1,2]:
  with np.load(M401/f'hrnet_seed{seed}_mean_valid'/f'{name}.npz') as z:arrays.append(z[name].astype(np.float32))
 return (.5*arrays[0]+.25*arrays[1]+.25*arrays[2])[[1,2,0,3]]

names=[r['name'] for r in manifest['images']]
base_scores=[];expert_scores=[]
with np.load(B4) as archive:
 for name in names:
  item,sources=loadrow(name);pred=infer.predict_fixed(item,sources,config)
  saved=np.asarray(Image.open(PARENT/'predicted_labels'/f'{name}.png'))
  assert np.array_equal(pred,saved),name
  target=infer.decode_gt(GT/f'{name}_mask.png',palette)
  base_scores.append(ious(pred,target));expert_scores.append(ious(np.array([2,0,1,3],dtype=np.uint8)[expert(name,archive)[[2,0,1,3]].argmax(0)],target))
assert abs(np.mean(base_scores)-expected)<1e-12
assert abs(np.mean(expert_scores)-.7941817088506788)<1e-10, np.mean(expert_scores)
(HERE/'probability_fusion_parity.json').write_text(json.dumps({'parent_percent':float(np.mean(base_scores)*100),'model401_percent':float(np.mean(expert_scores)*100),'parent_png_exact':True,'cpu_only':True},indent=2)+'\n')
print('parity passed',np.mean(base_scores)*100,np.mean(expert_scores)*100,flush=True)
for i in range(len(GRID)):
 p=OUT/f'config{i}';(p/'predicted_labels').mkdir(parents=True);(p/'predicted_masks').mkdir()
 (p/'input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
scores=np.zeros((len(GRID),20,4))
with np.load(B4) as archive:
 for j,name in enumerate(names):
  item,sources=loadrow(name);base=item['preaux'];settings=config['route']
  mixed=route.make_mixed({'item':item,'base':base,'sources':sources},settings['base'])
  for component,key in [('329','gamma329'),('336','gamma336'),('347','gamma347')]:
   for c,g in enumerate(settings[key]):
    if g:mixed=route.replace_classes(mixed,sources[component],(c,),g)
  ex=expert(name,archive);guide=route.bilateral_weights(item['z'])
  target=infer.decode_gt(GT/f'{name}_mask.png',palette)
  for i,(g0,g2) in enumerate(GRID):
   value=route.replace_classes(mixed,ex,(0,),g0)
   value=route.replace_classes(value,ex,(2,),g2)
   pred=route.predict(item,value,settings['base'],guide,exact=True).astype(np.uint8)
   scores[i,j]=ious(pred,target)
   p=OUT/f'config{i}'
   Image.fromarray(pred).save(p/'predicted_labels'/f'{name}.png')
   Image.fromarray(palette[pred]).save(p/'predicted_masks'/f'{name}_mask.png')
  np.savez(HERE/'probability_fusion_partial.npz',scores=scores,names=names,images_done=j+1)
  print('images',j+1,'/20',flush=True)
assert np.allclose(scores[0],base_scores,rtol=0,atol=1e-12)
means=scores.mean(axis=(1,2));best=int(means.argmax());cv=[]
for j,name in enumerate(names):
 group=name.split('_')[0];train=[n.split('_')[0]!=group for n in names]
 chosen=int(scores[:,train,:].mean(axis=(1,2)).argmax());cv.append(float(scores[chosen,j].mean()))
for i in range(len(GRID)):
 metrics={'image_miou':float(means[i]),'images':[{'image':n,'class_iou':scores[i,j].tolist(),'image_miou':float(scores[i,j].mean())} for j,n in enumerate(names)]}
 (OUT/f'config{i}'/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
result={'grid':GRID,'per_setting_percent':(means*100).tolist(),'best_index':best,'best_gamma':GRID[best],'best_percent':float(means[best]*100),'parent_percent':float(means[0]*100),'delta_pp':float((means[best]-means[0])*100),'specimen_leave_one_out_percent':float(np.mean(cv)*100),'names':names,'class_iou':scores.tolist(),'cpu_only':True,'test_used':False,'independent_generalization_verified':False}
(HERE/'probability_fusion_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['best_gamma','best_percent','delta_pp','specimen_leave_one_out_percent']},indent=2),flush=True)
