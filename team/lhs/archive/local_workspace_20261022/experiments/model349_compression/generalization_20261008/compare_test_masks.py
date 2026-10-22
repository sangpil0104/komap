"""Unlabelled test output agreement; no ground truth or accuracy claim."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import json
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path('/workspace/model349_compress/generalization_20261008');BASE=ROOT.parent;OLD=BASE/'audit_20261008'
def load(p):return json.loads(p.read_text())
meta=load(ROOT/'data_exposure_audit.json');lookup={x['name']:x for x in meta['images'] if x['split']=='test'}
assert (ROOT/'test10_complete.json').is_file()
references={'lite15':BASE/'runs/lite_rep_free_bundle_v2_test10','lite14':BASE/'runs/lite14_ema_pruned_test10'}
paths={'fp16_12':ROOT/'runs/cpu2_fp16_12_test10','fp16_10':ROOT/'runs/cpu2_fp16_10_test10'}
def pair(a,b):
 ma=load(a/'input_manifest.json');mb=load(b/'input_manifest.json')
 assert {v['name']:v['sha256'] for v in ma['images']}=={v['name']:v['sha256'] for v in mb['images']}=={n:v['image_sha256'] for n,v in lookup.items()}
 assert ma['calibration_sha256']==mb['calibration_sha256'];rows=[]
 for n in sorted(lookup):
  x=np.asarray(Image.open(a/'predicted_labels'/f'{n}.png'));y=np.asarray(Image.open(b/'predicted_labels'/f'{n}.png'))
  assert x.shape==y.shape and np.isin(x,range(4)).all() and np.isin(y,range(4)).all();bad=x!=y
  transitions=np.bincount((4*x.astype(np.int64)+y.astype(np.int64)).ravel(),minlength=16).reshape(4,4)
  rows.append({'image':n,'specimen':lookup[n]['specimen'],'train_specimen_seen':lookup[n]['train_specimen_seen'],'valid_specimen_seen':lookup[n]['valid_specimen_seen'],'changed_pixels':int(bad.sum()),'total_pixels':int(x.size),'agreement':float((~bad).mean()),'transition_counts_reference_rows_candidate_columns':transitions.tolist(),'reference_class_area_fraction':(np.bincount(x.ravel(),minlength=4)/x.size).tolist(),'candidate_class_area_fraction':(np.bincount(y.ravel(),minlength=4)/y.size).tolist()})
 subsets={'all_test10':rows,'train_valid_specimen_unseen':[v for v in rows if not v['train_specimen_seen'] and not v['valid_specimen_seen']],'train_or_valid_specimen_seen':[v for v in rows if v['train_specimen_seen'] or v['valid_specimen_seen']]}
 summary={}
 for key,selected in subsets.items():
  t=sum(v['total_pixels'] for v in selected);c=sum(v['changed_pixels'] for v in selected);mat=np.sum([v['transition_counts_reference_rows_candidate_columns'] for v in selected],axis=0)
  class_dice=[]
  for i in range(4):
   denom=mat[i,:].sum()+mat[:,i].sum();class_dice.append(float(2*mat[i,i]/denom) if denom else None)
  summary[key]={'images':len(selected),'changed_pixels':c,'total_pixels':t,'pixel_weighted_agreement':1-c/t,'image_mean_agreement':float(np.mean([v['agreement'] for v in selected])),'worst_image_agreement':min(selected,key=lambda v:v['agreement'])['image'],'worst_agreement':min(v['agreement'] for v in selected),'class_mask_dice_agreement_only':class_dice,'transition_counts':mat.tolist()}
 return {'summaries':summary,'per_image':rows,'ground_truth_available':False,'accuracy_or_miou_claimed':False}
result={'scope':'Frozen 12/10 raw-image test10 predictions compared with existing models; agreement is not accuracy.','independent_generalization_verified':False,'class_order':load(paths['fp16_12']/'input_manifest.json')['config']['class_order'],'runs':{},'comparisons':{}}
for name,p in paths.items():
 result['runs'][name]=load(p/'completion.json');assert result['runs'][name]['mean_image_miou'] is None and not (p/'metrics.json').exists()
 for ref,r in references.items():result['comparisons'][name+'_vs_'+ref]=pair(r,p)
result['comparisons']['fp16_10_vs_fp16_12']=pair(paths['fp16_12'],paths['fp16_10'])
(ROOT/'test_mask_stability.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
for name,value in result['comparisons'].items():
 s=value['summaries']['all_test10'];print(name,s['changed_pixels'],s['total_pixels'],s['pixel_weighted_agreement'],flush=True)
