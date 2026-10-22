"""CPU mask rescoring and conditional cluster stability, not independent validation."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
ROOT=Path('/workspace/model349_compress/generalization_20261008');OLD=ROOT.parent/'audit_20261008';GT=Path('/workspace/komap_model1/data/valid/masks')
def load(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
meta=load(ROOT/'data_exposure_audit.json');lookup={r['name']:r for r in meta['images'] if r['split']=='valid'}
spec=importlib.util.spec_from_file_location('score_old',OLD/'analyze_runs.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
keys=['cpu2_fp16_12','cpu2_fp16_10','cpu2_fp16_t5','cpu2_lite15','lite15','historical349'];scores={}
for key in keys:
 scores[key]=mod.score(OLD/'runs'/key,GT);assert set(scores[key]['images'])==set(lookup)
 print('MASKS VERIFIED',key,flush=True)
names=sorted(lookup);subsets={'all_valid20':names,'train_specimen_unseen_valid':[n for n in names if not lookup[n]['train_specimen_seen']],'train_specimen_seen_valid':[n for n in names if lookup[n]['train_specimen_seen']]}
for b in range(4):subsets[f'train_brightness_quartile_{b+1}']=[n for n in names if lookup[n]['brightness_bin_train_quartiles']==b]
results={'scope':'Frozen candidates on already-selected valid20, paired specimen bootstrap. Conditional descriptive stability; selection bias NOT corrected.','independent_generalization_verified':False,'metric':'image-weighted mean of per-image four-class IoU, empty union=0; percent/percentage points','bootstrap':{'seed':20261008,'replicates':20000,'unit':'specimen','method':'percentile 95%; sampled groups contribute all images; candidate fixed','interpretation':'Not a fresh holdout confidence interval; no correction for adaptive candidate selection.'},'class_order':load(OLD/'runs/lite15/input_manifest.json')['config']['class_order'],'subsets':{},'comparisons':{},'per_image':{}}
rng=np.random.default_rng(20261008)
references=['cpu2_fp16_t5','cpu2_lite15','lite15','historical349']
for subset,selected in subsets.items():
 if not selected:continue
 groups=sorted({lookup[n]['specimen'] for n in selected});indices=[[i for i,n in enumerate(selected) if lookup[n]['specimen']==g] for g in groups]
 groupcounts=np.asarray([len(v) for v in indices],dtype=np.float64)
 draw=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=20000)
 denom=draw@groupcounts
 summary={'images':len(selected),'specimens':len(groups),'names':selected,'means':{k:float(100*np.mean([scores[k]['images'][n]['image_miou'] for n in selected])) for k in keys},'class_means':{k:(100*np.mean([scores[k]['images'][n]['class_iou'] for n in selected],axis=0)).tolist() for k in keys}}
 results['subsets'][subset]=summary
 for candidate in keys[:2]:
  c=results['comparisons'].setdefault(candidate,{})
  for reference in references:
   deltas=np.asarray([100*(scores[candidate]['images'][n]['image_miou']-scores[reference]['images'][n]['image_miou']) for n in selected])
   classes=np.asarray([100*(np.asarray(scores[candidate]['images'][n]['class_iou'])-scores[reference]['images'][n]['class_iou']) for n in selected])
   gs=np.asarray([deltas[ix].sum() for ix in indices]);gc=np.vstack([classes[ix].sum(axis=0) for ix in indices]);boot=(draw@gs)/denom;class_boot=(draw@gc)/denom[:,None]
   leave=[{'omitted_specimen':g,'remaining_images':len(selected)-len(ix),'delta_pp':float(np.delete(deltas,ix).mean())} for g,ix in zip(groups,indices) if len(ix)<len(selected)]
   worst=int(deltas.argmin());best=int(deltas.argmax());mean=float(deltas.mean())
   row={'delta_pp':mean,'bootstrap95_ci_pp':np.quantile(boot,[.025,.975]).tolist(),'class_delta_pp':classes.mean(axis=0).tolist(),'class_bootstrap95_ci_pp':np.quantile(class_boot,[.025,.975],axis=0).T.tolist(),'improved_images':int((deltas>0).sum()),'worse_images':int((deltas<0).sum()),'tied_images':int((deltas==0).sum()),'worst_image':selected[worst],'worst_delta_pp':float(deltas[worst]),'best_image':selected[best],'best_delta_pp':float(deltas[best]),'leave_one_specimen_out':leave,'leave_one_specimen_out_range_pp':[min(x['delta_pp'] for x in leave),max(x['delta_pp'] for x in leave)] if leave else None,'largest_group_influence':max(leave,key=lambda x:abs(x['delta_pp']-mean)) if leave else None,'quality_threshold_reference_only':mean>=-.01 and float(deltas.min())>=-.10,'quality_threshold_is_independent_validation':False}
   c.setdefault(reference,{})[subset]=row
for n in names:
 results['per_image'][n]={'specimen':lookup[n]['specimen'],'train_specimen_seen':lookup[n]['train_specimen_seen'],'gray_mean':lookup[n]['gray_mean'],'train_brightness_quartile':lookup[n]['brightness_bin_train_quartiles']+1,'class_iou_percent':{k:(100*np.asarray(scores[k]['images'][n]['class_iou'])).tolist() for k in keys},'miou_percent':{k:100*scores[k]['images'][n]['image_miou'] for k in keys},'delta_vs_fp16_14_pp':{k:100*(scores[k]['images'][n]['image_miou']-scores['cpu2_fp16_t5']['images'][n]['image_miou']) for k in keys[:2]}}
results['input_hashes']={k:scores[k]['input_sha256'] for k in keys};results['calibration_hashes']={k:scores[k]['calibration_sha256'] for k in keys}
assert all(scores[k]['input_sha256']==scores['lite15']['input_sha256'] and scores[k]['calibration_sha256']==scores['lite15']['calibration_sha256'] for k in keys)
write(ROOT/'valid_conditional_stability.json',results)
for c in keys[:2]:
 for subset in ['all_valid20','train_specimen_unseen_valid','train_specimen_seen_valid']:
  v=results['comparisons'][c]['cpu2_fp16_t5'][subset];print(c,subset,{k:v[k] for k in ['delta_pp','bootstrap95_ci_pp','worst_delta_pp','leave_one_specimen_out_range_pp']},flush=True)
