"""Dataset/specimen identity and deployed-fold exposure audit; CPU only."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
import hashlib,json,re
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(2)
from PIL import Image
ROOT=Path('/workspace/model349_compress/generalization_20261008');OLD=ROOT.parent/'audit_20261008';DATA=Path('/workspace/komap_model1/data');OUT=DATA.parent/'outputs'
def load(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def group(name):
 m=re.match(r'(A\d+)_',name);assert m,name
 return m.group(1)
def state(p):
 obj=torch.load(p,map_location='cpu',weights_only=False)
 for key in ['model','state_dict']:
  if isinstance(obj,dict) and key in obj and isinstance(obj[key],dict):return obj[key]
 return obj

def equivalent_to_frozen_fp16(source,frozen):
 a=state(source);b=state(frozen)
 if set(a)!=set(b):return {'equivalent':False,'reason':'state key mismatch'}
 mismatched=[]
 for k in a:
  x,y=a[k],b[k]
  if isinstance(x,torch.Tensor) and isinstance(y,torch.Tensor):
   same=x.shape==y.shape and torch.equal(x.to(y.dtype),y)
  else:same=x==y
  if not same:mismatched.append(k)
 return {'equivalent':not mismatched,'mismatched_keys':len(mismatched),'comparison':'CPU source tensors cast to frozen stored dtype; integer exact'}
rows=[]
for split,n in [('train',70),('valid',20),('test',10)]:
 paths=sorted((DATA/split/'images').glob('*.png'));assert len(paths)==n
 for p in paths:
  name=p.stem.removesuffix('_image');rgb=np.asarray(Image.open(p).convert('RGB'));gray=np.asarray(Image.open(p).convert('L'))
  pix=hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest()
  masks=[DATA/split/'masks'/f'{name}_mask.png',DATA/split/'masks'/f'{name}.png'];mask=next((x for x in masks if x.is_file()),None)
  assert (mask is not None)==(split!='test')
  rows.append({'name':name,'split':split,'specimen':group(name),'path':str(p),'image_sha256':sha(p),'decoded_rgb_sha256':pix,'shape':list(rgb.shape),'gray_mean':float(gray.mean()),'gray_std':float(gray.std()),'mask_path':str(mask) if mask else None,'mask_sha256':sha(mask) if mask else None})
sets={s:{v['specimen'] for v in rows if v['split']==s} for s in ['train','valid','test']}
quantiles=np.quantile([v['gray_mean'] for v in rows if v['split']=='train'],[.25,.5,.75]).tolist()
for v in rows:
 v['train_specimen_seen']=v['specimen'] in sets['train'];v['valid_specimen_seen']=v['specimen'] in sets['valid'];v['brightness_bin_train_quartiles']=int(np.searchsorted(quantiles,v['gray_mean'],side='right'))
overlap={f'{a}_{b}':sorted(sets[a]&sets[b]) for a,b in [('train','valid'),('train','test'),('valid','test')]}
duplicates={}
for a,b in [('train','valid'),('train','test'),('valid','test')]:
 duplicates[f'{a}_{b}']=[{'a':x['name'],'b':y['name']} for x in rows if x['split']==a for y in rows if y['split']==b and x['decoded_rgb_sha256']==y['decoded_rgb_sha256']]
manifest=load(OUT/'model303_boundary_oof_8ep/progress.json');cv=DATA.parent/'data_model322_group_cv';folds=[]
source_reports={v['path']:v for v in load(OLD/'bundles/fp16_15_batched/fp16_storage_report.json')}
for record in manifest['completed_folds']:
 fold=int(record['fold']);train_paths=sorted((cv/f'train_fold{fold}'/'images').glob('*.png'));held_paths=sorted((cv/f'oof_fold{fold}'/'images').glob('*.png'))
 t={group(p.name) for p in train_paths};h={group(p.name) for p in held_paths}
 assert t==set(record['train_groups']) and h==set(record['heldout_groups']) and not t&h and t|h==sets['train']
 assert len(train_paths)==record['train_image_count'] and len(held_paths)==record['heldout_image_count']
 for p in train_paths+held_paths:assert p.resolve()==(DATA/'train/images'/p.name).resolve()
 argdata={}
 for component,directory in [('326','model326_hybrid_fullstats_group_oof'),('329','model329_fullstats_width4000_uniform_group_oof')]:
  run=OUT/directory/f'fold{fold}';args=load(run/'args.json');assert args['train_splits']==f'train_fold{fold}' and Path(args['data']).resolve()==cv.resolve()
  weight=run/'last.pt';rel=f'weights/{directory}/fold{fold}/last.pt';actual=sha(weight);eq=equivalent_to_frozen_fp16(weight,OLD/'bundles/fp16_15_batched'/rel)
  frozenargs=load(OLD/'bundles/fp16_15_batched'/Path(rel).parent/'args.json')
  print('CHECK',component,fold,'source_sha_match',actual==source_reports[rel]['source_sha256'],'tensor_equivalent',eq['equivalent'],flush=True)
  argdata[component]={'args':args,'args_sha256':sha(run/'args.json'),'checkpoint_sha256':actual,'frozen_args':frozenargs,'frozen_original_source_sha256':source_reports[rel]['source_sha256'],'source_file_sha_matches_frozen_origin':actual==source_reports[rel]['source_sha256'],'source_state_vs_frozen_fp16':eq,'oof_score_exists':(OUT/directory/f'fold{fold}_oof/scores.json').is_file()}
 folds.append({'fold':fold,'train_groups':sorted(t),'heldout_groups':sorted(h),'train_images':len(train_paths),'heldout_images':len(held_paths),'components':argdata})
for v in rows:
 v['329_deployed_fold_specimen_training_exposure']=[f['fold'] for f in folds if v['specimen'] in f['train_groups']]
 v['329_declared_fold_exact_training_image_exposure']=v['329_deployed_fold_specimen_training_exposure'] if v['split']=='train' else []
 if v['split']=='train':assert len(v['329_declared_fold_exact_training_image_exposure'])==3
 v['326_lite12_specimen_training_exposure']=[f for f in v['329_deployed_fold_specimen_training_exposure'] if f in (1,3)]
hrargs=load(OUT/'model336_hrnet_best/args.json');assert Path(hrargs['data_root']).resolve()==DATA.resolve()
hrsource=ROOT/'reference_hrnet_train.py';hrtext=hrsource.read_text();assert 'root / "train" / "images"' in hrtext
hrsha=sha(OUT/'model336_hrnet_best/last.pt');hreq=equivalent_to_frozen_fp16(OUT/'model336_hrnet_best/last.pt',OLD/'bundles/fp16_15_batched/weights/model336_hrnet_best/last.pt')
summary={'scope':'existing 100 images only; exact frozen checkpoint/CV exposure audit','counts':{s:sum(v['split']==s for v in rows) for s in sets},'specimen_counts':{s:len(v) for s,v in sets.items()},'specimen_overlap':overlap,'decoded_image_duplicates':duplicates,'brightness_quartiles_from_train':quantiles,'valid_train_unseen_images':sum(v['split']=='valid' and not v['train_specimen_seen'] for v in rows),'test_train_valid_unseen_images':sum(v['split']=='test' and not v['train_specimen_seen'] and not v['valid_specimen_seen'] for v in rows),'train_images_unseen_by_all329_folds_under_declared_training_split':sum(v['split']=='train' and not v['329_declared_fold_exact_training_image_exposure'] for v in rows),'full_candidates_strict_oof_available':False,'exposure_interpretation':'CV membership is audited on current training output args and CV symlinks. Frozen bundle contains inference args only; source state mismatch means this training lineage and OOF caches cannot certify frozen weights. Exposure counts are conditional on the recorded original fold partition, not a certified frozen training history.', 'reason':'Current training-output OOF caches may not match the frozen checkpoints. Even under the recorded original group-CV partition, both candidates deploy all four 329 folds, so every train image is fitted by three members. Heldout-fold-only inference changes the deployed candidate. HRNet training metadata uses all train images; exact frozen lineage must be checked separately.','hrnet_evidence':{'args':hrargs,'args_sha256':sha(OUT/'model336_hrnet_best/args.json'),'source_train_py':str(hrsource),'script_evidence_scope':'Local repository reference training implementation; frozen execution lineage not independently certified','source_train_py_sha256':sha(hrsource),'checkpoint_sha256':hrsha,'frozen_original_source_sha256':source_reports['weights/model336_hrnet_best/last.pt']['source_sha256'],'source_file_sha_matches_frozen_origin':hrsha==source_reports['weights/model336_hrnet_best/last.pt']['source_sha256'],'source_state_vs_frozen_fp16':hreq,'train_dataset_expression':'root / train / images and masks'},'folds':folds,'images':rows}
summary['frozen_training_provenance_complete']=all(c['source_state_vs_frozen_fp16']['equivalent'] for f in folds for c in f['components'].values()) and hreq['equivalent']
write(ROOT/'data_exposure_audit.json',summary)
print(json.dumps({k:v for k,v in summary.items() if k not in ['images','folds','hrnet_evidence']},indent=2),flush=True)
