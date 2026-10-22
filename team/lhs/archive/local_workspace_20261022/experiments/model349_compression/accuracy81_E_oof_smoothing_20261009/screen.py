"""Train70-only grouped-OOF spatial probability smoothing of E; no valid/test reads, CPU only."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import json,hashlib,itertools
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
ROOT=Path(__file__).resolve().parent
DATA=Path('/workspace/komap_model1')
GRID=[(0.,0.)]+[(sigma,alpha) for sigma in [1.,2.] for alpha in [.1,.25,.5]] # sigma in native pixels, mixing alpha.
PALETTE=np.array([(153,127,76),(25,76,153),(76,178,76),(204,204,204)],dtype=np.uint8)
BASE=0
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def score(pred,truth):
 vals=[]
 for c in range(4):
  a=pred==c;b=truth==c;u=np.count_nonzero(a|b);vals.append(float(np.count_nonzero(a&b)/u) if u else 0.)
 return vals
def main():
 assert not (ROOT/'result.json').exists() and not (ROOT/'failure.json').exists()
 names=[];folds=[];scores=[];frozen={};group_fold={}
 for fold in range(1,5):
  paths=[DATA/f'outputs/model329_fullstats_width4000_uniform_group_oof/fold{fold}_oof/probabilities.npz']
  paths += [DATA/f'outputs/model392_widthnorm/seed{s}/fold{fold}_oof/valid_probabilities.npz' for s in [1,2]]
  for p in paths:frozen[str(p)]=sha(p)
  archives=[np.load(p,allow_pickle=False) for p in paths]
  try:
   assert set(archives[0].files)==set(archives[1].files)==set(archives[2].files)
   for name in sorted(archives[0].files):
    assert name not in names
    group=name.split('_')[0];assert group_fold.get(group,fold)==fold;group_fold[group]=fold
    parts=[z[name].astype(np.float32) for z in archives]
    prob=.5*parts[0]+.25*parts[1]+.25*parts[2]
    mask=DATA/f'data/train/masks/{name}_mask.png';frozen[str(mask)]=sha(mask)
    rgb=np.asarray(Image.open(mask).convert('RGB'));truth=np.full(rgb.shape[:2],255,dtype=np.uint8)
    for c,color in enumerate(PALETTE):truth[np.all(rgb==color,axis=-1)]=c
    assert np.all(truth<4) and prob.shape==(4,*truth.shape) and np.isfinite(prob).all()
    values=[]
    blurred={sigma:gaussian_filter(prob,sigma=(0.,sigma,sigma),mode="reflect") for sigma in [1.,2.]}
    for sigma,alpha in GRID:
     adjusted=prob if alpha==0 else (1-alpha)*prob+alpha*blurred[sigma]
     pred=adjusted.argmax(0).astype(np.uint8)
     values.append(score(pred,truth))
    names.append(name);folds.append(fold);scores.append(values)
    print(len(names),'/70',name,flush=True)
  finally:
   for z in archives:z.close()
 assert len(names)==70 and len(group_fold)==59
 scores=np.asarray(scores,dtype=np.float64).transpose(1,0,2);folds=np.asarray(folds)
 baseline=float(scores[BASE].mean()*100)
 assert abs(baseline-79.9858620080774)<1e-9,('Original E OOF parity failed',baseline)
 cv=np.zeros((70,4));choices=[]
 for fold in range(1,5):
  train=folds!=fold;test=folds==fold
  # Prefer identity when tied, then smallest total shift.
  means=scores[:,train].mean(axis=(1,2));best=max(range(len(GRID)),key=lambda k:(means[k],-sum(abs(v) for v in GRID[k])))
  cv[test]=scores[best,test]
  delta=float((cv[test].mean()-scores[BASE,test].mean())*100)
  choices.append({'excluded_fold':fold,'chosen_sigma_alpha':GRID[best],'delta_pp':delta})
 cv_percent=float(cv.mean()*100);delta=cv_percent-baseline
 checks={'nested_fold_gain_at_least_0_10pp':delta>=.1,'positive_in_at_least_3_folds':sum(r['delta_pp']>0 for r in choices)>=3,'worst_fold_at_least_minus_0_10pp':min(r['delta_pp'] for r in choices)>=-.1}
 means=scores.mean(axis=(1,2));best=max(range(len(GRID)),key=lambda k:(means[k],-sum(abs(v) for v in GRID[k])))
 assert all(sha(Path(p))==h for p,h in frozen.items())
 result={'scope':'train70 E grouped OOF; smoothing selected on other 3 folds for each held fold','baseline_percent':baseline,'nested_fold_percent':cv_percent,'nested_fold_delta_pp':delta,'fold_choices':choices,'checks':checks,'screen_pass':all(checks.values()),'all_train_selected_sigma_alpha':GRID[best],'all_train_fit_percent':float(means[best]*100),'grid':GRID,'names':names,'folds':folds.tolist(),'class_iou':scores.tolist(),'input_sha256':frozen,'valid_read':False,'test_read':False,'gpu_used':False,'parent_archives_unchanged':True,'not_official_valid_score':True}
 (ROOT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 (ROOT/'completion.json').write_text(json.dumps({'images':70,'screen_pass':result['screen_pass'],'valid_read':False,'test_read':False,'gpu_used':False},indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ['baseline_percent','nested_fold_percent','nested_fold_delta_pp','screen_pass','all_train_selected_sigma_alpha']},indent=2),flush=True)
if __name__=='__main__':
 import traceback,time
 try:main()
 except BaseException:
  (ROOT/'failure.json').write_text(json.dumps({'traceback':traceback.format_exc(),'time':time.time()},indent=2)+'\n');raise
