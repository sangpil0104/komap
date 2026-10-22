"""Train70-only grouped-OOF generalized probability means of E; no valid/test reads, CPU only."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import json,hashlib,itertools
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent
DATA=Path('/workspace/komap_model1')
GRID=[1.,0.,.5,2.,-1.] # arithmetic identity, geometric, root, quadratic, harmonic.
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
 assert not (ROOT/'both_result.json').exists()
 records=[];frozen={};seen=set();group_fold={}
 for fold in range(2,3):
  work=ROOT.parent/'accuracy81_HR_seed2_warmstart_fold2_20261010'
  assert json.loads((work/'completion.json').read_text())['all_stages_completed']
  paths=[DATA/f'outputs/model329_fullstats_width4000_uniform_group_oof/fold{fold}_oof/probabilities.npz',ROOT.parent/'accuracy81_HR_warmstart_fold2_20261009/original_eval/valid_probabilities.npz',DATA/f'outputs/model392_widthnorm/seed2/fold{fold}_oof/valid_probabilities.npz',work/'original_eval/valid_probabilities.npz']
  for p in paths:frozen[str(p)]=sha(p)
  archives=[np.load(p,allow_pickle=False) for p in paths]
  try:
   assert all(set(z.files)==set(archives[0].files) for z in archives)
   for name in sorted(archives[0].files):
    assert name not in seen;seen.add(name);group=name.split('_')[0]
    assert group_fold.get(group,fold)==fold;group_fold[group]=fold
    parts=[z[name].astype(np.float32) for z in archives]
    mask=DATA/f'data/train/masks/{name}_mask.png';frozen[str(mask)]=sha(mask)
    rgb=np.asarray(Image.open(mask).convert('RGB'));truth=np.full(rgb.shape[:2],255,np.uint8)
    for c,color in enumerate(PALETTE):truth[np.all(rgb==color,axis=-1)]=c
    assert np.all(truth<4) and all(p.shape==(4,*truth.shape) and np.isfinite(p).all() for p in parts)
    old=.5*parts[0]+.25*parts[1]+.25*parts[2]
    new=.5*parts[0]+.25*parts[1]+.25*parts[3]
    records.append({'name':name,'fold':fold,'E_before':score(old.argmax(0),truth),'E_after':score(new.argmax(0),truth),'HR_before':score(parts[2].argmax(0),truth),'HR_after':score(parts[3].argmax(0),truth)})
  finally:
   for z in archives:z.close()
 assert len(records)==18
 means={key:float(100*np.mean([r[key] for r in records])) for key in ['E_before','E_after','HR_before','HR_after']}
 assert all(sha(Path(p))==v for p,v in frozen.items())
 fold_delta=[float(100*(np.mean([r['E_after'] for r in records if r['fold']==f])-np.mean([r['E_before'] for r in records if r['fold']==f]))) for f in range(2,3)]
 result={'means_percent':means,'E_delta_pp':means['E_after']-means['E_before'],'HR_delta_pp':means['HR_after']-means['HR_before'],'E_fold_delta_pp':fold_delta,'records':records,'input_sha256':frozen,'scope':'train70 held fold2; seed1 warmstart E vs both warmstart E fixed E substitution; no official valid/test; diagnostic selection bias'}
 (ROOT/'both_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['records','input_sha256']}))
if __name__=='__main__':main()
