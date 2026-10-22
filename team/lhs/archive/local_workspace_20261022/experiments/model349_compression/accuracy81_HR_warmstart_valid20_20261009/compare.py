"""Fixed valid20 comparison. New HR fold exports are FP16: screening scope."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent
DATA=Path('/workspace/komap_model1')
OLD=ROOT.parent/'accuracy81_stage1_fresh_20261009/runs'
PALETTE=np.array([(153,127,76),(25,76,153),(76,178,76),(204,204,204)],np.uint8)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def score(p,t):
 return [float(np.count_nonzero((p==c)&(t==c))/np.count_nonzero((p==c)|(t==c))) if np.count_nonzero((p==c)|(t==c)) else 0. for c in range(4)]
def main():
 assert not (ROOT/'comparison.json').exists()
 assert json.loads((ROOT/'completion.json').read_text())['all_stages_completed']
 paths=[ROOT/f'fold{f}_eval/valid_probabilities.npz' for f in range(1,5)]
 bpath=DATA/'outputs/model329_fullstats_width4000_uniform_group_oof/valid_ensemble/probabilities.npz'
 frozen={str(p):sha(p) for p in paths+[bpath]};zs=[np.load(p,allow_pickle=False) for p in paths];b=np.load(bpath,allow_pickle=False)
 names=sorted(zs[0].files);assert len(names)==20 and all(set(z.files)==set(names) for z in zs)
 rows=[]
 for name in names:
  src=DATA/f'outputs/model401_model397_blend_valid_once/hrnet_seed2_mean_valid/{name}.npz'
  rpath=OLD/f'release_valid/predicted_labels/{name}.png';epath=OLD/f'expert_valid/{name}.npy';gtpath=DATA/f'data/valid/masks/{name}_mask.png'
  for p in [src,rpath,epath,gtpath]:frozen[str(p)]=sha(p)
  with np.load(src,allow_pickle=False) as z:hr2=z[name].astype(np.float32)
  hr1=sum((z[name].astype(np.float32)*.25 for z in zs)).astype(np.float16).astype(np.float32)
  new=.5*b[name].astype(np.float32)+.25*hr1+.25*hr2
  r=np.array([1,2,0,3],np.uint8)[np.asarray(Image.open(rpath))]
  old=np.load(epath)[[2,0,1,3]].argmax(0).astype(np.uint8)
  enew=new.argmax(0).astype(np.uint8)
  pred=np.where(np.isin(enew,[0,1]),enew,r).astype(np.uint8);base=np.where(np.isin(old,[0,1]),old,r).astype(np.uint8)
  rgb=np.asarray(Image.open(gtpath).convert('RGB'));truth=np.full(rgb.shape[:2],255,np.uint8)
  for c,col in enumerate(PALETTE):truth[np.all(rgb==col,axis=-1)]=c
  assert np.all(truth<4) and pred.shape==truth.shape
  out=ROOT/'candidate_masks';out.mkdir(exist_ok=True);Image.fromarray(PALETTE[pred]).save(out/f'{name}_mask.png')
  rows.append({'name':name,'before':score(base,truth),'after':score(pred,truth),'changed_pixels':int(np.count_nonzero(base!=pred)),'pixels':int(pred.size)})
 before=100*np.mean([x['before'] for x in rows]);after=100*np.mean([x['after'] for x in rows]);assert abs(before-80.47442360179173)<1e-9
 assert all(sha(Path(p))==v for p,v in frozen.items())
 result={'before_percent':float(before),'after_percent':float(after),'delta_pp':float(after-before),'records':rows,'input_sha256':frozen,'scope':'fixed rule/ratios; repeated valid20; new fold FP16 rounding differs from fresh FP32 fold averaging, requires fresh finalist inference'}
 (ROOT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['records','input_sha256']}))
if __name__=='__main__':main()
