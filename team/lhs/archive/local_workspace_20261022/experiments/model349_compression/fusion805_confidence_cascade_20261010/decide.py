"""Image-level cascade decisions from cheap E confidence, without GT."""
from pathlib import Path
import json,hashlib,itertools,time
import numpy as np
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent;DATA=Path('/workspace/komap_model1');B4=DATA/'outputs/model329_fullstats_width4000_uniform_group_oof/valid_ensemble/probabilities.npz'
FAMILIES={1:BASE/'fusion805_compression_round2_20261010/seed1_maps',2:BASE/'fusion805_compression_20261010/fold_maps'}
CONFIGS=[{'key':f'{metric}_{threshold}','metric':metric,'threshold':threshold} for metric,thresholds in [('hr_disagreement',[.01,.03,.05,.10]),('low_confidence_fraction',[.01,.05,.10,.20]),('mean_normalized_entropy',[.10,.15,.20,.25])] for threshold in thresholds]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def labels(p):return np.array([2,0,1,3],np.uint8)[p[[2,0,1,3]].argmax(0)]
def main():
 names=json.loads((BASE/'fusion805_compression_round3_20261010/replay_completion.json').read_text())['names'];features={};frozen={str(B4):sha(B4)}
 with np.load(B4,allow_pickle=False) as archive:
  for name in names:
   hr=[]
   for seed in [1,2]:
    p=next(p for p in FAMILIES[seed].glob(f'shard*/{name}_fold4.npy'));frozen[str(p)]=sha(p);hr.append(np.load(p,allow_pickle=False).astype(np.float16).astype(np.float32)[[1,2,0,3]])
   prob=.5*archive[name].astype(np.float32)[[1,2,0,3]]+.25*hr[0]+.25*hr[1]
   features[name]={'hr_disagreement':float(np.mean(labels(hr[0])!=labels(hr[1]))),'low_confidence_fraction':float(np.mean(prob.max(0)<.8)),'mean_normalized_entropy':float(np.mean(-np.sum(prob*np.log(np.maximum(prob,1e-7)),axis=0)/np.log(4)))}
 rows=[]
 for cfg in CONFIGS:
  fallback={name:bool(features[name][cfg['metric']]>cfg['threshold']) for name in names};rows.append(dict(cfg,use_full_lite23=fallback,full_images=sum(fallback.values()),cheap_images=20-sum(fallback.values()),average_HR_forwards=2+2*sum(fallback.values())/20))
 assert all(sha(Path(p))==h for p,h in frozen.items())
 (ROOT/'decisions.json').write_text(json.dumps({'names':names,'configs':rows,'features':features,'input_sha256':frozen,'gt_read':False,'cheap':'s1_4_s2_4_R_canonical14','full':'reference23_cache','inference_rule':'run B4 and HR seed1 fold4 and seed2 fold4; if uncertain, add seed1 fold2 and seed2 fold1; refine once','weights_retained':23,'speed_measured':False},indent=2)+'\n');print([(x['key'],x['cheap_images']) for x in rows])
if __name__=='__main__':main()
