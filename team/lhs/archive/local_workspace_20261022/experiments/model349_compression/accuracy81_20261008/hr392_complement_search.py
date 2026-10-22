"""Seven fixed cached candidates; retain exact existing fusion/postprocessing."""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import ast,hashlib,importlib.util,json,sys,time,types
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[2]
SCRATCH=Path('/tmp/claude-501/-Users-hyunseok-orca-workspaces-KoMaP-gurnard/8c455a47-b7eb-4c6e-b73d-ddf6da07fc20/scratchpad')
os.environ['SLIM']=str(SCRATCH/'slim');os.environ['GT']=str(SCRATCH/'gt/valid/masks')
source=WORK/'Model349_reproduction_code/experiments/historical_scripts/highres1024_search.py'
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MATRIX' for t in n.targets))
stub=types.ModuleType('highres1024_search');stub.MATRIX=np.array(ast.literal_eval(node.value.args[0]),dtype=np.float32);sys.modules['highres1024_search']=stub
os.chdir(WORK/'Model349_lite12_base')
p=ROOT.parent/'exhaustive_20261008/exhaustive_local.py';spec=importlib.util.spec_from_file_location('cached',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
import cv2
cv2.setNumThreads(1)
GRID=[('all',0.)]+[(mode,g) for mode in ['all','eutectic'] for g in [.25,.5,1.]]
COMBO=(12,3,13,1,1,1,0,1,7)
ref=np.load(ROOT.parent/'exhaustive_review_20261008/replay_iou.npz');assert list(ref['names'])==m.NAMES
manifest=json.loads((ROOT/'hr392_inputs/inference_manifest.json').read_text());assert sorted(manifest['image_ids'])==sorted(m.NAMES)
receipt=json.loads((ROOT/'hr392_collection.json').read_text())
input_hash={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [p,source,ROOT/'hr392_complement_protocol.md',Path(__file__).resolve(),ROOT/'hr392_inputs/inference_manifest.json']}
for row in receipt['maps']:
 f=ROOT/'hr392_inputs'/row['path'];assert hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256'];input_hash[str(f)]=row['sha256']
def score(pred,gt):
 return [np.count_nonzero((pred==c)&(gt==c))/u if (u:=np.count_nonzero((pred==c)|(gt==c))) else 0. for c in range(4)]
def new_source(name,shape):
 maps=[]
 for seed in [1,2]:
  with np.load(ROOT/f'hr392_inputs/hrnet_seed{seed}_mean_valid/{name}.npz') as z:
   assert z.files==[name]
   a=z[name].astype(np.float32);assert a.shape==shape and np.isfinite(a).all() and a.min()>=0 and a.max()<=1
   assert np.max(np.abs(a.sum(0)-1))<.002
   maps.append(a[[1,2,0,3]])
 return (maps[0]+maps[1])*.5
out=np.zeros((len(GRID),20,4));t=time.monotonic()
for i,name in enumerate(m.NAMES):
 z=m.load(name);original=z['336'];new=new_source(name,original.shape)
 for j,(mode,g) in enumerate(GRID):
  if mode=='all':z['336']=(1-g)*original+g*new
  else:
   z['336']=original.copy();z['336'][3]=(1-g)*original[3]+g*new[3]
  out[j,i]=score(m.predict(z,COMBO),z['gt'])
  if j==0:assert np.array_equal(out[j,i],ref['best14'][i]),(name,'base replay failed')
 z['336']=original
 print(i+1,name,round(time.monotonic()-t,1),flush=True)
best=int(out.mean((1,2)).argmax())
# Re-execute selected whole existing predictor without any cached scores.
for i,name in enumerate(m.NAMES):
 z=m.load(name);old=z['336'];new=new_source(name,old.shape);mode,g=GRID[best]
 if mode=='all':z['336']=(1-g)*old+g*new
 else:z['336']=old.copy();z['336'][3]=(1-g)*old[3]+g*new[3]
 assert np.array_equal(score(m.predict(z,COMBO),z['gt']),out[best,i])
 z['336']=old
groups=np.array([n.split('_')[0] for n in m.NAMES]);uniq=np.unique(groups);rng=np.random.default_rng(813)
delta=(out[best]-out[0]).mean(1)*100;boot=[];loso=[]
for _ in range(20000):
 ids=np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(uniq,len(uniq),replace=True)]);boot.append(delta[ids].mean())
for g in uniq:
 train=groups!=g;sel=int(out[:,train].mean((1,2)).argmax());loso.extend((out[sel,~train].mean(1)-out[0,~train].mean(1))*100)
result={'scope':'adaptively reused valid cached complement exploration; NOT whole candidate raw re-inference or independent generalization','candidate_count':7,'base_cached_percent':float(out[0].mean()*100),'selected_cached_percent':float(out[best].mean()*100),'selected_mode':GRID[best][0],'selected_gamma':GRID[best][1],'mean_delta_pp':float(delta.mean()),'worst_delta_pp':float(delta.min()),'worst_image':m.NAMES[int(delta.argmin())],'class_delta_pp':((out[best]-out[0]).mean(0)*100).tolist(),'all_candidates':[{'mode':mode,'gamma':g,'miou_percent':float(out[j].mean()*100)} for j,(mode,g) in enumerate(GRID)],'per_image':[{'image':name,'base_percent':float(out[0,i].mean()*100),'candidate_percent':float(out[best,i].mean()*100),'delta_pp':float(delta[i])} for i,name in enumerate(m.NAMES)],'conditional_group_bootstrap_95_pp':np.percentile(boot,[2.5,97.5]).tolist(),'fixed_grid_loso_selection_delta_pp':float(np.mean(loso)),'input_hashes':input_hash,'elapsed_seconds':time.monotonic()-t,'independent_generalization_verified':False,'selected_exact_replay':True}
assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in input_hash.items())
np.savez(ROOT/'hr392_complement_grid.npz',modes=np.array([x[0] for x in GRID]),gammas=np.array([x[1] for x in GRID]),iou=out,names=np.array(m.NAMES))
(ROOT/'hr392_complement_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['input_hashes','per_image']},ensure_ascii=False,indent=2))
