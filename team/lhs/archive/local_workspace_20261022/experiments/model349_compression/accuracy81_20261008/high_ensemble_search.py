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
p=ROOT.parent/'exhaustive_20261008/exhaustive_local.py'

GRID=[('routed','arithmetic',0.)]+[(bank,kind,g) for bank in ['deployed4','all5'] for kind in ['arithmetic','geometric'] for g in [.5,1.]]
COMBO=(12,3,13,1,1,1,0,1,7)
ref=np.load(ROOT.parent/'exhaustive_review_20261008/replay_iou.npz');assert list(ref['names'])==m.NAMES
fn=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='predict')
# Only replace high probability selection; all downstream fusion remains original.
assignment=next(n for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='high' for t in n.targets))
assignment.value=ast.Call(func=ast.Name(id='choose_high',ctx=ast.Load()),args=[ast.Name(id='z',ctx=ast.Load()),ast.Name(id='src',ctx=ast.Load())],keywords=[])
ns=dict(m.__dict__)
choice=GRID[0]
def choose_high(z,src):
 bank,kind,g=choice
 if not g:return z['high'][src]
 keys=['restart51','model109','seed456','pairwise']+(['ema'] if bank=='all5' else [])
 if kind=='arithmetic':mixed=sum(z['high'][k] for k in keys)/len(keys)
 else:
  mixed=np.exp(sum(np.log(np.maximum(z['high'][k],1e-7)) for k in keys)/len(keys));mixed/=np.maximum(mixed.sum(0,keepdims=True),1e-7)
 return (1-g)*z['high'][src]+g*mixed
ns['choose_high']=choose_high
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(p),'exec'),ns);predict=ns['predict']
input_hash={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [p,source,ROOT/'high_ensemble_protocol.md',Path(__file__).resolve()]}
def score(pred,gt):
 return [np.count_nonzero((pred==c)&(gt==c))/u if (u:=np.count_nonzero((pred==c)|(gt==c))) else 0. for c in range(4)]
out=np.zeros((len(GRID),20,4));t=time.monotonic()
for i,name in enumerate(m.NAMES):
 z=m.load(name)
 input_hash[str(m.HIGH/(name+'.npz'))]=hashlib.sha256((m.HIGH/(name+'.npz')).read_bytes()).hexdigest()
 for j,choice in enumerate(GRID):
  out[j,i]=score(predict(z,COMBO),z['gt'])
  if j==0:assert np.array_equal(out[j,i],ref['best14'][i]),(name,'base replay failed')
 print(i+1,name,round(time.monotonic()-t,1),flush=True)
best=int(out.mean((1,2)).argmax());choice=GRID[best]
for i,name in enumerate(m.NAMES):
 z=m.load(name);assert np.array_equal(score(predict(z,COMBO),z['gt']),out[best,i])
groups=np.array([n.split('_')[0] for n in m.NAMES]);uniq=np.unique(groups);rng=np.random.default_rng(813)
delta=(out[best]-out[0]).mean(1)*100;boot=[];loso=[]
for _ in range(20000):
 ids=np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(uniq,len(uniq),replace=True)]);boot.append(delta[ids].mean())
for g in uniq:
 train=groups!=g;sel=int(out[:,train].mean((1,2)).argmax());loso.extend((out[sel,~train].mean(1)-out[0,~train].mean(1))*100)
result={'scope':'adaptively reused valid cached high-ensemble exploration; NOT raw re-inference or independent generalization','candidate_count':9,'base_cached_percent':float(out[0].mean()*100),'selected_cached_percent':float(out[best].mean()*100),'selected_bank':GRID[best][0],'selected_kind':GRID[best][1],'selected_gamma':GRID[best][2],'mean_delta_pp':float(delta.mean()),'worst_delta_pp':float(delta.min()),'worst_image':m.NAMES[int(delta.argmin())],'class_delta_pp':((out[best]-out[0]).mean(0)*100).tolist(),'all_candidates':[{'bank':bank,'kind':kind,'gamma':g,'miou_percent':float(out[j].mean()*100)} for j,(bank,kind,g) in enumerate(GRID)],'per_image':[{'image':name,'base_percent':float(out[0,i].mean()*100),'candidate_percent':float(out[best,i].mean()*100),'delta_pp':float(delta[i])} for i,name in enumerate(m.NAMES)],'conditional_group_bootstrap_95_pp':np.percentile(boot,[2.5,97.5]).tolist(),'fixed_grid_loso_selection_delta_pp':float(np.mean(loso)),'input_hashes':input_hash,'elapsed_seconds':time.monotonic()-t,'independent_generalization_verified':False,'selected_exact_replay':True}
assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in input_hash.items())
np.savez(ROOT/'high_ensemble_grid.npz',banks=np.array([x[0] for x in GRID]),kinds=np.array([x[1] for x in GRID]),gammas=np.array([x[2] for x in GRID]),iou=out,names=np.array(m.NAMES))
(ROOT/'high_ensemble_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['input_hashes','per_image']},ensure_ascii=False,indent=2))
