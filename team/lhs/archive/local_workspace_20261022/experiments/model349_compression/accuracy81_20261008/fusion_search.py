import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import ast,hashlib,importlib.util,json,sys,time,types,itertools
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
fn=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='predict')
fn.name='fields';fn.body[-1]=ast.Return(value=ast.Tuple(elts=[ast.Name(id=k,ctx=ast.Load()) for k in ['item','mixed','s']],ctx=ast.Load()))
ns=dict(m.__dict__);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(p),'exec'),ns);fields=ns['fields']
import cv2
cv2.setNumThreads(1)
GRID=list(itertools.product([.5,.7,.9],[.25,.5,.75,1.],[.25,.5,.75,1.]))
BASE=GRID.index((.7,.5,.75));COMBO=(12,3,13,1,1,1,0,1,7)
ref=np.load(ROOT.parent/'exhaustive_review_20261008/replay_iou.npz');assert list(ref['names'])==m.NAMES
out=np.zeros((len(GRID),len(m.NAMES),4),np.float64);t=time.monotonic()
input_hash={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [p,source,ROOT/'fusion_protocol.md',ROOT/'fusion_search.py']}
def score(pred,gt):
 return [np.count_nonzero((pred==c)&(gt==c))/u if (u:=np.count_nonzero((pred==c)|(gt==c))) else 0. for c in range(4)]
for j,name in enumerate(m.NAMES):
 z=m.load(name)
 for k,(g326,g329,e336) in enumerate(GRID):
  ns['BASE_SET']=dict(m.BASE_SET,al3ni_gamma=g326);ns['G329']=[g329,1.,0.,0.];ns['G336']=[0.,.25,.75,e336]
  item,prob,s=fields(z,COMBO);pred=m.route.predict(item,prob,s,z['weights'],exact=True);out[k,j]=score(pred,z['gt'])
  if k==BASE:assert np.array_equal(out[k,j],ref['best14'][j]),(name,'baseline IoU differs')
 print(j+1,name,round(time.monotonic()-t,1),flush=True)
 np.savez(ROOT/'fusion_partial.npz',grid=np.array(GRID),iou=out,names=np.array(m.NAMES),images_done=j+1)
np.savez(ROOT/'fusion_grid.npz',grid=np.array(GRID),iou=out,names=np.array(m.NAMES))
best=int(out.mean((1,2)).argmax());selected=GRID[best];exact=out[best]
groups=np.array([n.split('_')[0] for n in m.NAMES]);uniq=np.unique(groups);rng=np.random.default_rng(813)
delta=(exact-out[BASE]).mean(1)*100;boot=[]
for _ in range(20000):
 ids=np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(uniq,len(uniq),replace=True)]);boot.append(delta[ids].mean())
loso=[]
for g in uniq:
 train=groups!=g;sel=int(out[:,train].mean((1,2)).argmax());loso.extend((out[sel,~train].mean(1)-out[BASE,~train].mean(1))*100)
result={'scope':'adaptively reused valid cached exploration, NOT raw inference or independent generalization','base_cached_percent':float(out[BASE].mean()*100),'selected_grid_index':best,'selected_params':dict(zip(['al3ni_gamma326','al3ni_gamma329','eut_gamma336'],selected)),'selected_cached_percent':float(exact.mean()*100),'class_order':['Al3Ni','PrimarySi','Al','EutecticSi'],'class_delta_pp':((exact-out[BASE]).mean(0)*100).tolist(),'mean_delta_pp':float(delta.mean()),'worst_delta_pp':float(delta.min()),'worst_image':m.NAMES[int(delta.argmin())],'conditional_group_bootstrap_95_pp':np.percentile(boot,[2.5,97.5]).tolist(),'fixed_grid_loso_selection_delta_pp':float(np.mean(loso)),'candidate_count':len(GRID),'per_image':[{'image':n,'base_percent':float(out[BASE,i].mean()*100),'candidate_percent':float(exact[i].mean()*100),'delta_pp':float(delta[i])} for i,n in enumerate(m.NAMES)],'input_hashes':input_hash,'elapsed_seconds':time.monotonic()-t,'independent_generalization_verified':False}
assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in input_hash.items())
(ROOT/'fusion_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
