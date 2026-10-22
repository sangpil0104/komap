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
import model198_fast_branch_search as post
q=WORK/'Model349_lite12_base/src/model198_fast_branch_search.py'
fn=next(n for n in ast.parse(q.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='predict');fn.name='logits_fields'
cut=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.If));fn.body=fn.body[:cut]+[ast.Return(value=ast.Name(id='logits',ctx=ast.Load()))]
postns=dict(post.__dict__);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(q),'exec'),postns);logits_fields=postns['logits_fields']
import cv2
cv2.setNumThreads(1)
GRID=list(itertools.product([0.,.25,.5,.75,1.],[-.25,0.,.125,.25,.5],[0,32,128,256]))
BASE=GRID.index((.5,.25,128));COMBO=(12,3,13,1,1,1,0,1,7)
ref=np.load(ROOT.parent/'exhaustive_review_20261008/replay_iou.npz');assert list(ref['names'])==m.NAMES
out=np.zeros((len(GRID),len(m.NAMES),4),np.float64);t=time.monotonic()
input_hash={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [p,q,source,ROOT/'spatial_protocol.md',ROOT/'spatial_search.py']}
def score(pred,gt):
 return [np.count_nonzero((pred==c)&(gt==c))/u if (u:=np.count_nonzero((pred==c)|(gt==c))) else 0. for c in range(4)]
for j,name in enumerate(m.NAMES):
 z=m.load(name);item,prob,s=fields(z,COMBO);logits=logits_fields(item,prob,s,z['weights'],True)
 gauss=post.gaussian_filter(logits[1],sigma=.50,mode='reflect');ps=post.apply_bilateral(logits[1],z['weights']);eut=post.apply_bilateral(logits[3],z['weights'])
 for k,(g,e,thr) in enumerate(GRID):
  refined=logits.copy();refined[1]=g*gauss+(1-g)*ps;refined[3]=(1-e)*logits[3]+e*eut
  pred=post.cleanup_primary(refined.argmax(0).astype(np.uint8),refined,thr)
  if (pred==1).mean()<=.01:
   mask=pred==1;alt=refined.copy();alt[1]=-np.inf;pred[mask]=alt[:,mask].argmax(0).astype(np.uint8)
  out[k,j]=score(pred,z['gt'])
  if k==BASE:
   assert np.array_equal(out[k,j],ref['best14'][j]),(name,'baseline IoU changed')
   old=cv2.imread(str(ROOT.parent/'exhaustive_review_20261008/cached_masks/best14'/(name+'.png')),cv2.IMREAD_UNCHANGED);assert np.array_equal(pred,old)
 print(j+1,name,round(time.monotonic()-t,1),flush=True)
 np.savez(ROOT/'spatial_partial.npz',grid=np.array(GRID),iou=out,names=np.array(m.NAMES),images_done=j+1)
np.savez(ROOT/'spatial_grid.npz',grid=np.array(GRID),iou=out,names=np.array(m.NAMES))
best=int(out.mean((1,2)).argmax());selected=GRID[best];exact=[]
for name in m.NAMES:
 z=m.load(name);item,prob,s=fields(z,COMBO);s.update(dict(zip(['gaussian_blend','eut_edge_blend','cleanup_threshold'],selected)))
 pred=m.route.predict(item,prob,s,z['weights'],exact=True);exact.append(score(pred,z['gt']))
exact=np.array(exact);assert np.array_equal(exact,out[best]),'selected original predict differs'
groups=np.array([n.split('_')[0] for n in m.NAMES]);uniq=np.unique(groups);rng=np.random.default_rng(814)
delta=(exact-out[BASE]).mean(1)*100;boot=[]
for _ in range(20000):
 ids=np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(uniq,len(uniq),replace=True)]);boot.append(delta[ids].mean())
loso=[]
for g in uniq:
 train=groups!=g;sel=int(out[:,train].mean((1,2)).argmax());loso.extend((out[sel,~train].mean(1)-out[BASE,~train].mean(1))*100)
result={'scope':'reused valid cached spatial exploration, NOT whole raw inference or independent generalization','base_cached_percent':float(out[BASE].mean()*100),'selected_grid_index':best,'selected_params':dict(zip(['gaussian_blend','eut_edge_blend','cleanup_threshold'],selected)),'selected_cached_percent':float(exact.mean()*100),'selected_original_predict_exact':True,'class_order':['Al3Ni','PrimarySi','Al','EutecticSi'],'class_delta_pp':((exact-out[BASE]).mean(0)*100).tolist(),'mean_delta_pp':float(delta.mean()),'worst_delta_pp':float(delta.min()),'worst_image':m.NAMES[int(delta.argmin())],'conditional_group_bootstrap_95_pp':np.percentile(boot,[2.5,97.5]).tolist(),'fixed_grid_loso_selection_delta_pp':float(np.mean(loso)),'candidate_count':len(GRID),'per_image':[{'image':n,'base_percent':float(out[BASE,i].mean()*100),'candidate_percent':float(exact[i].mean()*100),'delta_pp':float(delta[i])} for i,n in enumerate(m.NAMES)],'input_hashes':input_hash,'elapsed_seconds':time.monotonic()-t,'independent_generalization_verified':False}
assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in input_hash.items())
(ROOT/'spatial_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
