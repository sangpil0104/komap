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
from guided_filter import guided,refine,synthetic_smoke
from PIL import Image
synthetic_smoke()
GRID=[(1,.0001,0.)]+list(itertools.product([1,2,4],[.0001,.001,.01],[.25,.5,1.]))
COMBO=(12,3,13,1,1,1,0,1,7)
ref=np.load(ROOT.parent/'exhaustive_review_20261008/replay_iou.npz');assert list(ref['names'])==m.NAMES
manifest=json.loads((ROOT.parent/'exh14_base_search_20261008/exh14_valid20/input_manifest.json').read_text());image_sha={x['name']:x['sha256'] for x in manifest['images']}
out=np.zeros((len(GRID),20,4));t=time.monotonic()
input_hash={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [p,q,source,ROOT/'guided_protocol.md',ROOT/'guided_filter.py',Path(__file__).resolve()]}
def score(pred,gt):
 return [np.count_nonzero((pred==c)&(gt==c))/u if (u:=np.count_nonzero((pred==c)|(gt==c))) else 0. for c in range(4)]
def original_fields(z):
 item,prob,s=fields(z,COMBO);logits=logits_fields(item,prob,s,z['weights'],True)
 result=logits.copy();result[1]=s['gaussian_blend']*post.gaussian_filter(logits[1],sigma=.5,mode='reflect')+(1-s['gaussian_blend'])*post.apply_bilateral(logits[1],z['weights']);result[3]=(1-s['eut_edge_blend'])*logits[3]+s['eut_edge_blend']*post.apply_bilateral(logits[3],z['weights'])
 return result,s

def finish(refined,s):
 pred=post.cleanup_primary(refined.argmax(0).astype(np.uint8),refined,s['cleanup_threshold'])
 if (pred==1).mean()<=.01:
  mask=pred==1;alt=refined.copy();alt[1]=-np.inf;pred[mask]=alt[:,mask].argmax(0).astype(np.uint8)
 return pred

def load_guide(name,shape):
 p=SCRATCH/'gt/valid/images'/(name+'_image.png');h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==image_sha[name];input_hash[str(p)]=h
 a=np.array(Image.open(p).convert('L'),dtype=np.float32)/255.;assert a.shape==shape and np.isfinite(a).all();return a
for i,name in enumerate(m.NAMES):
 z=m.load(name);original,s=original_fields(z);guide=load_guide(name,z['gt'].shape)
 input_hash[str(m.HIGH/(name+'.npz'))]=hashlib.sha256((m.HIGH/(name+'.npz')).read_bytes()).hexdigest()
 filtered={}
 for j,(radius,eps,blend) in enumerate(GRID):
  if not blend:refined=original
  else:
   if (radius,eps) not in filtered:filtered[radius,eps]=np.stack([guided(guide,f,radius,eps) for f in original])
   refined=(1-blend)*original+blend*filtered[radius,eps]
  pred=finish(refined,s);out[j,i]=score(pred,z['gt'])
  if j==0:
   assert np.array_equal(out[j,i],ref['best14'][i]),(name,'base IoU differs')
   prior=cv2.imread(str(ROOT.parent/'exhaustive_review_20261008/cached_masks/best14'/(name+'.png')),cv2.IMREAD_UNCHANGED);assert np.array_equal(pred,prior)
 print(i+1,name,round(time.monotonic()-t,1),flush=True)
best=int(out.mean((1,2)).argmax());selected=GRID[best]
for i,name in enumerate(m.NAMES):
 z=m.load(name);original,s=original_fields(z);guide=load_guide(name,z['gt'].shape)
 assert np.array_equal(score(finish(refine(original,guide,*selected),s),z['gt']),out[best,i])
groups=np.array([n.split('_')[0] for n in m.NAMES]);uniq=np.unique(groups);rng=np.random.default_rng(819);delta=(out[best]-out[0]).mean(1)*100;boot=[];loso=[]
for _ in range(20000):
 ids=np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(uniq,len(uniq),replace=True)]);boot.append(delta[ids].mean())
for g in uniq:
 train=groups!=g;sel=int(out[:,train].mean((1,2)).argmax());loso.extend((out[sel,~train].mean(1)-out[0,~train].mean(1))*100)
result={'scope':'reused valid cached guided logit exploration; NOT raw inference or independent generalization','candidate_count':28,'base_cached_percent':float(out[0].mean()*100),'selected_cached_percent':float(out[best].mean()*100),'selected_radius':selected[0],'selected_epsilon':selected[1],'selected_blend':selected[2],'mean_delta_pp':float(delta.mean()),'worst_delta_pp':float(delta.min()),'worst_image':m.NAMES[int(delta.argmin())],'class_delta_pp':((out[best]-out[0]).mean(0)*100).tolist(),'all_candidates':[{'radius':r,'epsilon':e,'blend':b,'miou_percent':float(out[j].mean()*100)} for j,(r,e,b) in enumerate(GRID)],'per_image':[{'image':name,'base_percent':float(out[0,i].mean()*100),'candidate_percent':float(out[best,i].mean()*100),'delta_pp':float(delta[i])} for i,name in enumerate(m.NAMES)],'conditional_group_bootstrap_95_pp':np.percentile(boot,[2.5,97.5]).tolist(),'fixed_grid_loso_selection_delta_pp':float(np.mean(loso)),'input_hashes':input_hash,'elapsed_seconds':time.monotonic()-t,'selected_exact_replay':True,'independent_generalization_verified':False}
assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in input_hash.items())
np.savez(ROOT/'guided_grid.npz',grid=np.array(GRID),iou=out,names=np.array(m.NAMES))
(ROOT/'guided_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['per_image','all_candidates','input_hashes']},ensure_ascii=False,indent=2))
