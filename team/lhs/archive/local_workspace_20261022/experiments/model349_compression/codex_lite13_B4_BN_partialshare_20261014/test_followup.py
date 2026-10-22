import time,json,sys,subprocess,hashlib,pickle,importlib.util,fcntl
from pathlib import Path
import numpy as np,torch,yaml
from PIL import Image
from scipy import ndimage as ndi
D=Path(__file__).resolve().parent;K=Path('/home/sangpil/komap');L=K/'lite14_N3_R326plain3456_HRtrain70seed5_20261013';P13=K/'lite13_R322share326_20261014'
lock=(D/'test_followup.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (D/'test_completion.json').exists():sys.exit(0)
while not (D/'completion.json').exists():time.sleep(10)
result=json.loads((D/'results.json').read_text());passed=[k for k,v in result['cases'].items() if v['pass_mean_local']]
if not passed:
 (D/'test_completion.json').write_text(json.dumps({'skipped':True,'reason':'No shared-encoder candidate passed fixed mean/local gate; no test inference duplicated','test_GT_used':False}));sys.exit(0)
while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(10)
import run as bn
shared=bn.s
shared.torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
frozen=json.loads((D/'source_sha.json').read_text());assert all(shared.sha(p)==h for p,h in frozen.items())
case_groups={'BN_R':('BN_R',None),'BN_E':(None,'BN_E'),'BN_both':('BN_R','BN_E'),'BN_ALL':('BN_ALL','BN_ALL')}
for k in (4,8,16):case_groups.update({f'prefix{k}_R':(f'prefix{k}_R',None),f'prefix{k}_E':(None,f'prefix{k}_E'),f'prefix{k}_both':(f'prefix{k}_R',f'prefix{k}_E')})
needed=set(g for k in passed for g in case_groups[k] if g);images=sorted((K/'data/test/images').glob('*_image.png'));assert len(images)==10;inputs={str(p):shared.sha(p) for p in images}
specs={name:(mode,tags) for name,mode,tags in json.loads((D/'planned_jobs.json').read_text())}
for name in sorted(needed):
 mode,tags=specs[name];members=bn.load_members();bn.apply(members,tags,mode);o=D/'test_probabilities'/name;o.mkdir(parents=True,exist_ok=True)
 for p in images:
  n=p.name.removesuffix('_image.png');roles=[g for g in ('R','E') if any(t.startswith(g) for t in tags)]
  if all((o/f'{n}_{g}.npy').exists() for g in roles):continue
  for role in roles:
   mod=shared.rmod if role=='R' else shared.emod;selected=[members[tag] for tag in tags if tag.startswith(role)]
   value=mod.predict_probs(selected,shared.rmod.load_gray(p),tta=role=='E',device='cuda').cpu().numpy().astype(np.float32);np.save(o/f'{n}_{role}.npy',value)
  print('TEST',name,n,flush=True)
 del members;torch.cuda.empty_cache()
X=L/'parent_package/Model349_exh14';sys.path[:0]=[str(X),str(X/'src')]
import infer_lite
sp=importlib.util.spec_from_file_location('fu',L/'parent_package/infer_fusion.py');fu=importlib.util.module_from_spec(sp);sp.loader.exec_module(fu);ref=fu.Residual().eval();ref.load_state_dict(torch.load(L/'assets/refiner/epoch8.pt',map_location='cpu',weights_only=False)['state_dict']);cfg=yaml.safe_load((P13/'selected_R_lite13.yaml').read_text());rt=lambda p:p.astype(np.float16).astype(np.float32);rroot=K/'l18h/R_test/probabilities';oldhr=K/'codex_lite16_N3_L2_test_20261012/probabilities';rows={k:[] for k in passed}
def final(gray,r,e):
 with torch.no_grad():v,_=ref(torch.from_numpy(np.log(np.maximum(e,1e-7))[None]),gray)
 ex=fu.labels(e);fl=fu.labels(v.numpy()[0]);return np.where(fl==2,2,np.where(np.isin(ex,[0,2]),ex,r)).astype(np.uint8)
for p in images:
 n=p.name.removesuffix('_image.png');gray=torch.from_numpy(fu.prepare_gray(np.asarray(Image.open(p).convert('L'),np.float32)/255)[None,None]);src={c:np.load(rroot/f'{c}/{n}.npy').astype(np.float32) for c in ('269','329','347')};src['326']=rt(np.load(K/f'codex_R326_plain_EMA_test10_20261013/probabilities/plain3456/{n}.npy'));item=pickle.loads((rroot/f'base/{n}.pkl').read_bytes());e=np.load(L/f'fresh_test/expert/{n}.npy').astype(np.float32);rr=infer_lite.predict_fixed(item,src,cfg);base=final(gray,rr,e);assert np.array_equal(base,np.asarray(Image.open(P13/f'e2e_test/fusion/predicted_labels/{n}.png'))),n
 parts={tag:np.load(oldhr/f'{tag}/{n}.npy').astype(np.float32)[[1,2,0,3]] for tag in ('hr_s1_f2','hr_s1_f4','hr_s2_f1','hr_s2_f4')};eold=np.load(K/f'codex_lite16_b4_local_20261012/fresh/test/expert/b4_13/{n}.npy').astype(np.float32);bold=rt((eold-.25*rt((parts['hr_s1_f2']+parts['hr_s1_f4'])/2)-.25*rt((parts['hr_s2_f1']+parts['hr_s2_f4'])/2))/.5)
 for tag in passed:
  rg,eg=case_groups[tag];newr=rr;newe=e
  if rg:newr=infer_lite.predict_fixed(item,{**src,'329':rt(np.load(D/f'test_probabilities/{rg}/{n}_R.npy'))[[1,2,0,3]]},cfg)
  if eg:newe=e+.5*(rt(np.load(D/f'test_probabilities/{eg}/{n}_E.npy'))[[1,2,0,3]]-bold)
  pred=final(gray,newr,newe);ch=pred!=base;out=D/'test_predictions'/tag;out.mkdir(parents=True,exist_ok=True);Image.fromarray(pred).save(out/f'{n}.png');row={'image':n,'pixels':pred.size,'changed_pixels':int(ch.sum()),'changed_percent':100*float(ch.mean())};rows[tag].append(row)
assert all(shared.sha(p)==h for p,h in inputs.items());assert all(shared.sha(p)==h for p,h in frozen.items())
(D/'test_comparison.json').write_text(json.dumps({'baseline_exact10':True,'test_GT_used':False,'input_sha_unchanged':True,'cases':{k:{'rows':rs,'changed_percent':100*sum(r['changed_pixels'] for r in rs)/sum(r['pixels'] for r in rs)} for k,rs in rows.items()}},indent=2));(D/'test_completion.json').write_text(json.dumps({'images':10,'cases':passed,'test_GT_used':False,'release_replaced':False}));print('TEST_COMPLETE',flush=True)
