import os,json,sys,time,copy,itertools,traceback,importlib.util,hashlib,fcntl
from pathlib import Path
from multiprocessing import get_context
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
D=Path(__file__).resolve().parent;ROOT=D.parent
sp=importlib.util.spec_from_file_location('core',ROOT/'codex_lite14_top_candidates_20261013/run.py');c=importlib.util.module_from_spec(sp);sp.loader.exec_module(c)
def dump(name,value):
 p=D/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2));t.replace(p)
c.dump=lambda path,data:dump('upstream_'+path.name,data)
BASE=80.72098350464233;G=None

def init():
 global G
 import yaml,torch
 torch.set_num_threads(2);c.init();G=c.v.audit.GLOBAL
 G['cfg']=yaml.safe_load((D/'baseline/selected_R_lite13.yaml').read_text())
 for n,r in G['records'].items():
  r['r']=G['infer'].predict_fixed(r['item'],r['sources'],G['cfg']);r['base']=c.v.finish(r,r['ex'],r['fl'])
  assert np.array_equal(r['base'],np.asarray(Image.open(D/f'baseline/predicted_labels/{n}.png'))),n
  r['rawgray']=np.asarray(Image.open(c.v.audit.B/f'data/valid/images/{n}_image.png').convert('L'),np.float32)/255
  # These per-fold sources are MAIN already (probe_r_b4's TEAM_TO_MAIN conversion).
  assert all(np.max(np.abs(p.sum(0)-1))<.04 for p in r['rfolds'].values())
 assert abs(100*np.mean([c.v.audit.iou(r['base'],r['gt'])[1] for r in G['records'].values()])-BASE)<1e-9
 c.v.BASE=BASE;dump('baseline_preflight.json',{'baseline':BASE,'exact_fresh_masks':20,'input_sha_checked_by_upstream':True,'channel_order':'MAIN; R fold caches already converted','R_E_F_coefficients_frozen':True,'GT_used_only_for_valid_scoring':True})

def temp(p,T):
 q=np.maximum(p,1e-7)**(1/T);return q/q.sum(0,keepdims=True)
def smooth(p,g,alpha):
 h,w=g.shape;padp=np.pad(p,((0,0),(1,1),(1,1)),mode='edge');padg=np.pad(g,1,mode='edge');num=np.zeros_like(p);den=np.zeros_like(g)
 for dy,dx in itertools.product((-1,0,1),repeat=2):
  neigh=padg[dy+1:dy+1+h,dx+1:dx+1+w];weight=np.exp(-((neigh-g)/.05)**2/2)*np.exp(-(dy*dy+dx*dx)/2)
  num+=weight[None]*padp[:,dy+1:dy+1+h,dx+1:dx+1+w];den+=weight
 q=(1-alpha)*p+alpha*num/den[None];return q/q.sum(0,keepdims=True)
def change(p,g,method):
 if method=='control':return p
 if method.startswith('temperature_'):return temp(p,float(method.split('_')[1]))
 return smooth(p,g,float(method.split('_')[-1]))

def task(entry):
 family,folds,method=entry;tag=family+'_f'+''.join(map(str,folds))+'_'+method;dest=D/'results'/tag
 if (dest/'result.json').exists():return json.loads((dest/'result.json').read_text())
 try:
  rows={};digest=hashlib.sha256()
  for n,r in G['records'].items():
   p=sum((r['rfolds'][('329',f)] if family=='R329' else r['b4folds'][f]) for f in folds)/len(folds);p=c.v.roundtrip(p);digest.update(p.tobytes());p=change(p,r['rawgray'],method)
   if family=='R329':
    src={**r['sources'],'329':p};rr=G['infer'].predict_fixed(r['item'],src,G['cfg']);pred=c.v.finish(r,r['ex'],r['fl'],rr);count=13-3+len(folds)
   else:
    e=r['e']+.5*(p-r['b4']);pred=c.v.infer_e(r,e);count=13-2+len(folds)
   rows[n]=c.v.score(r,pred)
  result={'case':tag,'family':family,'folds':folds,'method':method,'actual_weight_count':count,'baseline':BASE,'summary':c.v.summarize(rows),'per_image':rows,'source_probability_sha256':digest.hexdigest(),'matched_control_only':method=='control','valid_development_screen':True,'fresh_single_fold_precision_confirmation_pending':True}
  dest.mkdir(parents=True,exist_ok=True);(dest/'result.json').write_text(json.dumps(result,indent=2));return result
 except Exception:return {'case':tag,'error':traceback.format_exc()}

if __name__=='__main__':
 lock=(D/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (D/'completion.json').exists():sys.exit(0)
 try:
  dump('status.json',{'phase':'preflight','pid':os.getpid()});init()
  methods=['control','temperature_1.15','temperature_1.30','temperature_1.50','edge_smooth_0.15','edge_smooth_0.30']
  specs=[('R329',fs,m) for size in (1,2) for fs in itertools.combinations((1,3,4),size) for m in methods]+[('B4ws',(f,),m) for f in (1,3) for m in methods]
  dump('preregistered_cases.json',{'cases':specs,'new_cases':40,'controls':8,'no_new_coefficient_grid':True,'not_independent_generalization':True});out=[]
  with get_context('fork').Pool(2) as pool:
   for r in pool.imap_unordered(task,specs):
    out.append(r);dump('progress.json',{'completed':len(out),'total':len(specs),'errors':sum('error' in x for x in out),'last':r['case']});print('DONE',len(out),'/',len(specs),r['case'],flush=True)
  ranked=sorted([r for r in out if 'error' not in r and not r['matched_control_only']],key=lambda r:-r['summary']['percent'])
  controls={(r['family'],tuple(r['folds'])):r for r in out if 'error' not in r and r['matched_control_only']}
  for r in ranked:r['vs_same_subset_control_pp']=r['summary']['percent']-controls[(r['family'],tuple(r['folds']))]['summary']['percent']
  dump('ranking.json',ranked);dump('controls.json',list(controls.values()));dump('completion.json',{'new_cases':len(ranked),'controls':len(controls),'errors':sum('error' in r for r in out),'CPU_only':True,'no_new_training':True,'release_replaced':False,'finished':time.time()});dump('status.json',{'phase':'complete'})
 except Exception:dump('failure.json',{'trace':traceback.format_exc()});raise
