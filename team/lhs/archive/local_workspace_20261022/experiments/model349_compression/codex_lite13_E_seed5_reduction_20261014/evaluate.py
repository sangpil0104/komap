import importlib.util,json,itertools,sys,time
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('base',D/'baseline_snapshot.py'); b=importlib.util.module_from_spec(sp);sp.loader.exec_module(b)
def dump(n,v): (D/n).write_text(json.dumps(v,indent=2))
def load(root,n):return np.load(root/f'{n}.npy').astype(np.float32)
assert abs(b.BASE-80.72098350464233)<1e-8,b.BASE
dump('baseline_preflight.json',{'baseline':b.BASE,'exact_masks':20,'config_frozen':True,'test_gt_used':False})
if sys.argv[1]=='preflight':sys.exit(0)
results={}
def record(tag,preds,count):
 r=b.score(preds);r.update(score=b.BASE+r['delta'],actual_weight_count=count);results[tag]=r
 print(tag,r['score'],r['worst_d'],flush=True)
 if tag.startswith(('HRseed','WSseed','R326seed')):dump('partial_results.json',results)
# Stored prediction screen: fixed route, E replacement, unchanged F weights.
for seed,root in [(5001,b.K/'eseed/pe/hr5_s1_f4'),(6001,b.K/'eseed/pe/hr6_s1_f4')]+[(s,D/f'probabilities/hr_{s}') for s in (7001,8001,9001)]:
 preds={}
 for n,r in b.recs.items():
  p=load(root,n)[b.T2M];g=b.rt((r['N3']+p)/2)
  preds[n]=b.finalize(r,r['r13'],r['e14']+.25*(g-r['g1']))
 record(f'HRseed{seed}',preds,13)
for s in (4,5,6,7,8):
 roots=[b.K/f'eseed/pe/ws{s}_f{f}' if s<7 else D/f'probabilities/ws{s}_f{f}' for f in (1,3)]
 vals={n:[load(p,n)[b.T2M] for p in roots] for n in b.names}
 for selected in ((0,1),(0,),(1,)):
  preds={}
  for n,r in b.recs.items():
   p=b.rt(sum(vals[n][j] for j in selected)/len(selected));e=r['e14']+.5*(p-r['bold']);preds[n]=b.finalize(r,r['r13'],e)
  record(f'WSseed{s}_folds'+''.join(str((1,3)[i]) for i in selected),preds,11+len(selected))
for s in range(4):
 preds={}
 root=b.K/f'codex_r326_ema_seed4_20261012/probabilities/set{s}/fold2/plain'
 for n,r in b.recs.items():
  p=b.rt(load(root,n));rnew=b.predict_r(r,b.CFG,{**r['src'],'326':p});preds[n]=b.finalize(r,rnew)
 record(f'R326seed_set{s}_shared',preds,13)
# R329 single-view MAIN channel producer (b4tta source is TEAM; explicit conversion).
rf={}
for f in (1,3,4):
 rf[f]={n:load(b.PR/f'R/f{f}/none',n)[b.T2M] for n in b.names}
full_diffs={n:float(np.max(np.abs(b.rt(sum(rf[f][n] for f in rf)/3)-b.recs[n]['src']['329']))) for n in b.names}
dump('R329_cache_precision.json',{'full_group_max_abs':max(full_diffs.values()),'per_image':full_diffs,'note':'Full group retains original fresh family probabilities; reduced subsets use per-fold cached probability means, require fresh confirmation.'})
for size in (1,2,3):
 for fs in itertools.combinations((1,3,4),size):
  for ws in ((1,3),(1,),(3,)):
   preds={}
   for n,r in b.recs.items():
    rp=r['src']['329'] if size==3 else b.rt(sum(rf[f][n] for f in fs)/size)
    rr=r['r13'] if size==3 else b.predict_r(r,b.CFG,{**r['src'],'329':rp})
    ep=None if len(ws)==2 else r['e14']+.5*(r['wsf1' if ws[0]==1 else 'wsf3']-r['bold'])
    preds[n]=b.finalize(r,rr,ep)
   record('reduce_R329_'+''.join(map(str,fs))+'_WS_'+''.join(map(str,ws)),preds,8+size+len(ws))
summary={}
for group,keys in {'HRsingle5':[f'HRseed{s}' for s in (5001,6001,7001,8001,9001)],'B4ws_pair5':[f'WSseed{s}_folds13' for s in (4,5,6,7,8)],'R326_shared4':[f'R326seed_set{s}_shared' for s in range(4)]}.items():
 a=np.array([results[k]['score'] for k in keys]);summary[group]={'n':len(a),'mean_percent':float(a.mean()),'sample_sd_pp':float(a.std(ddof=1)),'range_pp':float(np.ptp(a)),'worst_image_seed_ranges':sorted([{'image':n,'range_pp':float(np.ptp([results[k]['per_image'][n] for k in keys]))} for n in b.names],key=lambda r:-r['range_pp']),'cases':keys}
dump('results.json',results);dump('seed_variability.json',summary)
dump('ranking.json',sorted([{'case':k,**v} for k,v in results.items()],key=lambda r:-r['score']))
