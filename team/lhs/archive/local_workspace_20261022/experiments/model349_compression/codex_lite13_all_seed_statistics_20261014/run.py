import importlib.util,json,time
from pathlib import Path
import numpy as np,torch
K=Path('/home/sangpil/komap');D=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('b',K/'codex_lite13_E_seed5_reduction_20261014/baseline_snapshot.py');b=importlib.util.module_from_spec(sp);sp.loader.exec_module(b)
summary=json.loads((K/'codex_lite13_E_seed5_reduction_20261014/current_completed_seed_statistics.json').read_text())['families'];cases={}
def score(tag,preds):
 r=b.score(preds);r['score']=b.BASE+r['delta'];cases[tag]=r
 (D/'partial_results.json').write_text(json.dumps(cases,indent=2));print(tag,r['score'],r['worst_d'],flush=True)
 return r
for s in range(4021,4026):
 root=K/f'm269/p269/s{s}';pp={}
 for n,r in b.recs.items():
  p=np.load(root/f'{n}.npy').astype(np.float32);rr=b.predict_r(r,b.CFG,{**r['src'],'269':p});pp[n]=b.finalize(r,rr)
 score(f'Res269_seed{s}',pp)
for s in range(4031,4036):
 m=b.fu.Residual().eval();m.load_state_dict(torch.load(K/f'fseed/s{s}/epoch8.pt',map_location='cpu',weights_only=False)['state_dict']);pp={}
 for n,r in b.recs.items():
  with torch.no_grad():v,_=m(torch.from_numpy(np.log(np.maximum(r['e14'],1e-7))[None]),r['gray'])
  fl=b.fu.labels(v.numpy()[0]);ex=b.fu.labels(r['e14']);pp[n]=np.where(fl==2,2,np.where(np.isin(ex,[0,2]),ex,r['r13'])).astype(np.uint8)
 score(f'F_seed{s}',pp)
for s in (4,5,6):
 pp={}
 for n,r in b.recs.items():
  p=b.rt((np.load(K/f'eseed/pe/ws{s}_f1/{n}.npy').astype(np.float32)+np.load(K/f'eseed/pe/ws{s}_f3/{n}.npy').astype(np.float32))/2)[b.T2M]
  pp[n]=b.finalize(r,r['r13'],r['e14']+.5*(p-r['bold']))
 score(f'B4ws_pair_seed{s}',pp)
for s in (5001,6001,7001):
 root=K/f'eseed/pe/hr{s//1000}_s1_f4' if s!=7001 else K/'codex_lite13_E_seed5_reduction_20261014/probabilities/hr_7001'
 if not (root/'complete.json').exists():continue
 pp={}
 for n,r in b.recs.items():
  p=np.load(root/f'{n}.npy').astype(np.float32)[b.T2M];g=b.rt((r['N3']+p)/2);pp[n]=b.finalize(r,r['r13'],r['e14']+.25*(g-r['g1']))
 score(f'HRfold4_seed{s}',pp)
for family,tag in [('Res269','Res269_seed'),('F','F_seed'),('B4ws_pair','B4ws_pair_seed'),('HRfold4','HRfold4_seed')]:
 rows=[{'seed':key,**r} for key,r in cases.items() if key.startswith(tag)];a=np.array([r['score'] for r in rows]);summary[family]={'n':len(a),'mean':float(a.mean()),'sample_sd_pp':float(a.std(ddof=1)) if len(a)>1 else None,'min':float(a.min()),'max':float(a.max()),'range_pp':float(np.ptp(a)),'rows':rows,'max_image_seed_range_pp':max(float(np.ptp([r['per_image'][n] for r in rows])) for n in b.names),'parent_note':'P100 model257 parent differs from published269' if family=='Res269' else None}
(D/'all_available_seed_statistics.json').write_text(json.dumps({'baseline':b.BASE,'all_scored_on_current_lite13':True,'seed_sd_policy':'sample ddof1, pp','families':summary,'missing_current_context':['R ResNet restart61','R HR347','R B4 width4000/329'],'test_gt_used':False},indent=2));(D/'completion.json').write_text(json.dumps({'completed':time.time(),'families':len(summary)}));print('COMPLETE',flush=True)
