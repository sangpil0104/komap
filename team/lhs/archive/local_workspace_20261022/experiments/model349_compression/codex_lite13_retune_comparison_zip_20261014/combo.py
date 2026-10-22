import importlib.util,json,copy
from pathlib import Path
import numpy as np
K=Path('/home/sangpil/komap');D=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('b',K/'codex_lite13_E_seed5_reduction_20261014/baseline_snapshot.py');b=importlib.util.module_from_spec(sp);sp.loader.exec_module(b)
st=json.loads((K/'b4tta/search13/retune_full.json').read_text())['state'];cfg=copy.deepcopy(b.CFG);cfg['route']['base'].update(st['R']);preds=b.state_eval(st,None);control=b.score(preds);cs=b.candidates();out=[]
for key in ('p336_3410','p100_hr336_control','p100_hr336_context'):
 order=json.loads((K/f'b4tta/search13/cand_{key}.json').read_text())['_order']
 for alpha in (.05,.1):
  pp={}
  for n,r in b.recs.items():
   w=st['E']['wb4'];g=st['E']['g1share'];e=r['e14']+(w-.5)*r['bold']+((1-w)*g-.25)*r['g1']+((1-w)*(1-g)-.25)*r['g2'];c=b.rt(b.load_cand(cs[key],n)[order]);ee=(1-alpha)*e+alpha*c
   pp[n]=b.finalize(r,b.predict_r(r,cfg),ee)
  s=b.score(pp);s.update(candidate=key,alpha=alpha,score=b.BASE+s['delta'],vs_retune_pp=s['delta']-control['delta'],actual_weight_count=14);out.append(s);print(key,alpha,s['score'],s['worst_d'],flush=True)
(D/'combo_results.json').write_text(json.dumps({'baseline':b.BASE,'retune_only':b.BASE+control['delta'],'new_coefficient_search':False,'cases':out},indent=2));print('COMBO_COMPLETE',flush=True)
