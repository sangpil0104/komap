"""Independently reaggregate recorded scores without model dependencies."""
from pathlib import Path
import json,statistics as st
r=Path(__file__).resolve().parent;d=json.loads((r/'dual_seed/result.json').read_text());keys=d['keys'];names=d['names'];summary={}
for stage,values in enumerate(d['class_iou'],1):
 images=[[st.mean(x) for x in candidate] for candidate in values];rows=d['results'][f'stage{stage}']['candidates'];counts=[x['checkpoint_count'] for x in rows]
 def passes(k,indices):
  return all(st.mean(delta)>=-.01-1e-10 and min(delta)>=-.1-1e-10 for delta in [[(images[k][i]-images[keys.index(ref)][i])*100 for i in indices] for ref in ['original27','reference25','r_replay25']])
 def choose(indices):
  return min((k for k in range(len(keys)) if passes(k,indices)),key=lambda k:(counts[k],-st.mean(images[k][i] for i in indices),k))
 selected=choose(range(len(names)));assert keys[selected]==d['results'][f'stage{stage}']['selected'];cv=[]
 for i,n in enumerate(names):
  k=choose([j for j,m in enumerate(names) if m.split('_')[0]!=n.split('_')[0]]);assert keys[k]==d['results'][f'stage{stage}']['loso_choices'][i]['chosen'];cv.append(images[k][i])
 for k,row in enumerate(rows):
  assert abs(st.mean(images[k])*100-row['mean_percent'])<1e-10;assert passes(k,range(len(names)))==row['quality_pass']
 loso=st.mean(cv)*100;assert abs(loso-d['results'][f'stage{stage}']['conditional_loso_percent'])<1e-10
 summary[f'stage{stage}']={'selected':keys[selected],'checkpoint_count':counts[selected],'mean_percent':st.mean(images[selected])*100,'conditional_loso_percent':loso,'all_candidate_gates_independently_match':True}
(r/'independent_aggregation.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
