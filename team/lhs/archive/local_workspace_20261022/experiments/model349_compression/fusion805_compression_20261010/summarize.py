"""Independent pure-Python reaggregation of recorded IoUs and gates."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent

def avg(x):return sum(x)/len(x)
def verify(result):
 for stage_index,stage in enumerate(['stage1','stage2']):
  values=result['class_iou'][stage_index];rows=result['results'][stage]['candidates'];image_values=[[avg(classes) for classes in candidate] for candidate in values]
  for k,row in enumerate(rows):
   deltas=[(v-b)*100 for v,b in zip(image_values[k],image_values[0])]
   assert abs(avg(image_values[k])*100-row['mean_percent'])<1e-9
   assert abs(avg(deltas)-row['delta_pp'])<1e-9
   assert abs(min(deltas)-row['worst_image_delta_pp'])<1e-9
   assert row['quality_pass']==(avg(deltas)>=-.01-1e-10 and min(deltas)>=-.1-1e-10)
  cv=[]
  for j,name in enumerate(result['names']):
   mask=[i for i,n in enumerate(result['names']) if n.split('_')[0]!=name.split('_')[0]];qualified=[]
   for k,row in enumerate(rows):
    delta=[(image_values[k][i]-image_values[0][i])*100 for i in mask]
    if avg(delta)>=-.01-1e-10 and min(delta)>=-.1-1e-10:qualified.append(k)
   best=min(qualified,key=lambda k:(rows[k]['checkpoint_count'],-avg([image_values[k][i] for i in mask]),k));cv.append(image_values[best][j])
  assert abs(avg(cv)*100-result['results'][stage]['conditional_loso_percent'])<1e-8

def main():
 results={key:json.loads((ROOT/path).read_text()) for key,path in [('family','result.json'),('prefix','fold_prune/result.json'),('subsets','subset_search/result.json')]}
 for r in results.values():verify(r)
 correspondence={'drop_seed2_fold4':'hr2_keep_123','drop_seed2_folds34':'hr2_keep_12','drop_seed2_folds234':'hr2_keep_1'}
 for before,after in correspondence.items():
  a=results['prefix']['keys'].index(before);b=results['subsets']['keys'].index(after)
  for stage in range(2):assert results['prefix']['class_iou'][stage][a]==results['subsets']['class_iou'][stage][b]
 assets=json.loads((ROOT/'asset_layouts.json').read_text());decision={}
 for stage in ['stage1','stage2']:
  rows=[dict(row,experiment=experiment) for experiment,result in results.items() for row in result['results'][stage]['candidates']]
  passing=[r for r in rows if r['quality_pass']];selected=min(passing,key=lambda r:(r['checkpoint_count'],-r['mean_percent']))
  selected['checkpoint_bytes']=assets[selected['key']][stage]['checkpoint_bytes'];selected['byte_reduction_percent']=(1-selected['checkpoint_bytes']/assets['control'][stage]['checkpoint_bytes'])*100
  selected['lighter_than_original']=selected['key']!='control';decision[stage]=selected
 out={'results':decision,'stored_iou_reaggregation_pass':True,'conditional_loso_reaggregation_pass':True,'prefix_subset_outputs_iou_exact':True,'repeated_valid_selection':True,'independent_generalization_verified':False,'test_used':False,'speed_measured':False}
 (ROOT/'decision.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(decision,indent=2))
if __name__=='__main__':main()
