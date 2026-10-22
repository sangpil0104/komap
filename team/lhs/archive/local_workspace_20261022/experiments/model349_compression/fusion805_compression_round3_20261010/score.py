"""Separate PNG scoring against frozen original27, lite25, and final lite23."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent
PALETTE=np.array([(25,76,153),(76,178,76),(153,127,76),(204,204,204)],np.uint8)
def main():
 receipt=json.loads((ROOT/'replay_completion.json').read_text());names=receipt['names'];cfg=receipt['configs'];keys=[c['key'] for c in cfg];scores=np.zeros((2,len(keys),20,4))
 for j,name in enumerate(names):
  rgb=np.asarray(Image.open(f'/workspace/komap_model1/data/valid/masks/{name}_mask.png').convert('RGB'));truth=np.full(rgb.shape[:2],255,np.uint8)
  for c,color in enumerate(PALETTE):truth[np.all(rgb==color,axis=-1)]=c
  assert np.all(truth<4)
  for stage in [1,2]:
   for k,key in enumerate(keys):
    pred=np.asarray(Image.open(ROOT/f'runs/{key}/stage{stage}/predicted_labels/{name}.png'));assert pred.shape==truth.shape and np.all(pred<4)
    cm=np.bincount(truth.ravel()*4+pred.ravel(),minlength=16).reshape(4,4);intersection=np.diag(cm);union=cm.sum(0)+cm.sum(1)-intersection;scores[stage-1,k,j]=np.divide(intersection,union,out=np.zeros(4),where=union>0)
  print('scored',name,flush=True)
 results={}
 for stage,values in enumerate(scores,1):
  image=values.mean(2);counts=[c['R_checkpoint_count']+4+len(c['hr1'])+len(c['hr2'])+(stage==2) for c in cfg];references=['original27','reference25','reference23','reference23_cache'];rows=[]
  def passes(k,mask):
   return all((delta.mean()>=-.01-1e-10 and delta.min()>=-.1-1e-10) for delta in [(image[k,mask]-image[keys.index(ref),mask])*100 for ref in references])
  for k,key in enumerate(keys):
   deltas={ref:{'mean_pp':float(((image[k]-image[keys.index(ref)])*100).mean()),'worst_pp':float(((image[k]-image[keys.index(ref)])*100).min())} for ref in references}
   rows.append({'key':key,'checkpoint_count':counts[k],'mean_percent':float(values[k].mean()*100),'quality_pass':bool(passes(k,np.ones(20,bool))),'deltas':deltas})
  def choose(mask):
   qualified=[k for k in range(len(keys)) if passes(k,mask)]
   assert qualified
   return min(qualified,key=lambda k:(counts[k],-float(values[k,mask].mean()),k))
  best=choose(np.ones(20,bool));cv=[];choices=[]
  for j,name in enumerate(names):
   mask=np.array([n.split('_')[0]!=name.split('_')[0] for n in names]);k=choose(mask);cv.append(values[k,j]);choices.append({'name':name,'key':keys[k]})
  results[f'stage{stage}']={'selected':keys[best],'candidates':rows,'conditional_loso_percent':float(np.mean(cv)*100),'conditional_loso_delta_vs_lite23_pp':float((np.mean(cv)-values[keys.index('reference23')].mean())*100),'loso_choices':choices}
 assert abs(scores[1,keys.index('reference23')].mean()*100-80.51727047204673)<1e-8
 result={'names':names,'keys':keys,'configs':cfg,'class_iou':scores.tolist(),'results':results,'repeated_valid':True,'independent_generalization_verified':False,'test_read':False,'speed_measured':False}
 (ROOT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 for stage,d in results.items():
  print(stage,d['selected'],'LOSO',d['conditional_loso_percent'])
  for row in d['candidates']:
   if row['quality_pass']:print(row)
if __name__=='__main__':main()
