"""Separate PNG/GT scoring and conditional specimen leave-one-out selection."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent
PALETTE=np.array([(25,76,153),(76,178,76),(153,127,76),(204,204,204)],dtype=np.uint8)
COUNTS={'control':12,'seed1':8,'seed2':8,'b4':4,'hr':8,'hr1':4,'hr2':4}
def iou(pred,truth):
 return [float(np.count_nonzero((pred==c)&(truth==c))/np.count_nonzero((pred==c)|(truth==c))) if np.count_nonzero((pred==c)|(truth==c)) else 0. for c in range(4)]
def main():
 receipt=json.loads((ROOT/'replay_completion.json').read_text());names=receipt['names'];keys=list(receipt['grid']);scores=np.zeros((2,len(keys),len(names),4));changed=np.zeros((2,len(keys),len(names)),dtype=int)
 for j,name in enumerate(names):
  rgb=np.asarray(Image.open(f'/workspace/komap_model1/data/valid/masks/{name}_mask.png').convert('RGB'));truth=np.full(rgb.shape[:2],255,dtype=np.uint8)
  for c,color in enumerate(PALETTE):truth[np.all(rgb==color,axis=-1)]=c
  assert np.all(truth<4)
  for stage in [1,2]:
   base=np.asarray(Image.open(ROOT/f'runs/control/stage{stage}/predicted_labels/{name}.png'))
   for k,key in enumerate(keys):
    pred=np.asarray(Image.open(ROOT/f'runs/{key}/stage{stage}/predicted_labels/{name}.png'));assert pred.shape==truth.shape and np.all(pred<4)
    scores[stage-1,k,j]=iou(pred,truth);changed[stage-1,k,j]=np.count_nonzero(pred!=base)
 results={}
 for stage in [1,2]:
  values=scores[stage-1];baseline=float(values[0].mean()*100);assert abs(baseline-[80.474620617154,80.50068909458149][stage-1])<1e-7,baseline
  rows=[]
  for k,key in enumerate(keys):
   diff=(values[k].mean(1)-values[0].mean(1))*100
   rows.append({'key':key,'checkpoint_count':14+COUNTS[key]+(stage==2),'mean_percent':float(values[k].mean()*100),'delta_pp':float(diff.mean()),'worst_image_delta_pp':float(diff.min()),'worst_image':names[int(diff.argmin())],'class_delta_pp':((values[k].mean(0)-values[0].mean(0))*100).tolist(),'quality_pass':bool(diff.mean()>=-.01-1e-10 and diff.min()>=-.1-1e-10),'changed_pixels':int(changed[stage-1,k].sum())})
  def choose(mask):
   qualified=[]
   for k,key in enumerate(keys):
    diff=(values[k,mask].mean(1)-values[0,mask].mean(1))*100
    if diff.mean()>=-.01-1e-10 and diff.min()>=-.1-1e-10:qualified.append(k)
   return min(qualified,key=lambda k:(COUNTS[keys[k]],-float(values[k,mask].mean()),k))
  best=choose(np.ones(len(names),dtype=bool));cv=[];choices=[]
  for name in names:
   mask=np.array([n.split('_')[0]!=name.split('_')[0] for n in names]);k=choose(mask);j=names.index(name);cv.append(values[k,j]);choices.append({'name':name,'chosen':keys[k]})
  results[f'stage{stage}']={'baseline_percent':baseline,'candidates':rows,'selected':keys[best],'conditional_loso_percent':float(np.mean(cv)*100),'conditional_loso_delta_pp':float((np.mean(cv)-values[0].mean())*100),'loso_choices':choices}
 result={'names':names,'results':results,'class_iou':scores.tolist(),'keys':keys,'repeated_valid':True,'test_read':False,'fresh_raw_inference':False,'speed_measured':False,'independent_generalization_verified':False,'parent_choice_bias_remaining':True}
 (ROOT/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(results,indent=2))
if __name__=='__main__':main()
