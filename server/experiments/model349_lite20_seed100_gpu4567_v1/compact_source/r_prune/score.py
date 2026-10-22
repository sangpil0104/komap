"""PNG scoring; dual gates against original27 and preserved lite25."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent
PALETTE=np.array([(25,76,153),(76,178,76),(153,127,76),(204,204,204)],dtype=np.uint8)
def iou(pred,truth):
 return [float(np.count_nonzero((pred==c)&(truth==c))/np.count_nonzero((pred==c)|(truth==c))) if np.count_nonzero((pred==c)|(truth==c)) else 0. for c in range(4)]
def main():
 receipt=json.loads((ROOT/'replay_completion.json').read_text());names=receipt['names'];keys=[c['key'] for c in receipt['configs']];scores=np.zeros((2,len(keys),len(names),4));changed=np.zeros((2,len(keys),len(names)),dtype=int)
 counts=[c['checkpoint_count_stage2'] for c in receipt['configs']]
 for j,name in enumerate(names):
  rgb=np.asarray(Image.open(f'/workspace/komap_model1/data/valid/masks/{name}_mask.png').convert('RGB'));truth=np.full(rgb.shape[:2],255,dtype=np.uint8)
  for c,color in enumerate(PALETTE):truth[np.all(rgb==color,axis=-1)]=c
  assert np.all(truth<4)
  for stage in [1,2]:
   base=np.asarray(Image.open(ROOT/f'runs/reference25/stage{stage}/predicted_labels/{name}.png'))
   for k,key in enumerate(keys):
    pred=np.asarray(Image.open(ROOT/f'runs/{key}/stage{stage}/predicted_labels/{name}.png'));assert pred.shape==truth.shape and np.all(pred<4)
    scores[stage-1,k,j]=iou(pred,truth);changed[stage-1,k,j]=np.count_nonzero(pred!=base)
 results={}
 for stage in [1,2]:
  values=scores[stage-1];image_scores=values.mean(2);baseline=float(values[1].mean()*100)
  assert abs(baseline-[80.48884418839398,80.51589248148477][stage-1])<1e-8
  assert abs(values[0].mean()*100-[80.47462061710547,80.50068909458147][stage-1])<1e-8
  rows=[]
  for k,key in enumerate(keys):
   d25=(image_scores[k]-image_scores[1])*100;d27=(image_scores[k]-image_scores[0])*100
   check=lambda delta:bool(delta.mean()>=-.01-1e-10 and delta.min()>=-.1-1e-10)
   rows.append({'key':key,'checkpoint_count':counts[k]-(stage==1),'mean_percent':float(values[k].mean()*100),'delta_vs_lite25_pp':float(d25.mean()),'worst_vs_lite25_pp':float(d25.min()),'worst_vs_lite25_image':names[int(d25.argmin())],'delta_vs_original_pp':float(d27.mean()),'worst_vs_original_pp':float(d27.min()),'quality_pass':check(d25) and check(d27),'class_delta_vs_lite25_pp':((values[k].mean(0)-values[1].mean(0))*100).tolist(),'changed_pixels':int(changed[stage-1,k].sum())})
  def choose(mask):
   qualified=[]
   for k in range(len(keys)):
    d25=(image_scores[k,mask]-image_scores[1,mask])*100;d27=(image_scores[k,mask]-image_scores[0,mask])*100
    if all(delta.mean()>=-.01-1e-10 and delta.min()>=-.1-1e-10 for delta in [d25,d27]):qualified.append(k)
   return min(qualified,key=lambda k:(counts[k],-float(values[k,mask].mean()),k))
  best=choose(np.ones(len(names),dtype=bool));cv=[];choices=[]
  for j,name in enumerate(names):
   mask=np.array([n.split('_')[0]!=name.split('_')[0] for n in names]);k=choose(mask);cv.append(values[k,j]);choices.append({'name':name,'chosen':keys[k]})
  results[f'stage{stage}']={'baseline25_percent':baseline,'original27_percent':float(values[0].mean()*100),'candidates':rows,'selected':keys[best],'conditional_loso_percent':float(np.mean(cv)*100),'conditional_loso_delta_vs_lite25_pp':float(np.mean(cv)*100-baseline),'loso_choices':choices}
 result={'names':names,'results':results,'class_iou':scores.tolist(),'keys':keys,'repeated_valid':True,'test_read':False,'whole_model_raw_inference':False,'speed_measured':False,'independent_generalization_verified':False,'parent_choice_bias_remaining':True,'dual_baseline_gates':True}
 (ROOT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 for stage,v in results.items():
  print(stage,v['selected'],v['conditional_loso_percent'])
  for r in v['candidates']:print(r['key'],r['checkpoint_count'],r['mean_percent'],r['worst_vs_lite25_pp'],r['quality_pass'])
if __name__=='__main__':main()
