"""Independent CPU scoring of exported FP16 probabilities; separate from FP32 evaluator score."""
import json, csv, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent
DATA=Path('/workspace/komap_model1/data_model382_seed1_fold1/valid/masks')
PALETTE=np.array([[153,127,76],[25,76,153],[76,178,76],[204,204,204]],np.uint8)
def main():
 assert json.loads((ROOT/'completion.json').read_text())['all_stages_completed']
 results={}
 for arm in ['original','texture']:
  directory=ROOT/(arm+'_eval');rows=[]
  with np.load(directory/'valid_probabilities.npz') as z:
   for name in sorted(z.files):
    p=z[name].astype(np.float32);rgb=np.array(Image.open(DATA/(name+'_mask.png')).convert('RGB'))
    matches=(rgb[:,:,None,:]==PALETTE[None,None,:,:]).all(-1)
    assert matches.sum(-1).min()==1 and matches.sum(-1).max()==1
    y=matches.argmax(-1);assert p.shape==(4,*y.shape) and np.isfinite(p).all()
    pred=p.argmax(0);ious=[]
    for c in range(4):
     union=np.count_nonzero((pred==c)|(y==c));intersection=np.count_nonzero((pred==c)&(y==c))
     ious.append(intersection/union if union else 0.)
    rows.append({'image':name,'iou':ious,'prediction_sha256':hashlib.sha256(pred.astype(np.uint8).tobytes()).hexdigest()})
  saved=list(csv.DictReader((directory/'valid_per_image.csv').open()))
  assert {r['image_id'] for r in saved}=={r['image'] for r in rows} and len(rows)==18
  summary=json.loads((directory/'valid_summary.json').read_text())
  fp32=100*np.mean([float(r['image_miou_empty_zero']) for r in saved])
  assert abs(fp32-100*summary['image_miou_empty_zero'])<1e-10
  results[arm]={'fp32_evaluator_percent':float(fp32),'fp16_independent_percent':float(100*np.mean([r['iou'] for r in rows])),'fp16_class_percent':(100*np.mean([r['iou'] for r in rows],axis=0)).tolist(),'rows':rows}
 delta=results['texture']['fp16_independent_percent']-results['original']['fp16_independent_percent']
 classes=np.array(results['texture']['fp16_class_percent'])-results['original']['fp16_class_percent']
 result={'arms':results,'texture_minus_control_pp':delta,'class_delta_pp':classes.tolist(),'pilot_gate_pass':bool(delta>=.1 and classes[1]>=-.1 and classes[2]>=-.1),'scope':'held train70 fold1 only; not official valid20 or OOF90; warmstart matched pilot','quantization_note':'Independent scores use exported FP16 probabilities; FP32 evaluator scores recorded separately'}
 (ROOT/'independent_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='arms'},indent=2))
if __name__=='__main__':main()
