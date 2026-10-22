import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import copy,json,time
import numpy as np
import torch
from PIL import Image
from fields_io import api,finish,sha,write
from residual_model import Residual
ROOT=Path(__file__).resolve().parent
def main():
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 train=json.loads((ROOT/'training/completion.json').read_text());checkpoint=Path(train['final_checkpoint']);assert sha(checkpoint)==train['final_sha256'] and train['epochs']==8 and train['optimizer_updates']==560
 model=Residual().cuda();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict']);model.eval();zero=Residual().cuda().eval()
 teacher=ROOT/'runs/teacher_valid20';m=json.loads((teacher/'input_manifest.json').read_text());fields=json.loads((ROOT/'features/valid/manifest.json').read_text());assert len(m['images'])==20 and fields['valid_gt_decoded'] is False
 out=ROOT/'runs/refined_valid20';out.mkdir(exist_ok=False);(out/'predicted_labels').mkdir();(out/'predicted_masks').mkdir()
 new=copy.deepcopy(m);new['refiner']={'checkpoint_sha256':train['final_sha256'],'parameters':11876,'class_order':['Al3Ni','PrimarySi','Al','EutecticSi'],'training_receipt_sha256':sha(ROOT/'training/completion.json'),'code_sha256':sha(Path(__file__))};write(out/'input_manifest.json',new)
 palette=np.array(m['config']['palette'],dtype=np.uint8);entry=api()[0];scores=[];start=time.monotonic();zero_exact=0
 feature_rows={x['name']:x for x in fields['images']}
 for row in m['images']:
  name=row['name'];p=ROOT/'features/valid'/f'{name}.npz';assert sha(p)==feature_rows[name]['features_sha256']
  with np.load(p) as z:lp=z['logits'];gp=z['gray'];assert 'target' not in z.files
  logits=torch.from_numpy(lp[None]).cuda();gray=torch.from_numpy(gp[None,None]).cuda()
  with torch.no_grad():
   identity,d=zero(logits,gray);assert torch.equal(identity,logits) and torch.count_nonzero(d)==0
   value,delta=model(logits,gray);assert torch.isfinite(value).all() and torch.isfinite(delta).all() and delta.abs().max()<=1
  baseline=np.array(Image.open(teacher/'predicted_labels'/f'{name}.png'));assert np.array_equal(finish(identity.cpu().numpy()[0],m['config']),baseline);zero_exact+=1
  pred=finish(value.cpu().numpy()[0],m['config']);Image.fromarray(pred).save(out/'predicted_labels'/f'{name}.png');Image.fromarray(palette[pred]).save(out/'predicted_masks'/f'{name}_mask.png')
  gt=entry.decode_gt(Path('/workspace/komap_model1/data/valid/masks')/f'{name}_mask.png',palette);assert pred.shape==gt.shape
  iou=[]
  for c in range(4):
   inter=np.count_nonzero((pred==c)&(gt==c));union=np.count_nonzero((pred==c)|(gt==c));iou.append(float(inter/union) if union else 0.)
  scores.append({'image':name,'class_iou':iou,'image_miou':float(np.mean(iou))});print('REFINED',name,flush=True)
 write(out/'metrics.json',{'metric':'mean of per-image four-class mIoU; empty union = 0','class_order':m['config']['class_order'],'count':20,'image_miou':float(np.mean([x['image_miou'] for x in scores])),'class_image_mean_iou':np.mean([x['class_iou'] for x in scores],0).tolist(),'images':scores})
 write(out/'completion.json',{'model':'Model349 plus native residual correction','images':20,'elapsed_seconds':time.monotonic()-start,'elapsed_kind':'refiner plus evaluation only; controller adds fresh parent and field preparation time','mean_image_miou':float(np.mean([x['image_miou'] for x in scores])),'zero_refiner_parent_masks_exact':zero_exact==20,'zero_refiner_images_checked':zero_exact,'training_started_during_inference':False,'inference_sha256':sha(Path(__file__)),'refiner_sha256':train['final_sha256'],'gpus':[os.environ.get('KOMAP_PHYSICAL_GPU','7')],'tent_steps':5})
if __name__=='__main__':main()
