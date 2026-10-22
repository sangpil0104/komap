"""CPU component replay and official metrics after all label-free maps exist."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import copy,json,pickle,time
import numpy as np
from PIL import Image
from fields_io import api,SOURCE,sha,write
ROOT=Path(__file__).resolve().parent;PRIOR=ROOT.parent/'accuracy81_residual_20261008/runs/teacher_valid20'
def main():
 gpu=json.loads((ROOT/'gpu_completion.json').read_text());assert gpu['baseline_probability_maps_exact'] and not gpu['valid_gt_opened']
 manifest=json.loads((PRIOR/'input_manifest.json').read_text());cfg=manifest['config'];entry=api()[0];palette=np.array(cfg['palette'],np.uint8)
 old={x['image']:x for x in json.loads((PRIOR/'metrics.json').read_text())['images']};(ROOT/'runs').mkdir()
 for kind in ['baseline','alpha025','alpha1']:
  out=ROOT/'runs'/kind;out.mkdir();(out/'predicted_labels').mkdir();(out/'predicted_masks').mkdir();scores=[];start=time.monotonic();m=copy.deepcopy(manifest);m['hr336_moments']={'alpha':{'baseline':0.,'alpha025':.25,'alpha1':1.}[kind],'image_specific':True,'source_sha256':sha(ROOT/'predict_moments.py'),'raw_unchanged_components_reused':True};write(out/'input_manifest.json',m)
  for row in manifest['images']:
   name=row['name']
   with (PRIOR/'probabilities/base'/f'{name}.pkl').open('rb') as f:item=pickle.load(f)
   sources={c:np.load(PRIOR/'probabilities'/c/f'{name}.npy').astype(np.float32) for c in entry.active_components(cfg) if c!='base'}
   sources['336']=np.load(ROOT/'probabilities'/kind/f'{name}.npy').astype(np.float32)
   pred=entry.predict_fixed(item,sources,cfg)
   if kind=='baseline':assert np.array_equal(pred,np.array(Image.open(PRIOR/'predicted_labels'/f'{name}.png')))
   target=entry.decode_gt(Path('/workspace/komap_model1/data/valid/masks')/f'{name}_mask.png',palette);iou=[]
   for c in range(4):
    union=np.count_nonzero((pred==c)|(target==c));intersection=np.count_nonzero((pred==c)&(target==c));iou.append(float(intersection/union) if union else 0.)
   if kind=='baseline':assert np.array_equal(iou,old[name]['class_iou'])
   Image.fromarray(pred).save(out/'predicted_labels'/f'{name}.png');Image.fromarray(palette[pred]).save(out/'predicted_masks'/f'{name}_mask.png')
   scores.append({'image':name,'class_iou':iou,'image_miou':float(np.mean(iou))});print('REPLAY',kind,name,flush=True)
  score=float(np.mean([x['image_miou'] for x in scores]));write(out/'metrics.json',{'metric':'imagewise mean4class IoU; emptyunion0','count':20,'image_miou':score,'class_order':cfg['class_order'],'images':scores});write(out/'completion.json',{'images':20,'mean_image_miou':score,'elapsed_seconds':time.monotonic()-start,'elapsed_kind':'CPU component replay scoring only; changed HR336 GPU stages recorded separately; not whole fresh pipeline','raw_unchanged_components_reused':True,'baseline_all80_iou_exact':True,'checkpoint_count':14})
if __name__=='__main__':main()
