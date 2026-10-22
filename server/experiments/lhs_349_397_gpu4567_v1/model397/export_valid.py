import json,sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from PIL import Image
from tqdm import tqdm
run,repo=map(Path,sys.argv[1:])
sys.path.insert(0,str(repo/'experiments/lhs_349_397_gpu4567_v1/model349/adapters'))
import local_result_export as panels
# Model397 channel order differs from Model349; explicitly set display palette.
panels.PALETTE=np.array([(153,127,76),(25,76,153),(76,178,76),(204,204,204)],dtype=np.uint8)
names=['Al','Al3Ni','Primary Si','Eutectic Si']
images=sorted((repo/'dataset/valid/images').glob('*_image.png'))
if len(images)!=20: raise ValueError('valid20 required')
archives=[]
for family,seeds,weight in [('b4',[0],.5/4),('hr',[1,2],.25/4)]:
 for seed in seeds:
  for fold in range(1,5):
   file=run/'valid'/f'{family}_s{seed}_f{fold}'/('probabilities.npz' if family=='b4' else 'valid_probabilities.npz')
   archives.append((np.load(file,allow_pickle=False),weight))
try:
 for number in range(1,1000000):
  out=repo/'results'/f'model_results_{datetime.now(ZoneInfo("Asia/Seoul")):%m%d}_{number}'
  try: out.mkdir(); break
  except FileExistsError: continue
 for d in ['valid_mask_comparison','predicted_masks']: (out/d).mkdir()
 rows=[]
 for image in tqdm(images,desc='Model397 valid20 export',unit='image'):
  stem=image.name.removesuffix('_image.png'); prob=None
  for archive,weight in archives:
   keys=[stem,stem+'_image']; key=next((k for k in keys if k in archive.files),None)
   if key is None: raise KeyError(stem)
   v=archive[key].astype(np.float32); v=v/np.maximum(v.sum(axis=0,keepdims=True),1e-8)
   prob=v*weight if prob is None else prob+v*weight
  pred=prob.argmax(0).astype(np.uint8)
  rgb=np.array(Image.open(repo/'dataset/valid/masks'/f'{stem}_mask.png').convert('RGB'))
  target=np.empty(rgb.shape[:2],dtype=np.uint8); seen=np.zeros(target.shape,bool)
  for c,color in enumerate(panels.PALETTE):
   match=np.all(rgb==color,axis=-1); target[match]=c; seen|=match
  if not seen.all() or pred.shape!=target.shape: raise ValueError(stem)
  ious=[]
  for c in range(4):
   a,b=pred==c,target==c; u=np.logical_or(a,b).sum(); ious.append(float(np.logical_and(a,b).sum()/u) if u else 0.)
  score=float(np.mean(ious)); rows.append(dict(image=stem,image_miou=score,class_iou=dict(zip(names,ious))))
  panels.comparison(image,pred,target,score,out/'valid_mask_comparison'/f'{stem}.png')
  Image.fromarray(panels.PALETTE[pred]).save(out/'predicted_masks'/f'{stem}_mask.png')
 score=float(np.mean([r['image_miou'] for r in rows]))
 (out/'metrics.json').write_text(json.dumps(dict(model='Model397 local valid extension',image_miou=score,empty_union=0,images=rows,run=str(run)),indent=2))
 (out/f'{out.name}.md').write_text(f'# Model397 — local valid20 evaluation\n\nimage mIoU: **{score*100:.6f}%**\n\n원본 OOF 전용 번들에 추가한 로컬 검증 평가입니다. 각 family의 4-fold 확률 평균을 0.50 B4 + 0.25 HR seed1 + 0.25 HR seed2로 결합합니다. 원본 기록의 공식 valid 결과가 아닙니다. 빈 클래스 IoU=0.\n\nRun: `{run}`\n\n| Image | image mIoU | Comparison |\n|---|---:|---|\n'+'\n'.join(f"| {r['image']} | {r['image_miou']*100:.4f}% | [image](valid_mask_comparison/{r['image']}.png) |" for r in rows))
 (run/'result_path.txt').write_text(str(out)+'\n'); print(f'RESULT_DIRECTORY={out}')
finally:
 for archive,_ in archives: archive.close()
