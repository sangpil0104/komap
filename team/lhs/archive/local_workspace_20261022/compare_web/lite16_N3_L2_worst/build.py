from pathlib import Path
import json,shutil
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent;W=ROOT.parents[1];E=W/'experiments/model349_compression';A=ROOT/'assets';pal=np.array([[25,76,153],[76,178,76],[153,127,76],[204,204,204]],np.uint8);out=A/'viewer';out.mkdir(exist_ok=True)
base=json.loads((E/'codex_lite16_N3_L2_fresh_valid_20261013/N3/result.json').read_text());follow=json.loads((E/'lite16_N3_adopted_20261013/L2_followup_comparison.json').read_text());cases=[]
for name in ['A153_step3_sample3','A63_step1_sample4']:
 raw=np.asarray(Image.open(W/f'valid/images/{name}_image.png').convert('RGB'));rgb=np.asarray(Image.open(W/f'valid/masks/{name}_mask.png').convert('RGB'));gt=np.full(rgb.shape[:2],255,np.uint8)
 for c,col in enumerate(pal):gt[np.all(rgb==col,-1)]=c
 assert (gt<4).all();before=np.asarray(Image.open(A/f'codex_lite16_N3_L2_fresh_valid_20261013/N3/fusion/predicted_labels/{name}.png'));Image.fromarray(raw).save(out/f'{name}_raw.png');Image.fromarray(rgb).save(out/f'{name}_gt.png');Image.fromarray(pal[before]).save(out/f'{name}_before.png')
 for seed in [361,362,363,364]:
  after=np.asarray(Image.open(A/f'codex_L2_seed4_20261013/evaluation/seed{seed}/both/predicted_labels/{name}.png'));tag=f'{name}_{seed}';Image.fromarray(pal[after]).save(out/f'{tag}_after.png');loss=(before==gt)&(after!=gt);gain=(before!=gt)&(after==gt);diff=before!=after;wrong=(before!=gt)&(after!=gt)&diff
  over=raw.astype(float)*.55+255*.45
  for mask,col in [(loss,[235,55,55]),(gain,[15,185,110]),(wrong,[239,172,30])]:over[mask]=col
  Image.fromarray(over.astype(np.uint8)).save(out/f'{tag}_error.png')
  changes=raw.copy();changes[diff]=[211,65,220];Image.fromarray(changes).save(out/f'{tag}_change.png')
  def iou(p):return np.array([((p==c)&(gt==c)).sum()/u if (u:=((p==c)|(gt==c)).sum()) else 0 for c in range(4)])
  bi=iou(before);ai=iou(after);expected=json.loads((E/f'codex_L2_seed4_20261013/evaluation/seed{seed}/result.json').read_text())['variants']['both']['per_image'][name]
  assert abs(ai.mean()-expected['miou'])<1e-12 and abs(bi.mean()-base['per_image'][name]['miou'])<1e-12
  h,w=loss.shape;size=min(180,h,w);integ=np.pad(loss.astype(np.int32),((1,0),(1,0))).cumsum(0).cumsum(1);density=integ[size:,size:]-integ[:-size,size:]-integ[size:,:-size]+integ[:-size,:-size];regions=[];work=density.copy()
  for j in range(3):
   yy,xx=np.unravel_index(work.argmax(),work.shape);count=int(work[yy,xx]);regions.append({'x':int(xx+size/2),'y':int(yy+size/2),'size':size,'loss':count});work[max(0,yy-size):yy+size,max(0,xx-size):xx+size]=-1
  cases.append({'image':name,'seed':seed,'isWorst':follow[str(seed)]['worst_image']==name,'before':float(bi.mean()*100),'after':float(ai.mean()*100),'delta':float((ai.mean()-bi.mean())*100),'classDelta':((ai-bi)*100).tolist(),'newErrors':int(loss.sum()),'fixed':int(gain.sum()),'changed':int(diff.sum()),'wrongChanges':int(wrong.sum()),'width':w,'height':h,'regions':regions,'sources':[f'{name}_raw.png',f'{name}_gt.png',f'{name}_before.png',f'{tag}_after.png',f'{tag}_error.png',f'{tag}_change.png']})
(ROOT/'data.json').write_text(json.dumps({'cases':cases,'palette':pal.tolist(),'classNames':['Al3Ni','Primary Si','Al','Eutectic Si']},ensure_ascii=False))
print('8 case pairs verified against recorded mIoU')
