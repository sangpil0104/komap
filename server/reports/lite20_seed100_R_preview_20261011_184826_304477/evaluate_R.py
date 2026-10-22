from pathlib import Path
import sys,json,pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
cv2.setNumThreads(2)
EXPERIMENT=Path('/Data3/sppark/komap/experiments/model349_lite20_seed100_gpu4567_v1')
run=Path(json.loads((EXPERIMENT/'outputs/latest_run.json').read_text())['run'])
sys.path.insert(0,str(run/'implementation'))
from common import BASE,read_json,write_json,sha
from engine import samples,frozen
from report_utils import class_ious
from features import decode
out=Path(__file__).resolve().parent
candidates=[read_json(run/f'completed/336_{i:02d}.json') for i in range(8)]
config=frozen.load_config(run/'implementation/frozen/configs/model349_lite.yaml')
measurements={'baseline':[] ,**{r['id']:[] for r in candidates}}
for row in tqdm(samples(),desc='R-only: baseline + 8 seeds',unit='image',mininterval=5):
 name=row['name'];target=decode(row['gt'])
 with (BASE/f'release_inference/probabilities/base/{name}.pkl').open('rb') as f:item=pickle.load(f)
 sources={c:np.load(BASE/f'release_inference/probabilities/{c}/{name}.npy').astype(np.float32) for c in ['322','326','329','336','347','269']}
 for candidate in [None,*candidates]:
  selected=dict(sources)
  if candidate:
   path=Path(candidate['context']['probabilities']['336'])/f'{name}.npy'
   assert sha(path)==candidate['artifacts'][str(path)],str(path)
   assert set(candidate['context']['probabilities'])=={'336'}
   selected['336']=np.load(path).astype(np.float32)
  pred=frozen.predict_fixed(item,selected,config).astype(np.uint8)
  if candidate is None:
   assert np.array_equal(pred,np.asarray(Image.open(BASE/f'release_inference/predicted_labels/{name}.png'))),name
  ious=class_ious(pred,target)
  measurements[candidate['id'] if candidate else 'baseline'].append(dict(image=name,class_iou=ious,image_miou=float(np.mean(ious))))
baseline=float(np.mean([r['image_miou'] for r in measurements['baseline']]))
assert abs(baseline-read_json(BASE/'release_inference/metrics.json')['image_miou'])<1e-12
rows=[]
for c in candidates:
 score=float(np.mean([r['image_miou'] for r in measurements[c['id']]]))
 rows.append(dict(id=c['id'],seed=c['seed'],R_image_miou=score,delta_pp=(score-baseline)*100,REF_image_miou=c['image_miou'],images=measurements[c['id']]))
rows.sort(key=lambda r:-r['R_image_miou'])
write_json(out/'metrics.json',dict(run=str(run),baseline_R_image_miou=baseline,baseline_pixel_exact=True,metric='mean image-wise four-class IoU; empty union=0',count=20,candidates=rows))
lines=['# Lite20 seed100: first 8 candidates, R-only preview','',f'Baseline R image mIoU: {baseline*100:.6f}%','', 'HRNet336 alone is replaced; all other R components are fixed. E/F are excluded.','', '| Rank | Seed | R image mIoU (%) | Change (pp) |','|---:|---:|---:|---:|']
for i,r in enumerate(rows,1):lines.append(f'| {i} | {r["seed"]} | {r["R_image_miou"]*100:.6f} | {r["delta_pp"]:+.6f} |')
(out/'preview.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(baseline=baseline,rows=[{k:v for k,v in r.items() if k!='images'} for r in rows],report=str(out/'preview.md')),ensure_ascii=False),flush=True)
