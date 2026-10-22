"""Independent official image-wise four-class mask score; empty union=0."""
from pathlib import Path
import argparse,json
import numpy as np
from PIL import Image
import yaml
ROOT=Path(__file__).resolve().parents[1]/'inference'
p=argparse.ArgumentParser();p.add_argument('--labels',type=Path,required=True);p.add_argument('--gt',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
palette=np.asarray(yaml.safe_load((ROOT/'config/r_config.yaml').read_text())['palette'],dtype=np.uint8)
rows=[]
for path in sorted(a.labels.glob('*.png')):
 name=path.stem;pred=np.asarray(Image.open(path));gtpath=a.gt/f'{name}_mask.png';raw=np.asarray(Image.open(gtpath))
 if raw.ndim==2 and np.isin(raw,range(4)).all():target=raw
 else:
  rgb=np.asarray(Image.open(gtpath).convert('RGB'));target=np.full(rgb.shape[:2],255,dtype=np.uint8)
  for c,color in enumerate(palette):target[np.all(rgb==color,axis=-1)]=c
 assert pred.shape==target.shape and np.isin(pred,range(4)).all() and np.isin(target,range(4)).all()
 vals=[]
 for c in range(4):
  x=pred==c;y=target==c;u=np.count_nonzero(x|y);vals.append(float(np.count_nonzero(x&y)/u) if u else 0.)
 rows.append({'image':name,'class_iou':vals,'image_miou':float(np.mean(vals))})
assert rows
result={'metric':'mean image-wise 4class IoU; empty union=0','count':len(rows),'image_miou':float(np.mean([r['image_miou'] for r in rows])),'images':rows}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(result['image_miou']*100)
