"""Image-only stage1 hard fusion, no refiner and no GT."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from PIL import Image
p=argparse.ArgumentParser()
p.add_argument('--release',type=Path,required=True)
p.add_argument('--expert',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists()
m=json.loads((a.release/'input_manifest.json').read_text());palette=np.asarray(m['config']['palette'],dtype=np.uint8)
for kind in ['predicted_labels','predicted_masks']:(a.output/kind).mkdir(parents=True)
rows=[]
for row in m['images']:
 name=row['name'];r=np.asarray(Image.open(a.release/f'predicted_labels/{name}.png'));e=np.load(a.expert/f'{name}.npy')
 assert e.shape==(4,*r.shape) and np.isfinite(e).all() and np.all(r<4)
 label=np.array([2,0,1,3],dtype=np.uint8)[e[[2,0,1,3]].argmax(0)]
 pred=np.where(np.isin(label,[0,2]),label,r).astype(np.uint8)
 Image.fromarray(pred).save(a.output/f'predicted_labels/{name}.png');Image.fromarray(palette[pred]).save(a.output/f'predicted_masks/{name}_mask.png')
 rows.append({'name':name,'changed_pixels_vs_fresh_R':int(np.count_nonzero(pred!=r)),'total_pixels':int(r.size),'label_pixels_sha256':hashlib.sha256(pred.tobytes()).hexdigest()})
(a.output/'input_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
(a.output/'completion.json').write_text(json.dumps({'images':rows,'gt_read':False,'refiner_used':False,'rule':'E labels Al3Ni(0) or Al(2) override R'},indent=2)+'\n')
