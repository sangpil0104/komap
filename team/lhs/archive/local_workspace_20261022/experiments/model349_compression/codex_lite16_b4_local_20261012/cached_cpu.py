import json,importlib.util,hashlib
from pathlib import Path
from multiprocessing import Pool
import numpy as np,torch
from PIL import Image
K=Path.home()/'komap';O=K/'codex_lite16_b4_local_20261012';A=K/'add';PE=A/'pe/valid';T2M=[1,2,0,3]
CANDS={'control':(1,2,3,4),'b4_13':(1,3),'b4_14':(1,4)}
def task(n):
 torch.set_num_threads(2)
 spec=importlib.util.spec_from_file_location('fu',A/'parent0/infer_fusion.py');fu=importlib.util.module_from_spec(spec);spec.loader.exec_module(fu)
 m=fu.Residual().eval();m.load_state_dict(torch.load(K/'weights/all/assets/refiner/epoch8.pt',map_location='cpu',weights_only=False)['state_dict'])
 man=json.loads((K/'l18h/e2e_valid/release/input_manifest.json').read_text());pal=np.array(man['config']['palette'],np.uint8)
 b4={f:np.load(PE/f'model329_ws_f{f}/{n}.npy').astype(np.float32)[T2M] for f in range(1,5)};full=np.mean(list(b4.values()),0)
 e0=np.load(K/f'l18h/e2e_valid/expert/{n}.npy').astype(np.float32)
 parent=np.array(Image.open(K/f'l18h/e2e_valid/release/predicted_labels/{n}.png'))
 reference=np.array(Image.open(K/f'l18h/e2e_valid/fusion/predicted_labels/{n}.png'))
 ip=K/f'data/valid/images/{n}_image.png';gray=torch.from_numpy(fu.prepare_gray(np.array(Image.open(ip).convert('L'),dtype=np.float32)/255.)[None,None])
 rgb=np.array(Image.open(K/f'data/valid/masks/{n}_mask.png').convert('RGB'));gt=np.full(rgb.shape[:2],255,np.uint8)
 for c,col in enumerate(pal):gt[np.all(rgb==col,-1)]=c
 assert np.all(gt<4);rows={}
 for tag,folds in CANDS.items():
  e=e0+.5*(np.mean([b4[f] for f in folds],0)-full)
  with torch.no_grad():v,_=m(torch.from_numpy(np.log(np.maximum(e,1e-7))[None]),gray)
  ex=fu.labels(e);r=fu.labels(v.numpy()[0]);pred=np.where(r==2,r,np.where(np.isin(ex,[0,2]),ex,parent)).astype(np.uint8)
  ci=[]
  for c in range(4):
   inter=((pred==c)&(gt==c)).sum();union=((pred==c)|(gt==c)).sum();ci.append(float(inter/union) if union else 0.)
  dest=O/'cached'/tag/'predicted_labels';Image.fromarray(pred).save(dest/f'{n}.png')
  rows[tag]={'class_iou':ci,'image_miou':float(np.mean(ci)),'changed_pixels':int((pred!=reference).sum()),'total_pixels':pred.size}
 assert rows['control']['changed_pixels']==0,'Control is not exact'
 return n,rows
if __name__=='__main__':
 O.mkdir(exist_ok=True)
 for tag in CANDS:(O/'cached'/tag/'predicted_labels').mkdir(parents=True,exist_ok=True)
 names=[r['name'] for r in json.loads((K/'l18h/e2e_valid/release/input_manifest.json').read_text())['images']]
 if (O/'cached/per_image.json').exists():data=json.loads((O/'cached/per_image.json').read_text())
 else:
  with Pool(2) as pool:data=dict(pool.map(task,names))
 (O/'cached/per_image.json').write_text(json.dumps(data,indent=2))
 summary={}
 for tag in CANDS:
  delta={n:100*(data[n][tag]['image_miou']-data[n]['control']['image_miou']) for n in names};cd={n:100*(np.array(data[n][tag]['class_iou'])-np.array(data[n]['control']['class_iou'])) for n in names}
  summary[tag]={'score_percent':float(np.mean([data[n][tag]['image_miou'] for n in names])*100),'delta_pp':float(np.mean(list(delta.values()))),'worst_image':min(delta,key=delta.get),'worst_image_delta_pp':min(delta.values()),'mean_class_delta_pp':np.mean(list(cd.values()),0).tolist(),'worst_image_class_delta_pp':min(float(x.min()) for x in cd.values()),'changed_pixels':sum(data[n][tag]['changed_pixels'] for n in names),'total_pixels':sum(data[n][tag]['total_pixels'] for n in names),'screening_pass':bool(min(delta.values())>=-.10 and np.mean(list(delta.values()))>=-1e-12)}
 (O/'cached/summary.json').write_text(json.dumps(summary,indent=2));(O/'cached/completion.json').write_text(json.dumps({'images':20,'control_exact':True,'cpu_only':True}))
 print(json.dumps(summary,indent=2),flush=True)
