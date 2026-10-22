import sys,json,importlib.util
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw
from scipy import ndimage as ndi
K=Path.home()/'komap';O=K/'codex_lite16_complement_rules_20261012';P=K/'Model349_lite20_b4ws_bundle/Model349_fusion805';FRESH=K/'codex_lite16_b4_local_20261012/fresh/valid';PE=K/'add/pe/valid'
spec=importlib.util.spec_from_file_location('fu',P/'infer_fusion.py');fu=importlib.util.module_from_spec(spec);spec.loader.exec_module(fu);torch.set_num_threads(2)
m=fu.Residual().eval();m.load_state_dict(torch.load(K/'weights/all/assets/refiner/epoch8.pt',map_location='cpu',weights_only=False)['state_dict']);man=json.load(open(K/'l18h/e2e_valid/release/input_manifest.json'));pal=np.array(man['config']['palette'],np.uint8);rows={}
for item in man['images']:
 n=item['name'];raw=Image.open(K/f'data/valid/images/{n}_image.png').convert('RGB');rgb=np.array(Image.open(K/f'data/valid/masks/{n}_mask.png').convert('RGB'));gt=np.full(rgb.shape[:2],255,np.uint8)
 for c,col in enumerate(pal):gt[np.all(rgb==col,-1)]=c
 r=np.array(Image.open(K/f'l18h/e2e_valid/release/predicted_labels/{n}.png'));base=np.array(Image.open(FRESH/f'b4_13/fusion/predicted_labels/{n}.png'));e=np.load(FRESH/f'expert/b4_13/{n}.npy').astype(np.float32);gray=torch.from_numpy(fu.prepare_gray(np.array(raw.convert('L'),dtype=np.float32)/255.)[None,None])
 def final(p,votes=False,f_gate=False):
  ex=fu.labels(p)
  with torch.no_grad():v,_=m(torch.from_numpy(np.log(np.maximum(p,1e-7))[None]),gray)
  fl=fu.labels(v.numpy()[0]);gate=True
  if votes:
   ls=[fu.labels(np.load(PE/f'hr392_s{s}_f{f}/{n}.npy').astype(np.float32)[[1,2,0,3]]) for s,f in [(1,2),(1,4),(2,1),(2,4)]]
   b4=np.mean([np.load(PE/f'model329_ws_f{f}/{n}.npy').astype(np.float32)[[1,2,0,3]] for f in [1,3]],0);ls.append(fu.labels(b4));gate=sum((l==ex).astype(np.uint8) for l in ls)>=2
  stage=np.where(np.isin(ex,[0,2])&gate,ex,r)
  return np.where((fl==2)&((p[2]>=.4) if f_gate else True),2,stage).astype(np.uint8)
 preds={'control':final(e)};assert np.array_equal(preds['control'],base)
 for tag,path in [('N3','newmem/N3_hrnet_zcdf'),('L2','eut_spec/L2')]:
  with np.load(O/f'candidates/{path}/{n}.npz') as z:q=z['v1'].astype(np.float32)
  for ss,ff in [(1,2),(1,4),(2,1),(2,4)]:
   old=np.load(PE/f'hr392_s{ss}_f{ff}/{n}.npy').astype(np.float32)[[1,2,0,3]]
   mixed=np.maximum(e+.125*(q-old),1e-7);mixed/=mixed.sum(0,keepdims=True)
   preds[f'{tag}_replace_s{ss}f{ff}']=final(mixed)
 rows[n]={}
 for tag,pred in preds.items():
  dest=O/f'replacement/{tag}/predicted_labels';dest.mkdir(parents=True,exist_ok=True);Image.fromarray(pred).save(dest/f'{n}.png')
  ci=[]
  for c in range(4):
   u=((pred==c)|(gt==c)).sum();ci.append(float(((pred==c)&(gt==c)).sum()/u) if u else 0.)
  rows[n][tag]={'miou':float(np.mean(ci)),'class_iou':ci,'corrected':int(((base!=gt)&(pred==gt)).sum()),'new_errors':int(((base==gt)&(pred!=gt)).sum())}
 if False:
  pred=preds['N3'];loss=(base==gt)&(pred!=gt);gain=(base!=gt)&(pred==gt);den=ndi.uniform_filter(loss.astype(np.float32),size=180);cy,cx=np.unravel_index(den.argmax(),den.shape);h,w=gt.shape;l=max(0,min(w-256,int(cx)-128));t=max(0,min(h-256,int(cy)-128));box=(l,t,l+256,t+256)
  im=np.array(raw);im[loss]=[255,30,30];im[gain]=[0,255,70];canvas=Image.new('RGB',(256*6,284),'white');dr=ImageDraw.Draw(canvas)
  for j,(title,img) in enumerate([('original',raw),('GT',Image.fromarray(pal[gt])),('lite16',Image.fromarray(pal[base])),('N3 E10%',Image.fromarray(pal[pred])),('L2 E10%',Image.fromarray(pal[preds['L2']])),('red=new green=fix',Image.fromarray(im))]):canvas.paste(img.crop(box),(j*256,28));dr.text((j*256+2,5),title,fill='black')
  canvas.save(O/f'{n}_N3.png')
summary={};base_score=np.mean([r['control']['miou'] for r in rows.values()])*100
for tag in next(iter(rows.values())):
 ds={n:100*(v[tag]['miou']-v['control']['miou']) for n,v in rows.items()};summary[tag]={'percent':base_score+float(np.mean(list(ds.values()))),'delta_pp':float(np.mean(list(ds.values()))),'worst_pp':min(ds.values()),'worst_image':min(ds,key=ds.get),'better':sum(v>0 for v in ds.values()),'worse':sum(v<0 for v in ds.values())}
(O/'replacement_per_image.json').write_text(json.dumps(rows,indent=2));(O/'replacement_summary.json').write_text(json.dumps({'posthoc_combination':True,'summary':summary},indent=2));print(json.dumps(summary),flush=True)
