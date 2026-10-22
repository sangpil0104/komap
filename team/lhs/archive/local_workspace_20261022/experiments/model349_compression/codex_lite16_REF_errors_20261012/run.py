from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage as ndi
K=Path.home()/'komap';O=K/'codex_lite16_REF_errors_20261012';R=K/'l18h/e2e_valid/release';E=K/'codex_lite16_b4_local_20261012/fresh/valid/expert/b4_13';F=K/'codex_lite16_b4_local_20261012/fresh/valid/b4_13/fusion/predicted_labels'
(O/'figures').mkdir(parents=True,exist_ok=True)
man=json.loads((R/'input_manifest.json').read_text());pal=np.array(man['config']['palette'],np.uint8);rows={}
def metrics(p,g):
 ci=[]
 for c in range(4):
  u=((p==c)|(g==c)).sum();ci.append(float(((p==c)&(g==c)).sum()/u) if u else 0.)
 return {'class_iou':ci,'miou':float(np.mean(ci)),'error_pixels':int((p!=g).sum())}
for item in man['images']:
 n=item['name'];ip=K/f'data/valid/images/{n}_image.png';assert hashlib.sha256(ip.read_bytes()).hexdigest()==item['sha256']
 raw=Image.open(ip).convert('RGB');gtrgb=np.array(Image.open(K/f'data/valid/masks/{n}_mask.png').convert('RGB'));gt=np.full(gtrgb.shape[:2],255,np.uint8)
 for c,col in enumerate(pal):gt[np.all(gtrgb==col,-1)]=c
 assert (gt<4).all()
 r=np.array(Image.open(R/f'predicted_labels/{n}.png'));ep=np.load(E/f'{n}.npy');ex=np.array([2,0,1,3],np.uint8)[ep[[2,0,1,3]].argmax(0)];e=np.where(np.isin(ex,[0,2]),ex,r).astype(np.uint8);f=np.array(Image.open(F/f'{n}.png'));stages={'R':r,'E':e,'F':f}
 assert all(v.shape==gt.shape and (v<4).all() for v in stages.values())
 boundary=(gt!=ndi.minimum_filter(gt,3))|(gt!=ndi.maximum_filter(gt,3));near=ndi.maximum_filter(boundary,5)
 row={'stages':{k:metrics(p,gt) for k,p in stages.items()},'transitions':{}}
 for a,b in [('R','E'),('E','F')]:
  x,y=stages[a],stages[b];gain=(x!=gt)&(y==gt);loss=(x==gt)&(y!=gt);other=(x!=gt)&(y!=gt)&(x!=y)
  mat=np.zeros((4,4,4),np.int64);np.add.at(mat,(gt[x!=y],x[x!=y],y[x!=y]),1)
  row['transitions'][a+'->'+b]={'corrected':int(gain.sum()),'new_errors':int(loss.sum()),'wrong_to_other_wrong':int(other.sum()),'changed':int((x!=y).sum()),'by_gt_class':[{'corrected':int((gain&(gt==c)).sum()),'new_errors':int((loss&(gt==c)).sum()),'wrong_to_other_wrong':int((other&(gt==c)).sum())} for c in range(4)],'corrected_near_boundary':int((gain&near).sum()),'new_errors_near_boundary':int((loss&near).sum()),'gt_old_new':mat.tolist(),'delta_pp':100*(row['stages'][b]['miou']-row['stages'][a]['miou'])}
  # The densest new-error area; alongside fixes so visual comparison cannot hide tradeoffs.
  density=ndi.uniform_filter(loss.astype(np.float32),size=180);cy,cx=np.unravel_index(density.argmax(),density.shape);h,w=gt.shape;side=256;l=max(0,min(w-side,int(cx)-side//2));t=max(0,min(h-side,int(cy)-side//2));box=(l,t,l+side,t+side)
  colors=np.array(raw);colors[loss]=[255,30,30];colors[gain]=[0,255,80];colors[other]=[255,0,255]
  images=[('Original',raw),('GT',Image.fromarray(pal[gt]))]+[(s,Image.fromarray(pal[p])) for s,p in stages.items()]+[('red=new green=fix',Image.fromarray(colors))]
  canvas=Image.new('RGB',(6*256,285),'white');dr=ImageDraw.Draw(canvas)
  for j,(title,im) in enumerate(images):canvas.paste(im.crop(box),(j*256,28));dr.text((j*256+3,5),title,fill='black')
  canvas.save(O/f'figures/{n}_{a}{b}.png');row['transitions'][a+'->'+b]['crop_box']=box
 rows[n]=row
summary={'images':20,'palette':pal.tolist(),'class_names':['Al3Ni','Primary Si','Al','Eutectic Si'],'stages':{},'transitions':{},'valid_development_diagnostic':True,'GPU_rerun':False}
for s in ['R','E','F']:summary['stages'][s]={'miou_percent':float(np.mean([r['stages'][s]['miou'] for r in rows.values()])*100),'mean_class_iou_percent':(np.mean([r['stages'][s]['class_iou'] for r in rows.values()],axis=0)*100).tolist(),'error_pixels':sum(r['stages'][s]['error_pixels'] for r in rows.values())}
assert abs(summary['stages']['F']['miou_percent']-80.55464328764197)<1e-10
for tag in ['R->E','E->F']:
 rs=[r['transitions'][tag] for r in rows.values()];total={k:sum(r[k] for r in rs) for k in ['corrected','new_errors','wrong_to_other_wrong','changed','corrected_near_boundary','new_errors_near_boundary']}
 total['by_gt_class']=[{k:sum(r['by_gt_class'][c][k] for r in rs) for k in ['corrected','new_errors','wrong_to_other_wrong']} for c in range(4)]
 total['gt_old_new']=np.sum([r['gt_old_new'] for r in rs],axis=0).tolist();total['worst_images']=sorted([{'image':n,**r['transitions'][tag]} for n,r in rows.items()],key=lambda r:r['delta_pp'])[:5];summary['transitions'][tag]=total
(O/'per_image.json').write_text(json.dumps(rows,indent=2));(O/'summary.json').write_text(json.dumps(summary,indent=2));(O/'completion.json').write_text(json.dumps({'control_final_exact_score':True,'images':20,'cpu_only':True}));print(json.dumps(summary,indent=2),flush=True)
