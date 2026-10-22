from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage as ndi
O=Path(__file__).resolve().parent;D=O/'data';V={'lite20':D/'p100/f20_test/predicted_labels','lite19a':D/'ana/t10/l19a_EF/fusion/predicted_labels','lite18':D/'p100/test18/f18_test/predicted_labels','lite18h':D/'ana/l18h/e2e_test/fusion/predicted_labels','lite16':D/'ana/codex_lite16_b4_local_20261012/fresh/test/b4_13/fusion/predicted_labels'}
pal=np.array([[25,76,153],[76,178,76],[153,127,76],[204,204,204]],np.uint8);classes=['Al3Ni','Primary Si','Al','Eutectic Si']
# Team palette order, confirmed from model source; record below.
names=[x['name'] for x in json.load(open(D/'ana/l18h/e2e_test/release/input_manifest.json'))['images']]
font=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',13)
manifest=json.load(open(D/'lite20_input_manifest.json'));assert {r['name']:r['sha256'] for r in manifest['images']}=={r['name']:r['sha256'] for r in json.load(open(D/'ana/l18h/e2e_test/release/input_manifest.json'))['images']}
pairs=[('lite20','lite19a'),('lite19a','lite18'),('lite19a','lite18h'),('lite18h','lite16'),('lite20','lite16')];stats={f'{a}->{b}':[] for a,b in pairs};(O/'figures').mkdir(exist_ok=True)
def tile(im,size=(260,200)):
 im=im.copy();im.thumbnail(size);t=Image.new('RGB',size,'white');t.paste(im,((size[0]-im.width)//2,(size[1]-im.height)//2));return t
for n in names:
 ip=D/f'ana/data/test/images/{n}_image.png';assert hashlib.sha256(ip.read_bytes()).hexdigest()==next(r['sha256'] for r in manifest['images'] if r['name']==n)
 raw=Image.open(ip).convert('RGB');pred={k:np.array(Image.open(p/n.__add__('.png'))) for k,p in V.items()};assert all(x.ndim==2 and x.shape==pred['lite20'].shape and x.max()<4 for x in pred.values())
 for a,b in pairs:
  x,y=pred[a],pred[b];diff=x!=y;trans=np.bincount((4*x+y)[diff],minlength=16).reshape(4,4)
  lab,num=ndi.label(diff);areas=np.bincount(lab.ravel())[1:];near=ndi.maximum_filter((x!=ndi.maximum_filter(x,3))|(x!=ndi.minimum_filter(x,3)),size=5)
  stats[f'{a}->{b}'].append({'image':n,'pixels':int(diff.sum()),'fraction_percent':float(diff.mean()*100),'total':diff.size,'transitions':trans.tolist(),'largest_change_component':int(areas.max()) if num else 0,'near_old_boundary_percent':float(near[diff].mean()*100) if diff.any() else None,'class_area_delta_px':[(int((y==c).sum())-int((x==c).sum())) for c in range(4)]})
 board=Image.new('RGB',(260*7,245),'white');dr=ImageDraw.Draw(board)
 for j,(tag,img) in enumerate([('original',raw)]+[(k,Image.fromarray(pal[pred[k]])) for k in V]+[('20->16 changed',Image.fromarray(np.where((pred['lite20']!=pred['lite16'])[...,None],np.array([255,0,255]),np.array(raw)).astype(np.uint8)))]):
  board.paste(tile(img),(j*260,35));dr.text((j*260+5,5),tag,font=font,fill='black')
 board.save(O/f'figures/{n}_overview.png')
 # Up to four spatial hotspots, selected by changed pixel density; compare every version.
 diff=pred['lite20']!=pred['lite16'];remain=diff.copy();boxes=[];h,w=diff.shape
 for _ in range(4):
  if not remain.any():break
  density=ndi.uniform_filter(remain.astype(np.float32),size=160);cy,cx=np.unravel_index(np.argmax(density),density.shape);size=220
  l=max(0,min(w-size,cx-size//2));t=max(0,min(h-size,cy-size//2));boxes.append((l,t,l+size,t+size));remain[max(0,t-50):min(h,t+size+50),max(0,l-50):min(w,l+size+50)]=False
 cb=Image.new('RGB',(220*7,260*len(boxes)),'white');draw=ImageDraw.Draw(cb)
 for row,box in enumerate(boxes):
  ims=[raw]+[Image.fromarray(pal[pred[k]]) for k in V]+[Image.fromarray(np.where(diff[...,None],np.array([255,0,255]),np.array(raw)).astype(np.uint8))]
  for j,img in enumerate(ims):cb.paste(img.crop(box),(j*220,row*260+35));draw.text((j*220+3,row*260+3),(['original']+list(V)+['20->16 change'])[j],font=font,fill='black')
  draw.text((3,row*260+20),str(box),font=font,fill='black')
 if boxes:cb.save(O/f'figures/{n}_hotspots.png')
summary={k:{'changed_pixels':sum(r['pixels'] for r in rows),'total_pixels':sum(r['total'] for r in rows),'changed_percent':100*sum(r['pixels'] for r in rows)/sum(r['total'] for r in rows),'transitions':np.sum([r['transitions'] for r in rows],axis=0).tolist(),'images':sorted(rows,key=lambda r:r['pixels'],reverse=True)} for k,rows in stats.items()}
(O/'comparison.json').write_text(json.dumps({'versions':{k:str(v) for k,v in V.items()},'input_sha_match':True,'palette':pal.tolist(),'comparisons':summary},indent=2))
for k,v in summary.items():print(k,v['changed_pixels'],round(v['changed_percent'],4),[(r['image'].split('_')[0],r['pixels'],round(r['fraction_percent'],3)) for r in v['images'][:4]])
