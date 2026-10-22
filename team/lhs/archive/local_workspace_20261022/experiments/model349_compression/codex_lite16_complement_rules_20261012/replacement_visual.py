from pathlib import Path
import numpy as np,json
from PIL import Image,ImageDraw
from scipy import ndimage as ndi
K=Path.home()/'komap';O=K/'codex_lite16_complement_rules_20261012';pal=np.array([[25,76,153],[76,178,76],[153,127,76],[204,204,204]],np.uint8)
for n in ['A72_step4_sample4','A153_step3_sample3','A71_step2_sample1']:
 raw=Image.open(K/f'data/valid/images/{n}_image.png').convert('RGB');rgb=np.array(Image.open(K/f'data/valid/masks/{n}_mask.png').convert('RGB'));gt=np.full(rgb.shape[:2],255,np.uint8)
 for c,col in enumerate(pal):gt[np.all(rgb==col,-1)]=c
 base=np.array(Image.open(K/f'codex_lite16_b4_local_20261012/fresh/valid/b4_13/fusion/predicted_labels/{n}.png'));new=np.array(Image.open(O/f'replacement/N3_replace_s1f2/predicted_labels/{n}.png'));loss=(base==gt)&(new!=gt);gain=(base!=gt)&(new==gt);canvas=Image.new('RGB',(256*5,285*2),'white');dr=ImageDraw.Draw(canvas)
 for row,mask in enumerate([loss,gain]):
  density=ndi.uniform_filter(mask.astype(np.float32),size=180);cy,cx=np.unravel_index(density.argmax(),density.shape);h,w=gt.shape;l=max(0,min(w-256,int(cx)-128));t=max(0,min(h-256,int(cy)-128));box=(l,t,l+256,t+256);colors=np.array(raw);colors[loss]=[255,0,0];colors[gain]=[0,255,70]
  for j,(tag,im) in enumerate([('original',raw),('GT',Image.fromarray(pal[gt])),('lite16',Image.fromarray(pal[base])),('replace s1f2 with N3',Image.fromarray(pal[new])),('red=new green=fix',Image.fromarray(colors))]):canvas.paste(im.crop(box),(j*256,row*285+28));dr.text((j*256+2,row*285+4),tag,fill='black')
 canvas.save(O/f'{n}_replacement.png')
