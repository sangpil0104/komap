from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage
import numpy as np,json,csv,hashlib
W=Path.cwd();O=Path(__file__).resolve().parent;ref=W/'valid/images/A71_step2_sample1_image.png';test=sorted((W/'compare_web/assets/original').glob('*_image.png'))
def feature(path):
 raw=np.array(Image.open(path).convert('L'));a=np.asarray(Image.open(path).convert('L').resize((675,480),Image.Resampling.LANCZOS)).astype(float);z=(a-a.mean())/(a.std()+1e-8);hist=np.bincount(raw.reshape(-1),minlength=256).astype(float);hist/=hist.sum();grad=np.hypot(*np.gradient(z));gh,_=np.histogram(grad,bins=[0,.05,.1,.2,.4,.8,1.6,3.2,np.inf]);gh=gh/gh.sum()
 center=a[1:-1,1:-1];neighbors=[a[:-2,:-2],a[:-2,1:-1],a[:-2,2:],a[1:-1,2:],a[2:,2:],a[2:,1:-1],a[2:,:-2],a[1:-1,:-2]];bits=np.stack([n>=center for n in neighbors]);trans=(bits!=np.roll(bits,1,axis=0)).sum(0);lbp=np.where(trans<=2,bits.sum(0),9);lh=np.bincount(lbp.reshape(-1),minlength=10);lh=lh/lh.sum()
 dark=a<30;lab,k=ndimage.label(dark);sizes=np.bincount(lab.ravel())[1:];large=sizes[sizes>=.002*a.size];return {'mean':float(raw.mean()),'std':float(raw.std()),'size':[int(raw.shape[1]),int(raw.shape[0])],'dark_below30_fraction':float(dark.mean()),'largest_dark_component_fraction':float(sizes.max()/a.size) if len(sizes) else 0.,'large_dark_components':int(len(large)),'large_dark_component_fraction':float(large.sum()/a.size),'hist':hist,'grad_hist':gh,'lbp_hist':lh}
def js(p,q):
 m=(p+q)/2
 def kl(a,b):
  ok=a>0;return float(np.sum(a[ok]*np.log2(a[ok]/b[ok])))
 return float(np.sqrt((kl(p,m)+kl(q,m))/2))
f=feature(ref);rows=[]
for p in test:
 v=feature(p);r={'image':p.stem.removesuffix('_image'),'path':str(p),'brightness_CDF_distance_gray_units':float(np.abs(np.cumsum(f['hist'])-np.cumsum(v['hist'])).sum()),'normalized_gradient_JS_distance':js(f['grad_hist'],v['grad_hist']),'LBP_JS_distance':js(f['lbp_hist'],v['lbp_hist']),**{k:val for k,val in v.items() if not isinstance(val,np.ndarray)}};rows.append(r)
for key in ['brightness_CDF_distance_gray_units','normalized_gradient_JS_distance','LBP_JS_distance']:
 for i,r in enumerate(sorted(rows,key=lambda v:v[key])):r[key+'_rank']=i+1
for r in rows:r['mean_feature_rank']=float(np.mean([r[k+'_rank'] for k in ['brightness_CDF_distance_gray_units','normalized_gradient_JS_distance','LBP_JS_distance']]))
rows.sort(key=lambda r:r['mean_feature_rank']);result={'reference':{k:v for k,v in f.items() if not isinstance(v,np.ndarray)},'ranked_test_images':rows,'method':'exploratory equal average of ranks from grayscale CDF distance, standardized gradient histogram JS distance, and rotation-invariant uniform LBP histogram JS distance; texture/dark components measured after width675 height480 resize','not_learned_embedding':True,'not_test_accuracy':True,'dark_component_not_semantic_class_label':True,'input_SHA256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ref,*test]}}
(O/'similarity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
paths=[ref,*[Path(r['path']) for r in rows[:3]]];pw=675;ph=480;im=Image.new('RGB',(2*(pw+30),2*(ph+80)), 'white');d=ImageDraw.Draw(im);font=ImageFont.truetype('/System/Library/Fonts/AppleSDGothicNeo.ttc',28,index=6)
for i,p in enumerate(paths):
 x=(i%2)*(pw+30);y=(i//2)*(ph+80);d.text((x+12,y+12),('기준: ' if i==0 else '')+p.stem.removesuffix('_image'),font=font,fill='#1b293c');im.paste(Image.open(p).convert('RGB').resize((pw,ph),Image.Resampling.LANCZOS),(x+12,y+60))
im.save(O/'A71_closest_test_inputs.png')
