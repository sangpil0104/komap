import sys,json,pickle,hashlib,importlib.util,copy,time
from pathlib import Path
import numpy as np,torch,yaml
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage as ndi
D=Path(__file__).resolve().parent;K=Path('/home/sangpil/komap');L=K/'lite14_N3_R326plain3456_HRtrain70seed5_20261013';P13=K/'lite13_R322share326_20261014';X=L/'parent_package/Model349_exh14'
sys.path[:0]=[str(X),str(X/'src')]
import model335_exhaustive_b4_route_search as route
spec=importlib.util.spec_from_file_location('fu',L/'parent_package/infer_fusion.py');fu=importlib.util.module_from_spec(spec);spec.loader.exec_module(fu)
torch.set_num_threads(2);ref=fu.Residual().eval();ref.load_state_dict(torch.load(L/'assets/refiner/epoch8.pt',map_location='cpu',weights_only=False)['state_dict'])
rt=lambda p:p.astype(np.float16).astype(np.float32)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
retune=K/'b4tta/search13/retune_full.json';selected=json.loads(retune.read_text())['state'];cfg=yaml.safe_load((P13/'selected_R_lite13.yaml').read_text());newcfg=copy.deepcopy(cfg);newcfg['route']['base'].update(selected['R'])
for sub in ('original_labels','retuned_labels','figures'): (D/sub).mkdir(exist_ok=True)
(D/'frozen_retune.json').write_text(json.dumps({'selected':selected,'source_sha':sha(retune),'original_config_sha':sha(P13/'selected_R_lite13.yaml')},indent=2))
pal=np.array(cfg['palette'],dtype=np.uint8);cl=['Al3Ni','Primary Si','Al','Eutectic Si'];R=K/'l18h/R_test/probabilities';oldparts=K/'codex_lite16_N3_L2_test_20261012/probabilities';newparts=K/'codex_N3_L2_seed_retrained_test_20261013/probabilities'
def ld(p):return np.load(p).astype(np.float32)
def predict_r(item,src,cfg):
 s=dict(src)
 for slot,donor in cfg['inference'].get('hr_alias',{}).items():s[slot]=s[donor]
 base=item['preaux'] if 'preaux' in item else route.undo_model181_primary_blend(item);st=cfg['route']
 for key in ('primary','al3ni','eutectic'):
  if not st['base'][key+'_gamma']:s.setdefault(st['base'][key+'_source'],base)
 mixed=route.make_mixed({'item':item,'base':base,'sources':s},st['base'])
 for comp,key in (('329','gamma329'),('336','gamma336'),('347','gamma347')):
  for c,g in enumerate(st[key]):
   if g:mixed=route.replace_classes(mixed,s[comp],(c,),g)
 return route.predict(item,mixed,st['base'],route.bilateral_weights(item['z']),exact=True)
def final(gray,r,e):
 with torch.no_grad():v,_=ref(torch.from_numpy(np.log(np.maximum(e,1e-7))[None]),gray)
 ex=fu.labels(e);fl=fu.labels(v.numpy()[0]);return np.where(fl==2,2,np.where(np.isin(ex,[0,2]),ex,r)).astype(np.uint8)
def edge(a):
 e=np.zeros(a.shape,bool);e[1:]|=a[1:]!=a[:-1];e[:-1]|=a[1:]!=a[:-1];e[:,1:]|=a[:,1:]!=a[:,:-1];e[:,:-1]|=a[:,1:]!=a[:,:-1];return e
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
def figure(name,raw,before,after,changed,box=None):
 maps=[raw,Image.fromarray(pal[before]),Image.fromarray(pal[after]),raw.copy()]
 z=np.asarray(raw).copy();z[changed]=[255,30,190];maps[3]=Image.fromarray(z)
 if box:maps=[m.crop(box) for m in maps]
 width=430;h=round(maps[0].height*width/maps[0].width);canvas=Image.new('RGB',(width*4,h+68),'white');dr=ImageDraw.Draw(canvas)
 for j,(m,title) in enumerate(zip(maps,['Input','lite13','Retuned','Changed (magenta)'])):
  canvas.paste(m.resize((width,h)),(j*width,60));dr.text((j*width+12,32),title,font=font,fill='black')
 dr.text((12,5),name+('  crop '+str(box) if box else ''),font=font,fill='black');return canvas
expected=json.loads((L/'fresh_test/expert/completion.json').read_text())['input_image_sha256'];rows=[];trans=np.zeros((4,4),np.int64);npx=0;changedtotal=0;near=0;exact=0;precision=[]
for image in sorted((K/'data/test/images').glob('*_image.png')):
 n=image.name.removesuffix('_image.png');assert sha(image)==expected[str(image)],n
 raw=Image.open(image).convert('RGB');gray=torch.from_numpy(fu.prepare_gray(np.asarray(raw.convert('L'),np.float32)/255)[None,None]);src={c:ld(R/f'{c}/{n}.npy') for c in ('269','329','347')};src['326']=rt(ld(K/f'codex_R326_plain_EMA_test10_20261013/probabilities/plain3456/{n}.npy'));item=pickle.loads((R/f'base/{n}.pkl').read_bytes());e=ld(L/f'fresh_test/expert/{n}.npy')
 orig=final(gray,predict_r(item,src,cfg),e);reference=np.asarray(Image.open(P13/f'e2e_test/fusion/predicted_labels/{n}.png'));assert np.array_equal(orig,reference),(n,int((orig!=reference).sum()));exact+=1
 # Recover original B4 family from its exact previous E decomposition; raw HR caches are FP32.
 oldhr={tag:ld(oldparts/f'{tag}/{n}.npy')[[1,2,0,3]] for tag in ('hr_s1_f2','hr_s1_f4','hr_s2_f1','hr_s2_f4')}
 eold=ld(K/f'codex_lite16_b4_local_20261012/fresh/test/expert/b4_13/{n}.npy');boldexact=(eold-.25*rt((oldhr['hr_s1_f2']+oldhr['hr_s1_f4'])/2)-.25*rt((oldhr['hr_s2_f1']+oldhr['hr_s2_f4'])/2))/.5
 n3=rt(ld(newparts/f'N3/{n}.npy'))[[1,2,0,3]];h1=ld(oldparts/f'hr_s1_f4/{n}.npy')[[1,2,0,3]];g1=rt((n3+h1)/2);g2=rt((e-.5*boldexact-.25*g1)/.25)
 reconstructed=.5*boldexact+.25*g1+.25*g2;maxabs=float(np.max(np.abs(reconstructed-e)));assert maxabs<=1e-7,(n,maxabs)
 # Match retune_full's selected cache precision, retaining exact original E as anchor.
 bscreen=rt((ld(K/f'b4tta/probs_test/E/f1/d4/{n}.npy')+ld(K/f'b4tta/probs_test/E/f3/d4/{n}.npy'))/2)[[1,2,0,3]]
 precision.append({'image':n,'original_E_reconstruction_maxabs':maxabs,'B4_cache_vs_exact_maxabs':float(np.max(np.abs(bscreen-boldexact)))})
 st=selected['E'];hr=1-st['wb4'];enew=e+(st['wb4']-.5)*bscreen+(hr*st['g1share']-.25)*g1+(hr*(1-st['g1share'])-.25)*g2
 anew=final(gray,predict_r(item,src,newcfg),enew);ch=orig!=anew;count=int(ch.sum());npx+=ch.size;changedtotal+=count;tr=np.zeros((4,4),np.int64);np.add.at(tr,(orig.ravel(),anew.ravel()),1);trans+=tr;dist=ndi.distance_transform_edt(~edge(orig));nnear=int((ch&(dist<=2)).sum());near+=nnear
 labels,ncomponents=ndi.label(ch);areas=np.bincount(labels.ravel())[1:];before=np.bincount(orig.ravel(),minlength=4);after=np.bincount(anew.ravel(),minlength=4)
 row={'image':n,'pixels':ch.size,'changed_pixels':count,'changed_percent':100*count/ch.size,'within_2px_old_boundary':nnear,'within_2px_share_percent':100*nnear/count if count else 100,'components':ncomponents,'largest_changed_component_pixels':int(areas.max()) if areas.size else 0,'class_area_delta_pp':{cl[i]:100*int(after[i]-before[i])/ch.size for i in range(4)},'class_transition_pixels':tr.tolist()};rows.append(row)
 Image.fromarray(orig).save(D/f'original_labels/{n}.png');Image.fromarray(anew).save(D/f'retuned_labels/{n}.png');figure(n,raw,orig,anew,ch).save(D/f'figures/{n}_full.png')
 den=ndi.uniform_filter(ch.astype(np.float32),160);cy,cx=np.unravel_index(den.argmax(),ch.shape);sz=384;x=max(0,min(ch.shape[1]-sz,int(cx)-sz//2));y=max(0,min(ch.shape[0]-sz,int(cy)-sz//2));box=(x,y,min(x+sz,ch.shape[1]),min(y+sz,ch.shape[0]));figure(n,raw,orig,anew,ch,box).save(D/f'figures/{n}_crop.png');row['crop_box']=box
 print(n,count,row['changed_percent'],flush=True)
assert all(sha(p)==h for p,h in expected.items())
report={'baseline':'lite13','retuned':'frozen retune_full selected on valid20','baseline_test_masks_exact':exact,'test_gt_used':False,'input_sha_unchanged':True,'weights_per_model':13,'total_pixels':npx,'changed_pixels':changedtotal,'changed_percent':100*changedtotal/npx,'within_2px_old_boundary_share_percent':100*near/changedtotal,'classes':cl,'class_transition_pixels':trans.tolist(),'rows':sorted(rows,key=lambda r:-r['changed_pixels']),'precision_checks':precision,'release_replaced':False}
(D/'comparison.json').write_text(json.dumps(report,indent=2));(D/'completion.json').write_text(json.dumps({'finished':time.time(),'images':10,'baseline_exact':10,'test_gt_used':False}));print('COMPLETE',report['changed_percent'],flush=True)
