import os,sys,json,time,hashlib,pickle,inspect,ast,importlib.util,traceback
from pathlib import Path
from multiprocessing import get_context
import numpy as np
from PIL import Image,ImageDraw
import torch
K=Path.home()/'komap';O=K/'codex_lite16_complement_rules_20261012';X=K/'l18h/parent/Model349_exh14';P=K/'Model349_lite20_b4ws_bundle/Model349_fusion805';PE=K/'add/pe/valid';FRESH=K/'codex_lite16_b4_local_20261012/fresh/valid';T=[1,2,0,3]
CAND={'N1':O/'candidates/newmem/N1_hrnet_s3','N2':O/'candidates/newmem/N2_convnext_tiny','N3':O/'candidates/newmem/N3_hrnet_zcdf','N4':O/'candidates/newmem/N4_b4_s4','L2':O/'candidates/eut_spec/L2'}
REMOVE={'s1f2':'b13_s14_s214','s1f4':'b13_s12_s214','s2f1':'b13_s124_s24','s2f4':'b13_s124_s21'}
def softmax(x):
 v=np.exp(x-x.max(0,keepdims=True));return v/v.sum(0,keepdims=True)
def margin(p):v=np.partition(p,-2,axis=0);return v[-1]-v[-2]
def task(n):
 torch.set_num_threads(2);sys.path[:0]=[str(X),str(X/'src')]
 import infer_lite,model335_exhaustive_b4_route_search as route
 spec=importlib.util.spec_from_file_location('fu',P/'infer_fusion.py');fu=importlib.util.module_from_spec(spec);spec.loader.exec_module(fu)
 m=fu.Residual().eval();m.load_state_dict(torch.load(K/'weights/all/assets/refiner/epoch8.pt',map_location='cpu',weights_only=False)['state_dict'])
 def refine(p):
  with torch.no_grad():v,_=m(torch.from_numpy(np.log(np.maximum(p,1e-7))[None]),gray)
  q=softmax(v.numpy()[0]);return q,fu.labels(q)
 man=json.loads((K/'l18h/R_valid/input_manifest.json').read_text());cfg=man['config'];pal=np.array(cfg['palette'],np.uint8)
 ip=K/f'data/valid/images/{n}_image.png';raw=Image.open(ip).convert('RGB');gray=torch.from_numpy(fu.prepare_gray(np.array(raw.convert('L'),dtype=np.float32)/255.)[None,None])
 gt_rgb=np.array(Image.open(K/f'data/valid/masks/{n}_mask.png').convert('RGB'));gt=np.full(gt_rgb.shape[:2],255,np.uint8)
 for c,col in enumerate(pal):gt[np.all(gt_rgb==col,-1)]=c
 assert (gt<4).all()
 r=np.array(Image.open(K/f'l18h/e2e_valid/release/predicted_labels/{n}.png'));e=np.load(FRESH/f'expert/b4_13/{n}.npy').astype(np.float32);ex=fu.labels(e);fq,fl=refine(e);s1=np.where(np.isin(ex,[0,2]),ex,r);base=np.where(fl==2,fl,s1).astype(np.uint8)
 assert np.array_equal(base,np.array(Image.open(FRESH/f'b4_13/fusion/predicted_labels/{n}.png'))),'F control mismatch'
 predictions={'control':base,'no_F':s1};diag={}
 with (K/f'l18h/R_valid/probabilities/base/{n}.pkl').open('rb') as f:item=pickle.load(f)
 sources={c:np.load(K/f'l18h/R_valid/probabilities/{c}/{n}.npy').astype(np.float32) for c in cfg['inference']['active_components'] if c!='base'}
 # Capture the exact route's calibrated logits without editing shared source files.
 original=route.predict;tree=ast.parse(inspect.getsource(original));tree.body[0].name='predict_with_logits'
 tree.body[0].body[-1]=ast.Return(value=ast.Tuple(elts=[ast.Name(id='prediction',ctx=ast.Load()),ast.Name(id='refined',ctx=ast.Load())],ctx=ast.Load()));ast.fix_missing_locations(tree);ns=dict(original.__globals__);exec(compile(tree,'route_capture','exec'),ns)
 captured={}
 def capture(*a,**kw):
  labels,logits=ns['predict_with_logits'](*a,**kw);captured['logits']=logits;return labels
 route.predict=capture
 try:rl=infer_lite.predict_fixed(item,sources,cfg)
 finally:route.predict=original
 diag['R_cached_vs_fresh_pixels']=int((rl!=r).sum());logits=captured['logits'];diag['R_logits_vs_fresh_pixels']=int((logits.argmax(0)!=r).sum())
 # Align spatial cleanup labels with fresh R. These are normalized route scores, not calibrated probabilities.
 disagree=logits.argmax(0)!=r
 for c in range(4):
  mask=disagree&(r==c);logits[c,mask]=logits.max(0)[mask]+1e-4
 rp=softmax(logits);assert np.array_equal(rp.argmax(0),r)
 hr={(s,f):np.load(PE/f'hr392_s{s}_f{f}/{n}.npy').astype(np.float32)[T] for s,fs in [(1,(2,4)),(2,(1,4))] for f in fs};hl={f's{s}f{f}':fu.labels(v) for (s,f),v in hr.items()};b4=np.mean([np.load(PE/f'model329_ws_f{f}/{n}.npy').astype(np.float32)[T] for f in [1,3]],0);bl=fu.labels(b4)
 vote=sum((l==ex).astype(np.uint8) for l in list(hl.values())+[bl]);fvotes=sum((l==2).astype(np.uint8) for l in list(hl.values())+[bl]);em=margin(e);fm=margin(fq)
 hrdiag={}
 for tag,l in hl.items():
  others=[v for k,v in hl.items() if k!=tag];own=l==gt;allwrong=np.logical_and.reduce([v!=gt for v in others]);allright=np.logical_and.reduce([v==gt for v in others]);major=np.logical_or.reduce([v==gt for v in others]);
  reduced=np.array(Image.open(K/f'codex_lite16_E_rescan_20261012/cached/{REMOVE[tag]}/predicted_labels/{n}.png'))
  lost=(base==gt)&(reduced!=gt);gain=(base!=gt)&(reduced==gt)
  hrdiag[tag]={'correct_pixels':int(own.sum()),'unique_correct_all_other_wrong':int((own&allwrong).sum()),'wrong_all_other_correct':int((~own&allright).sum()),'unique_correct_vs_any_other':int((own&~major).sum()),'removal_new_errors':int(lost.sum()),'removal_corrected':int(gain.sum()),'loss_by_gt_class':[int((lost&(gt==c)).sum()) for c in range(4)],'removed_fold_agrees_GT_on_lost':int((lost&own).sum())}
  if n=='A153_step3_sample3' and tag=='s2f1':
   from scipy import ndimage as ndi
   density=ndi.uniform_filter(lost.astype(np.float32),size=170);cy,cx=np.unravel_index(density.argmax(),density.shape);h,w=gt.shape;box=(max(0,min(w-240,int(cx)-120)),max(0,min(h-240,int(cy)-120)));box=(*box,box[0]+240,box[1]+240)
   colors=np.array(raw);colors[lost]=[255,0,0];colors[gain]=[0,255,70]
   ims=[('original',raw),('GT',Image.fromarray(pal[gt])),('lite16',Image.fromarray(pal[base])),('remove s2f1',Image.fromarray(pal[reduced])),('s2f1 alone',Image.fromarray(pal[l])),('s2f4 alone',Image.fromarray(pal[hl['s2f4']])),('red=lost green=gain',Image.fromarray(colors))]
   canvas=Image.new('RGB',(240*7,270),'white');dr=ImageDraw.Draw(canvas)
   for j,(title,im) in enumerate(ims):canvas.paste(im.crop(box),(j*240,28));dr.text((j*240+2,4),title,fill='black')
   canvas.save(O/'A153_hr392.png');hrdiag[tag]['crop']=box
 def gated(eg,fg):return np.where((fl==2)&fg,2,np.where(np.isin(ex,[0,2])&eg,ex,r)).astype(np.uint8)
 for t in [.05,.10,.20]:predictions[f'E_margin_{t}']=gated(em>=t,True)
 for t in [.6,.7,.8]:predictions[f'E_conf_{t}']=gated(e.max(0)>=t,True)
 for t in [2,3]:predictions[f'E_votes_{t}']=gated(vote>=t,True)
 for t in [.02,.05,.10]:predictions[f'F_margin_{t}']=gated(True,fm>=t)
 for t in [.02,.05,.10]:predictions[f'F_Al_gain_{t}']=gated(True,(fq[2]-e[2])>=t)
 for t in [.4,.6]:predictions[f'F_EAl_{t}']=gated(True,e[2]>=t)
 for t in [2,3]:predictions[f'F_votes_{t}']=gated(True,fvotes>=t)
 for t in [.05,.10,.20]:predictions[f'EF_margin_{t}']=gated(em>=t,fm>=t)
 for alpha in [.1,.25,.5]:
  mix=(1-alpha)*e+alpha*rp;ml=fu.labels(mix);stage=np.where(np.isin(ml,[0,2]),ml,r)
  predictions[f'RE_mix_{alpha}_keepF']=np.where(fl==2,2,stage).astype(np.uint8)
  _,mf=refine(mix);predictions[f'RE_mix_{alpha}_refineF']=np.where(mf==2,2,stage).astype(np.uint8)
  cand=fu.labels((1-alpha)*rp+alpha*e);predictions[f'RE_soft_all_E{alpha}']=cand
 # Diagnostic R/E conflict-only mixing produces the same argmax as mixing everywhere when labels agree.
 rejected={}
 for tag,folder in CAND.items():
  with np.load(folder/f'{n}.npz') as z:p=z['v1'].astype(np.float32)
  assert p.shape==e.shape and np.isfinite(p).all() and np.max(np.abs(p.sum(0)-1))<.04
  l=fu.labels(p);fix=(base!=gt)&(l==gt);harm=(base==gt)&(l!=gt)
  rejected[tag]={'single_correct_current_wrong':int(fix.sum()),'single_wrong_current_correct':int(harm.sum()),'by_gt_class_fix':[int((fix&(gt==c)).sum()) for c in range(4)]}
  for alpha in [.1,.25]:
   mix=(1-alpha)*e+alpha*p;ml=fu.labels(mix);_,mf=refine(mix);predictions[f'add_{tag}_E_{alpha}']=np.where(mf==2,2,np.where(np.isin(ml,[0,2]),ml,r)).astype(np.uint8)
  src=dict(sources);src['347']=p;rr=infer_lite.predict_fixed(item,src,cfg);predictions[f'replace_R347_{tag}']=np.where(fl==2,2,np.where(np.isin(ex,[0,2]),ex,rr)).astype(np.uint8)
  # Specialist only changes Eutectic score; useful for sources whose global complementarity is poor.
  mix=e.copy();mix[3]=.75*mix[3]+.25*p[3];mix/=mix.sum(0,keepdims=True);ml=fu.labels(mix);_,mf=refine(mix);predictions[f'add_{tag}_Eut025']=np.where(mf==2,2,np.where(np.isin(ml,[0,2]),ml,r)).astype(np.uint8)
 rows={}
 for tag,pred in predictions.items():
  ci=[]
  for c in range(4):
   u=((pred==c)|(gt==c)).sum();ci.append(float(((pred==c)&(gt==c)).sum()/u) if u else 0.)
  rows[tag]={'class_iou':ci,'miou':float(np.mean(ci)),'corrected_vs_control':int(((base!=gt)&(pred==gt)).sum()),'new_errors_vs_control':int(((base==gt)&(pred!=gt)).sum()),'changed':int((pred!=base).sum())}
 return n,{'candidates':rows,'HR_complement':hrdiag,'rejected_complement':rejected,'R_score_diagnostic':diag}
def main():
 names=[r['name'] for r in json.loads((K/'l18h/e2e_valid/release/input_manifest.json').read_text())['images']]
 with get_context('spawn').Pool(2) as pool:data=dict(pool.map(task,names))
 (O/'per_image.json').write_text(json.dumps(data,indent=2));summary={};base=np.mean([r['candidates']['control']['miou'] for r in data.values()])*100;assert abs(base-80.55464328764197)<1e-10
 for tag in data[names[0]]['candidates']:
  ds={n:100*(data[n]['candidates'][tag]['miou']-data[n]['candidates']['control']['miou']) for n in names};cd=np.array([100*(np.array(data[n]['candidates'][tag]['class_iou'])-data[n]['candidates']['control']['class_iou']) for n in names]);summary[tag]={'percent':base+float(np.mean(list(ds.values()))),'delta_pp':float(np.mean(list(ds.values()))),'worst_pp':min(ds.values()),'worst_image':min(ds,key=ds.get),'worst_class_pp':float(cd.min()),'class_mean_delta_pp':cd.mean(0).tolist(),'better':sum(v>1e-12 for v in ds.values()),'worse':sum(v< -1e-12 for v in ds.values()),'corrected':sum(data[n]['candidates'][tag]['corrected_vs_control'] for n in names),'new_errors':sum(data[n]['candidates'][tag]['new_errors_vs_control'] for n in names),'pass_mean_local_gate':bool(np.mean(list(ds.values()))>=-1e-12 and min(ds.values())>=-.1)}
 # Conditional specimen leave-out re-selection; development diagnostic, not honest nested CV.
 groups=sorted(set(n.split('_')[0] for n in names));select={}
 for family,prefix in [('E_rules','E_'),('F_rules','F_'),('EF_rules','EF_'),('RE_mix','RE_'),('rejected','add_'),('R347_replacement','replace_')]:
  tags=['control']+[t for t in summary if t.startswith(prefix)];held=[];picked=[]
  for group in groups:
   train=[n for n in names if n.split('_')[0]!=group];test=[n for n in names if n.split('_')[0]==group]
   eligible=[t for t in tags if min(100*(data[n]['candidates'][t]['miou']-data[n]['candidates']['control']['miou']) for n in train)>=-.1]
   tag=max(eligible,key=lambda t:np.mean([data[n]['candidates'][t]['miou'] for n in train]));picked.append(tag)
   held.extend(100*(data[n]['candidates'][tag]['miou']-data[n]['candidates']['control']['miou']) for n in test)
  select[family]={'conditional_leave_specimen_out_delta_pp':float(np.mean(held)),'worst_held_pp':min(held),'selected_counts':{t:picked.count(t) for t in set(picked)},'honest_generalization':False}
 (O/'summary.json').write_text(json.dumps({'baseline':base,'candidates':summary,'conditional_selection':select,'class_order':['Al3Ni','Primary Si','Al','Eutectic Si'],'CPU_only':True,'release_replaced':False},indent=2));(O/'completion.json').write_text(json.dumps({'images':20,'candidates':len(summary),'control_exact':True}));print('COMPLETE',len(summary),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:(O/'failure.json').write_text(json.dumps({'trace':traceback.format_exc()}));raise
