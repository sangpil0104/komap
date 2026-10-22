from pathlib import Path
import sys,json,pickle,importlib.util,hashlib
import numpy as np
K=Path.home()/'komap';P=K/'Model349_lite20_b4ws_bundle/Model349_fusion805';X=P/'Model349_exh14';A=K/'lite15_N3_R326plain3456_adopted_20261013';BASE=80.68255541753359
rt=lambda p:p.astype(np.float16).astype(np.float32)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def expert(parts,b4):return .5*b4+.25*rt((parts['N3']+parts['s1f4'])/2)+.25*rt((parts['s2f1']+parts['s2f4'])/2)
def init_records():
 import torch,yaml
 from PIL import Image
 torch.set_num_threads(2);sys.path[:0]=[str(X),str(X/'src')];import infer_lite
 sp=importlib.util.spec_from_file_location('fu',P/'infer_fusion.py');fu=importlib.util.module_from_spec(sp);sp.loader.exec_module(fu);ref=fu.Residual().eval();ref.load_state_dict(torch.load(K/'weights/all/assets/refiner/epoch8.pt',map_location='cpu',weights_only=False)['state_dict']);cfg=yaml.safe_load((A/'selected_R_lite16_N3.yaml').read_text());F=K/'codex_lite16_N3_L2_fresh_valid_20261013';old=K/'l18h/e2e_valid/release';man=json.load(open(old/'input_manifest.json'));pal=np.array(cfg['palette'],np.uint8);records={}
 for row in man['images']:
  n=row['name'];ip=K/f'data/valid/images/{n}_image.png';assert sha(ip)==row['sha256'];sources={c:np.load(old/f'probabilities/{c}/{n}.npy').astype(np.float32) for c in cfg['inference']['active_components'] if c!='base'};sources['326']=rt(np.load(K/f'codex_r326_ema_seed4_20261012/probabilities/set0/fold2/plain/{n}.npy'))
  with (old/f'probabilities/base/{n}.pkl').open('rb') as f:item=pickle.load(f)
  load=lambda tag:np.load(F/f'fresh_probabilities/{tag}/{n}.npy').astype(np.float32)[[1,2,0,3]];parts={'N3':load('N3'),'s1f4':load('hr_s1_f4'),'s2f1':load('hr_s2_f1'),'s2f4':load('hr_s2_f4')};b4=load('b4ws');e=expert(parts,b4);assert np.array_equal(e,np.load(F/f'N3/expert/{n}.npy'))
  rgb=np.asarray(Image.open(K/f'data/valid/masks/{n}_mask.png').convert('RGB'));gt=np.full(rgb.shape[:2],255,np.uint8)
  for c,col in enumerate(pal):gt[np.all(rgb==col,-1)]=c
  assert (gt<4).all();gray=torch.from_numpy(fu.prepare_gray(np.asarray(Image.open(ip).convert('L'),np.float32)/255)[None,None]);r=infer_lite.predict_fixed(item,sources,cfg);base=np.asarray(Image.open(A/f'valid20/predicted_labels/{n}.png'));rec={'name':n,'sources':sources,'item':item,'r':r,'e':e,'parts':parts,'b4':b4,'gt':gt,'gray':gray,'base':base};records[n]=rec
 G={'records':records,'infer':infer_lite,'fu':fu,'ref':ref,'cfg':cfg,'palette':pal}
 for rec in records.values():assert np.array_equal(final(G,rec,rec['r'],rec['e']),rec['base'])
 assert abs(np.mean([score(rec,rec['base'])['percent'] for rec in records.values()])-BASE)<1e-9;return G

def final(G,rec,r,e):
 import torch
 with torch.no_grad():v,_=G['ref'](torch.from_numpy(np.log(np.maximum(e,1e-7))[None]),rec['gray'])
 ex=G['fu'].labels(e);fl=G['fu'].labels(v.numpy()[0]);return np.where(fl==2,2,np.where(np.isin(ex,[0,2]),ex,r)).astype(np.uint8)
def iou(p,g):return np.array([float(((p==c)&(g==c)).sum()/u) if (u:=((p==c)|(g==c)).sum()) else 0. for c in range(4)])
def score(rec,pred):
 ci=iou(pred,rec['gt']);bc=iou(rec['base'],rec['gt']);return {'percent':100*float(ci.mean()),'delta_pp':100*float((ci-bc).mean()),'class_delta_pp':(100*(ci-bc)).tolist(),'fixed_pixels':int(((rec['base']!=rec['gt'])&(pred==rec['gt'])).sum()),'new_error_pixels':int(((rec['base']==rec['gt'])&(pred!=rec['gt'])).sum())}
def summary(rows):
 ds=[r['delta_pp'] for r in rows.values()];return {'percent':BASE+float(np.mean(ds)),'delta_pp':float(np.mean(ds)),'worst_image':min(rows,key=lambda n:rows[n]['delta_pp']),'worst_image_delta_pp':min(ds),'worst_class_delta_pp':min(min(r['class_delta_pp']) for r in rows.values()),'better':sum(x>0 for x in ds),'worse':sum(x<0 for x in ds),'fixed_pixels':sum(r['fixed_pixels'] for r in rows.values()),'new_error_pixels':sum(r['new_error_pixels'] for r in rows.values()),'pass_mean_local':bool(np.mean(ds)>=0 and min(ds)>=-.1)}
