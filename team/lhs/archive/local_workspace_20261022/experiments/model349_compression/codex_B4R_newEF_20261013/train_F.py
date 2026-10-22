import os,json,time,hashlib,traceback
from pathlib import Path
import numpy as np
import cv2
O=Path('/workspace/model349_compress/codex_B4R_newEF_20261013');D=Path('/workspace/komap_model1')
def dump(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=1))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def features(p,gray):
 p=np.maximum(p,1e-7);log=np.log(p);local=np.stack([cv2.blur(x,(3,3)) for x in p]);z=(gray-gray.mean())/(gray.std()+1e-6);grad=np.sqrt(cv2.Sobel(gray,cv2.CV_32F,1,0)**2+cv2.Sobel(gray,cv2.CV_32F,0,1)**2);ent=-(p*log).sum(0)/np.log(4);top=np.partition(p,-2,axis=0)[-2:];return np.concatenate([log,local,np.clip(z/4,-1,1)[None],grad[None],ent[None],(top[1]-top[0])[None]],0).astype(np.float32)
def main():
 if (O/'F_training_completion.json').exists():return
 import torch
 from PIL import Image
 torch.set_num_threads(2);cv2.setNumThreads(2);torch.manual_seed(819349);rng=np.random.default_rng(819349)
 audit=json.load(open('/workspace/model349_compress/accuracy81_oof_refiner_20261009/oof_input_audit.json'));assert audit['images']==70 and not audit['official_valid_or_test_inputs_opened']
 lock=json.load(open(D/'experiments/results/model397_fixed_b4_model392_blend_lock.json'));assert sha(D/'experiments/results/model397_fixed_b4_model392_blend_lock.json')==audit['model397_lock_sha256']
 teacher=json.load(open('/workspace/model349_compress/accuracy81_residual_20261008/runs/teacher_train70/input_manifest.json'));images={x['name']:x for x in teacher['images']};pal=np.array(teacher['config']['palette'],np.uint8);xs=[];ys=[];receipt=[];source_sha={}
 for fold in audit['folds']:
  key=next(k for k in fold['probability_archive_sha256'] if 'model329_' in k);path=D/key;assert sha(path)==fold['probability_archive_sha256'][key];source_sha[str(path)]=sha(path)
  with np.load(path,allow_pickle=False) as z:
   for name in sorted(z.files):
    ip=D/f'data/train/images/{name}_image.png';mp=D/f'data/train/masks/{name}_mask.png';assert sha(ip)==images[name]['sha256'];assert sha(mp)==lock['locked_files'][str(mp.relative_to(D))]['sha256']
    p=z[name].astype(np.float32)[[1,2,0,3]];gray=np.array(Image.open(ip).convert('L'),np.float32)/255;rgb=np.array(Image.open(mp).convert('RGB'));gt=np.full(gray.shape,255,np.uint8)
    for c,col in enumerate(pal):gt[np.all(rgb==col,-1)]=c
    assert p.shape==(4,*gt.shape) and (gt<4).all() and np.isfinite(p).all()
    f=features(p,gray).reshape(12,-1);ids=rng.choice(gt.size,min(6000,gt.size),replace=False);xs.append(f[:,ids].T.copy());ys.append(gt.reshape(-1)[ids].copy());receipt.append({'name':name,'held_fold':fold['fold'],'source_archive':str(path),'input_sha256':sha(ip),'GT_sha256':sha(mp)})
    dump('F_progress.json',{'stage':'OOF features','images':len(receipt),'total':70,'time':time.time(),'pid':os.getpid()})
 assert {x['name'] for x in receipt}==set(audit['all_image_ids']) and len(receipt)==70
 x=torch.from_numpy(np.concatenate(xs));y=torch.from_numpy(np.concatenate(ys).astype(np.int64));counts=torch.bincount(y,minlength=4).float();weights=(counts.sum()/counts).sqrt();weights/=weights.mean();init=torch.zeros(4,12);init[:,:4]=torch.eye(4);stats={}
 for tag,w in [('plain',None),('balanced',weights)]:
  model=torch.nn.Linear(12,4)
  with torch.no_grad():model.weight.copy_(init);model.bias.zero_()
  opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.001);losses=[]
  for epoch in range(1,13):
   idx=torch.randperm(len(y));total=0.
   for batch in idx.split(4096):
    opt.zero_grad(set_to_none=True);pred=model(x[batch]);loss=torch.nn.functional.cross_entropy(pred,y[batch],weight=w)+.005*((model.weight-init).square().mean()+model.bias.square().mean());loss.backward();opt.step();total+=float(loss.detach())*len(batch)
   losses.append(total/len(y));dump('F_progress.json',{'stage':'training new linear logit corrector','variant':tag,'epoch':epoch,'total_epochs':12,'train_loss':losses[-1],'time':time.time()});print(tag,epoch,losses[-1],flush=True)
  path=O/f'F_{tag}.pt';torch.save({'state_dict':model.state_dict(),'features':12,'class_order':['Al3Ni','PrimarySi','Al','EutecticSi'],'training_seed':819349,'epochs':12,'OOF_train_only':True,'feature_definition':'logprob4,local3x3prob4,Z/4,gray_gradient,entropy,margin','variant':tag},path);stats[tag]={'losses':losses,'sha256':sha(path),'bytes':path.stat().st_size}
 dump('F_training_completion.json',{'time':time.time(),'images':70,'sample_pixels':len(y),'class_counts':counts.tolist(),'models':stats,'OOF_receipt':receipt,'source_archive_sha256':source_sha,'valid20_used_for_training':False,'train_single_held_fold_vs_valid_fourfold_probability_distribution':True});print('F_TRAINING_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:dump('F_training_failure.json',{'trace':traceback.format_exc()});raise
