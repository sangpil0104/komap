from pathlib import Path
import os,sys,time,json,traceback,pickle,importlib.util
import numpy as np
K=Path.home()/'komap';O=K/'codex_lite16N3_B4_subsets_20261013';E=K/'codex_r326_ema_seed4_20261012';F=K/'codex_lite16_N3_L2_fresh_valid_20261013';P=K/'Model349_lite20_b4ws_bundle/Model349_fusion805';X=P/'Model349_exh14'
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=1)+'\n')
def main():
 if (O/'EMA_single_completion.json').exists():return
 while not (E/'completion.json').exists():
  if (E/'failure.json').exists():raise RuntimeError('Original EMA experiment failed')
  dump(O/'EMA_single_status.json',{'stage':'waiting original R326 full training/inference/evaluation','time':time.time(),'pid':os.getpid()});time.sleep(20)
 import torch
 from PIL import Image
 torch.set_num_threads(2);sys.path[:0]=[str(X),str(X/'src')];import infer_lite
 spec=importlib.util.spec_from_file_location('fu',P/'infer_fusion.py');fu=importlib.util.module_from_spec(spec);spec.loader.exec_module(fu);ref=fu.Residual().eval();ref.load_state_dict(torch.load(K/'weights/all/assets/refiner/epoch8.pt',map_location='cpu',weights_only=False)['state_dict']);old=K/'l18h/e2e_valid/release';man=json.loads((old/'input_manifest.json').read_text());pal=np.array(man['config']['palette'],np.uint8);data={};variants=['plain','ema','plain_bn','ema_bn']
 for row in man['images']:
  n=row['name'];sources={c:np.load(old/f'probabilities/{c}/{n}.npy').astype(np.float32) for c in man['config']['inference']['active_components'] if c!='base'}
  with (old/f'probabilities/base/{n}.pkl').open('rb') as file:item=pickle.load(file)
  e=np.load(F/f'N3/expert/{n}.npy').astype(np.float32);base=np.asarray(Image.open(F/f'N3/fusion/predicted_labels/{n}.png'));gray=torch.from_numpy(fu.prepare_gray(np.asarray(Image.open(K/f'data/valid/images/{n}_image.png').convert('L'),np.float32)/255)[None,None])
  with torch.no_grad():v,_=ref(torch.from_numpy(np.log(np.maximum(e,1e-7))[None]),gray)
  ex=fu.labels(e);fl=fu.labels(v.numpy()[0]);rgb=np.asarray(Image.open(K/f'data/valid/masks/{n}_mask.png').convert('RGB'));gt=np.full(rgb.shape[:2],255,np.uint8)
  for c,col in enumerate(pal):gt[np.all(rgb==col,-1)]=c
  assert (gt<4).all()
  def final(src):
   r=infer_lite.predict_fixed(item,src,man['config']);return np.where(fl==2,2,np.where(np.isin(ex,[0,2]),ex,r)).astype(np.uint8)
  assert np.array_equal(final(sources),base),'N3 baseline mismatch'
  def iou(p):return [float(((p==c)&(gt==c)).sum()/u) if (u:=((p==c)|(gt==c)).sum()) else 0. for c in range(4)]
  bc=iou(base)
  for s in range(4):
   for f in [1,2]:
    for variant in variants:
     p=np.load(E/f'probabilities/set{s}/fold{f}/{variant}/{n}.npy').astype(np.float16).astype(np.float32);src=dict(sources);src['326']=p;pred=final(src);ci=iou(pred);tag=f'set{s}_fold{f}_{variant}';data.setdefault(tag,{})[n]={'miou':float(np.mean(ci)),'delta_pp':100*(float(np.mean(ci))-float(np.mean(bc))),'class_delta_pp':(100*(np.array(ci)-bc)).tolist(),'fixed_pixels':int(((base!=gt)&(pred==gt)).sum()),'new_error_pixels':int(((base==gt)&(pred!=gt)).sum())}
  dump(O/'EMA_single_status.json',{'stage':'CPU single member group replacement','completed_images':len(next(iter(data.values()))),'total_images':20,'candidates':32,'time':time.time()});print(n,flush=True)
 result=[]
 for tag,rows in data.items():
  ds=[v['delta_pp'] for v in rows.values()];result.append({'key':tag,'percent':100*float(np.mean([v['miou'] for v in rows.values()])),'delta_pp':float(np.mean(ds)),'worst_image_delta_pp':min(ds),'worst_image':min(rows,key=lambda n:rows[n]['delta_pp']),'worst_class_delta_pp':min(min(v['class_delta_pp']) for v in rows.values()),'better':sum(v>0 for v in ds),'worse':sum(v<0 for v in ds),'checkpoint_count':15,'pass_mean_local':bool(np.mean(ds)>=0 and min(ds)>=-.1)})
 result.sort(key=lambda r:-r['delta_pp']);dump(O/'EMA_single_ranking.json',result);dump(O/'EMA_single_per_image.json',data);dump(O/'EMA_single_completion.json',{'time':time.time(),'candidates':32,'baseline_percent':80.62846148459062,'new_GPU_inference':False,'release_replaced':False,'valid_best_seed_adoption':False})
if __name__=='__main__':
 try:main()
 except Exception:dump(O/'EMA_single_failure.json',{'time':time.time(),'trace':traceback.format_exc()});raise
