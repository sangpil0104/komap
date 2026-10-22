import sys,json,hashlib,importlib.util,time,traceback
from pathlib import Path
import numpy as np,torch
from PIL import Image
K=Path.home()/'komap';O=K/'codex_lite16_N3_N2_screen_20261013';F=K/'codex_lite16_N3_L2_fresh_valid_20261013';P=K/'Model349_lite20_b4ws_bundle/Model349_fusion805';C=K/'codex_lite16_complement_rules_20261012/candidates/newmem/N2_convnext_tiny'
def dump(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def main():
 if (O/'completion.json').exists():return
 torch.set_num_threads(2);spec=importlib.util.spec_from_file_location('fu',P/'infer_fusion.py');fu=importlib.util.module_from_spec(spec);spec.loader.exec_module(fu);model=fu.Residual().eval();model.load_state_dict(torch.load(K/'weights/all/assets/refiner/epoch8.pt',map_location='cpu',weights_only=False)['state_dict']);man=json.loads((F/'R_fresh/input_manifest.json').read_text());pal=np.array(man['config']['palette'],np.uint8);rows={};hashes={};rt=lambda a:a.astype(np.float16).astype(np.float32)
 for item in man['images']:
  n=item['name'];path=C/f'{n}.npz';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
  with np.load(path,allow_pickle=False) as z:q=z['v1'].astype(np.float32)
  e=np.load(F/f'N3/expert/{n}.npy').astype(np.float32);assert q.shape==e.shape and np.isfinite(q).all() and np.max(np.abs(q.sum(0)-1))<.03
  load=lambda tag:np.load(F/f'fresh_probabilities/{tag}/{n}.npy').astype(np.float32)[[1,2,0,3]]
  b=load('b4ws');parts={(1,2):load('N3'),(1,4):load('hr_s1_f4'),(2,1):load('hr_s2_f1'),(2,4):load('hr_s2_f4')}
  direct=.5*b+.25*rt((parts[1,2]+parts[1,4])/2)+.25*rt((parts[2,1]+parts[2,4])/2);assert np.array_equal(direct,e),'new package family aggregation does not match fresh reference'
  r=np.asarray(Image.open(F/f'R_fresh/predicted_labels/{n}.png'));base=np.asarray(Image.open(F/f'N3/fusion/predicted_labels/{n}.png'));rgb=np.asarray(Image.open(K/f'data/valid/masks/{n}_mask.png').convert('RGB'));gt=np.full(rgb.shape[:2],255,np.uint8)
  for c,col in enumerate(pal):gt[np.all(rgb==col,-1)]=c
  assert (gt<4).all();gray=torch.from_numpy(fu.prepare_gray(np.asarray(Image.open(K/f'data/valid/images/{n}_image.png').convert('L'),np.float32)/255.)[None,None])
  def final(e):
   with torch.no_grad():v,_=model(torch.from_numpy(np.log(np.maximum(e,1e-7))[None]),gray)
   ex=fu.labels(e);fl=fu.labels(v.numpy()[0]);return np.where(fl==2,2,np.where(np.isin(ex,[0,2]),ex,r)).astype(np.uint8)
  assert np.array_equal(final(e),base)
  def iou(p):return [float(((p==c)&(gt==c)).sum()/u) if (u:=((p==c)|(gt==c)).sum()) else 0. for c in range(4)]
  bc=iou(base);rows[n]={}
  for s,f in [(1,4),(2,1),(2,4)]:
   new=dict(parts);new[s,f]=q;mix=.5*b+.25*rt((new[1,2]+new[1,4])/2)+.25*rt((new[2,1]+new[2,4])/2);pred=final(mix);ci=iou(pred);tag=f'N2_s{s}f{f}';rows[n][tag]={'miou':float(np.mean(ci)),'delta_pp':100*(float(np.mean(ci))-float(np.mean(bc))),'class_delta_pp':(100*(np.array(ci)-bc)).tolist(),'fixed':int(((base!=gt)&(pred==gt)).sum()),'new_errors':int(((base==gt)&(pred!=gt)).sum())};out=O/tag/'predicted_labels';out.mkdir(parents=True,exist_ok=True);Image.fromarray(pred).save(out/f'{n}.png')
  dump(O/'progress.json',{'completed':len(rows),'total':20,'time':time.time()});print(n,flush=True)
 summary={}
 for tag in next(iter(rows.values())):
  a=[v[tag] for v in rows.values()];ds=[x['delta_pp'] for x in a];ci=np.asarray([x['class_delta_pp'] for x in a]);summary[tag]={'percent':100*float(np.mean([x['miou'] for x in a])),'delta_vs_N3_pp':float(np.mean(ds)),'worst_image_delta_pp':min(ds),'worst_class_delta_pp':float(ci.min()),'mean_class_delta_pp':ci.mean(0).tolist(),'better':sum(x>0 for x in ds),'worse':sum(x<0 for x in ds),'fixed_pixels':sum(x['fixed'] for x in a),'new_error_pixels':sum(x['new_errors'] for x in a),'pass':bool(np.mean(ds)>0 and min(ds)>=-.1)}
 for p,h in hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 dump(O/'result.json',{'baseline':'lite16_N3_seed414','baseline_percent':80.62846148459062,'summary':summary,'per_image':rows,'N3_fresh_family_and_final_masks_exact':True,'N2_cached_not_fresh':True,'checkpoint_count':16,'input_sha':hashes,'valid_development_screen':True,'adopted_N2':False});dump(O/'completion.json',{'time':time.time(),'result':'result.json'})
if __name__=='__main__':
 try:main()
 except Exception:dump(O/'failure.json',{'trace':traceback.format_exc()});raise
