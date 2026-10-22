"""Frozen OOF residual on matched Model401 raw probabilities; no retraining."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import copy,fcntl,json,subprocess,time,traceback
import numpy as np
import torch
from PIL import Image
from fields_io import api,sha,write,SOURCE
from residual_model import Residual
ROOT=Path(__file__).resolve().parent;DATA=Path('/workspace/komap_model1');OUT=ROOT/'matched_source';GPU='3'
def idle():
 while True:
  clear=True
  for _ in range(3):
   mem,util=map(int,subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
   apps=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
   if mem>64 or util or apps:clear=False;break
   time.sleep(2)
  if clear:return
  write(OUT/'queue_status.json',{'state':'waiting_gpu3_idle','memory_mib':mem,'compute_pids':apps,'time':time.time()});time.sleep(30)
def labels(value):
 # Preserve original team argmax priority even at ties.
 return np.array([2,0,1,3],dtype=np.uint8)[value[[2,0,1,3]].argmax(0)]
def metric(pred,target):
 values=[]
 for c in range(4):
  union=np.count_nonzero((pred==c)|(target==c));intersection=np.count_nonzero((pred==c)&(target==c));values.append(float(intersection/union) if union else 0.)
 return values
def main(acquire_locks=True):
 OUT.mkdir(exist_ok=False);locks=[]
 for p in (sorted(set(ROOT.parent.glob('*/gpu3.lock'))|{ROOT/'gpu3.lock'}) if acquire_locks else []):
  f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 assert subprocess.run(['ps','-p','2185466'],capture_output=True).returncode!=0
 frozen=json.loads((ROOT/'parent_frozen.json').read_text());assert all(sha(SOURCE/k)==v for k,v in frozen.items())
 receipt=json.loads((ROOT/'training/completion.json').read_text());checkpoint=Path(receipt['final_checkpoint'])
 assert receipt['epochs']==8 and receipt['optimizer_updates']==560 and not receipt['valid_read'] and not receipt['test_read']
 assert sha(checkpoint)==receipt['final_sha256']
 src=DATA/'outputs/model401_model397_blend_valid_once';mp=src/'inference_manifest.json';manifest=json.loads(mp.read_text());before={str(mp):sha(mp)}
 bp=DATA/'outputs/model329_fullstats_width4000_uniform_group_oof/valid_ensemble/probabilities.npz'
 assert sha(bp)==manifest['b4_valid_archive_sha256'];before[str(bp)]=sha(bp)
 lookups={s:{x['image_id']:x for x in manifest[f'hrnet_seed{s}_mean_maps']} for s in [1,2]}
 for rows in lookups.values():
  assert set(rows)==set(manifest['image_ids']) and len(rows)==20
  for x in rows.values():p=DATA/x['path'];assert sha(p)==x['sha256'];before[str(p)]=x['sha256']
 parent=json.loads((ROOT.parent/'accuracy81_residual_20261008/runs/teacher_valid20/input_manifest.json').read_text());images={x['name']:x for x in parent['images']}
 assert set(images)==set(manifest['image_ids']);palette=np.array(parent['config']['palette'],dtype=np.uint8);entry=api()[0]
 for row in images.values():assert sha(Path(row['path']))==row['sha256']
 idle();torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 model=Residual().cuda().eval();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict']);zero=Residual().cuda().eval()
 results={'control':[],'refined':[]};zero_exact=0;start=time.monotonic()
 for name in results:
  run=OUT/name;run.mkdir();(run/'predicted_labels').mkdir();(run/'predicted_masks').mkdir()
  m={'images':parent['images'],'config':{'model':'Model401 raw '+name,'class_order':parent['config']['class_order'],'palette':parent['config']['palette']},'calibration_sha256':None,'model401_manifest_sha256':before[str(mp)],'refiner_sha256':receipt['final_sha256'] if name=='refined' else None,'raw_parent_reused':True,'prior_or_cleanup_applied':False};write(run/'input_manifest.json',m)
 with np.load(bp,allow_pickle=False) as b4:
  assert set(b4.files)==set(images)
  for name in sorted(images):
   parts=[np.asarray(b4[name],dtype=np.float32)]
   for s in [1,2]:
    with np.load(DATA/lookups[s][name]['path'],allow_pickle=False) as z:assert z.files==[name];parts.append(np.asarray(z[name],dtype=np.float32))
   gray=np.asarray(Image.open(images[name]['path']).convert('L'),dtype=np.float32)/255.
   for p in parts:assert p.shape==(4,*gray.shape) and np.isfinite(p).all() and np.max(np.abs(p.sum(0)-1))<.03
   probability=(.5*parts[0]+.25*parts[1]+.25*parts[2])[[1,2,0,3]];lp=np.log(np.maximum(probability,1e-7)).astype(np.float32)
   logits=torch.from_numpy(lp[None]).cuda();guide=torch.from_numpy(gray[None,None]).cuda()
   with torch.no_grad():identity,d=zero(logits,guide);value,delta=model(logits,guide)
   baseline=labels(probability);assert torch.equal(identity,logits) and torch.count_nonzero(d)==0 and np.array_equal(labels(identity.cpu().numpy()[0]),baseline);zero_exact+=1
   assert torch.isfinite(value).all() and torch.isfinite(delta).all() and delta.abs().max()<=1
   target=entry.decode_gt(DATA/'data/valid/masks'/f'{name}_mask.png',palette)
   for kind,pred in [('control',baseline),('refined',labels(value.cpu().numpy()[0]))]:
    run=OUT/kind;Image.fromarray(pred).save(run/'predicted_labels'/f'{name}.png');Image.fromarray(palette[pred]).save(run/'predicted_masks'/f'{name}_mask.png')
    iou=metric(pred,target);results[kind].append({'image':name,'class_iou':iou,'image_miou':float(np.mean(iou))})
   apps=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip().splitlines();assert len(apps)<=1,'Foreign compute appeared; exit our evaluation only'
   print('MATCHED',name,flush=True)
 elapsed=time.monotonic()-start
 for kind,rows in results.items():
  score=float(np.mean([x['image_miou'] for x in rows]))
  if kind=='control':assert abs(100*score-79.4181708850679)<1e-9,(score,'Model401 baseline reproduction failed')
  write(OUT/kind/'metrics.json',{'metric':'mean image four-class IoU; empty union0','count':20,'class_order':parent['config']['class_order'],'image_miou':score,'images':rows})
  write(OUT/kind/'completion.json',{'images':20,'mean_image_miou':score,'elapsed_seconds':elapsed,'elapsed_kind':'joint control/refiner evaluation only; parent probabilities reused','zero_refiner_parent_masks_exact':zero_exact==20,'gpus':[GPU],'raw_parent_reused':True,'training_started':False})
 assert all(sha(Path(p))==h for p,h in before.items()) and all(sha(SOURCE/k)==h for k,h in frozen.items())
 write(OUT/'completion.json',{'source_probabilities_unchanged':True,'parent14_source_weights_unchanged':True,'zero_refiner_images_exact':zero_exact,'training_started':False,'test_used':False,'source_manifest_sha256':before[str(mp)],'refiner_sha256':receipt['final_sha256'],'scores_percent':{k:100*float(np.mean([x['image_miou'] for x in rows])) for k,rows in results.items()},'time':time.time()})
 print('COMPLETE',json.dumps(json.loads((OUT/'completion.json').read_text())),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);write(OUT/'failure.json',{'traceback':traceback.format_exc(),'time':time.time()});raise
