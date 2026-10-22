"""Recover the fresh valid/test check after an import-path-only interruption.

The completed valid R source inference is reused only after exact image and mask
checks. The incomplete output is preserved and its original failure receipt is
renamed after successful completion.
"""
from pathlib import Path
import hashlib, json, os, subprocess, sys, time, traceback
import numpy as np
from PIL import Image

K=Path.home()/"komap"
O=K/"lite15_N3_R326plain3456_adopted_20261013"
PY=Path("/home/sangpil/miniforge3/envs/komap/bin/python")
PALETTE=np.array([[25,76,153],[76,178,76],[153,127,76],[204,204,204]],np.uint8)

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def dump(path,obj):
 Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def run(cmd,log,env=None):
 with Path(log).open('a') as f:
  rc=subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,env=env).returncode
 if rc: raise RuntimeError(f"command exited {rc}: {cmd}")

def wait_idle():
 stable=0
 while stable<3:
  p=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines()
  stable=stable+1 if not any(x.strip() for x in p) else 0
  if stable<3: time.sleep(10)

def score_and_compare(split):
 root=O/f'fresh_{split}';pred_dir=root/'fusion/predicted_labels'
 names=sorted(p.stem for p in pred_dir.glob('*.png'))
 expected=20 if split=='valid' else 10
 assert len(names)==expected,(split,len(names))
 differences={};scores={}
 for name in names:
  pred=np.asarray(Image.open(pred_dir/f'{name}.png'))
  assert pred.ndim==2 and pred.min()>=0 and pred.max()<4
  row={'shape':list(pred.shape)}
  if split=='valid':
   rgb=np.asarray(Image.open(K/f'data/valid/masks/{name}_mask.png').convert('RGB'))
   gt=np.full(pred.shape,255,np.uint8)
   for c,col in enumerate(PALETTE):gt[np.all(rgb==col,-1)]=c
   assert (gt<4).all()
   ci=[]
   for c in range(4):
    u=((pred==c)|(gt==c)).sum();ci.append(float(((pred==c)&(gt==c)).sum()/u) if u else 0.)
   scores[name]={'class_iou':ci,'miou':float(np.mean(ci))}
   cached=np.asarray(Image.open(O/f'valid20/predicted_labels/{name}.png'))
   differences[name]=int(np.count_nonzero(pred!=cached))
 return {'images':len(names),'scores':scores,'pixel_difference_vs_cached':differences}

def main():
 marker=O/'fresh_recovery_completion.json'
 if marker.exists(): return
 valid=O/'fresh_valid'; release=valid/'release'; expert=valid/'expert'; fusion=valid/'fusion'
 assert (release/'completion.json').is_file() and (release/'input_manifest.json').is_file()
 manifest=json.loads((release/'input_manifest.json').read_text())
 images=manifest['images']; assert len(images)==20
 by_name={x['name']:x['sha256'] for x in images}
 assert set(by_name)=={p.stem.removesuffix('_image') for p in (K/'data/valid/images').glob('*_image.png')}
 for name,want in by_name.items():
  assert sha(K/f'data/valid/images/{name}_image.png')==want
 rdir=release/'predicted_labels'; assert len(list(rdir.glob('*.png')))==20
 for p in rdir.glob('*.png'):
  ref=np.asarray(Image.open(O/f'valid20/R_release/predicted_labels/{p.name}'))
  assert np.array_equal(np.asarray(Image.open(p)),ref),f'R source result differs: {p.name}'
 assert not expert.exists() and not fusion.exists(),"partial E/fusion outputs already exist; inspect before retry"
 test=O/'fresh_test'
 assert not test.exists(),"partial test output already exists; inspect before retry"
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1',NO_ALBUMENTATIONS_UPDATE='1')
 dump(O/'fresh_recovery_status.json',{'stage':'valid E source inference','reusing_verified_R_only':True,'time':time.time()})
 wait_idle()
 run([PY,O/'infer_expert.py','--parent-package',O/'parent_package','--assets',O/'assets','--input',K/'data/valid/images','--output',expert,'--config',O/'selected_config.json','--device','cuda'],O/'fresh_recovery_valid_E.log',env)
 dump(O/'fresh_recovery_status.json',{'stage':'valid fusion and independent score','time':time.time()})
 run([PY,O/'parent_package/infer_fusion.py','--input',K/'data/valid/images','--release-labels',rdir,'--expert-probabilities',expert,'--refiner',O/'assets/refiner/epoch8.pt','--output',fusion,'--device','cpu'],O/'fresh_recovery_valid_fusion.log',env)
 valid_result=score_and_compare('valid')
 dump(O/'fresh_recovery_status.json',{'stage':'fresh source test inference','time':time.time(),'valid_images':valid_result['images']})
 wait_idle()
 run([PY,O/'infer_compact.py','--parent-package',O/'parent_package','--assets',O/'assets','--input',K/'data/test/images','--output',test,'--gpus','0'],O/'fresh_recovery_test.log',env)
 test_result=score_and_compare('test')
 hashes={}
 for split in ['valid','test']:
  for p in sorted((K/f'data/{split}/images').glob('*_image.png')): hashes[str(p)]=sha(p)
 failure=O/'fresh_failure.json'; archived=None
 if failure.exists():
  archived=O/'fresh_failure_before_import_path_fix.json'
  assert not archived.exists()
  failure.rename(archived)
 dump(marker,{'time':time.time(),'full_source_inference_valid20_test10':True,'valid':valid_result,'test':test_result,'input_sha256':hashes,'reused_valid_R_verified_pixel_exact20':True,'test_GT_read':False,'test_accuracy_claimed':False,'import_path_fix_sha256':sha(O/'infer_expert.py'),'previous_failure_archived':str(archived) if archived else None})
 dump(O/'fresh_completion.json',{'time':time.time(),'full_GPU_source_inference_valid20_test10':True,'recovered_after_import_path_fix':True,'test_GT_available':False,'test_accuracy_claimed':False})
 dump(O/'fresh_status.json',{'stage':'complete','time':time.time(),'recovered':True})

if __name__=='__main__':
 try: main()
 except Exception:
  dump(O/'fresh_recovery_failure.json',{'time':time.time(),'trace':traceback.format_exc()})
  dump(O/'fresh_recovery_status.json',{'stage':'failed','time':time.time()})
  raise
