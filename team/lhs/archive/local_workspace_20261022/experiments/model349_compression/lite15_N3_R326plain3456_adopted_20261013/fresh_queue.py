from pathlib import Path
import os,json,time,hashlib,subprocess,traceback
import numpy as np
from PIL import Image
K=Path.home()/'komap';O=K/'lite15_N3_R326plain3456_adopted_20261013';PY='/home/sangpil/miniforge3/envs/komap/bin/python'
def dump(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2))
def main():
 if (O/'fresh_completion.json').exists():return
 while not (O/'completion.json').exists():
  if (O/'failure.json').exists():raise RuntimeError('assembly failed')
  time.sleep(10)
 while not (K/'codex_HR392_train70_N3_20261013/completion.json').exists():
  if (K/'codex_HR392_train70_N3_20261013/failure.json').exists():raise RuntimeError('prior train70 queue failed')
  dump('fresh_status.json',{'stage':'waiting existing HR392 train70 queue completion','pid':os.getpid(),'time':time.time()});time.sleep(20)
 stable=0
 while stable<3:
  busy=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();stable=0 if busy else stable+1
  if stable<3:time.sleep(20)
 for split in ['valid','test']:
  dest=O/f'fresh_{split}';images=K/f'data/{split}/images'
  if not (dest/'completion.json').exists():
   if dest.exists():raise RuntimeError('partial fresh output exists; inspect before retry')
   dump('fresh_status.json',{'stage':'full source-image GPU inference','split':split,'time':time.time()})
   cmd=[PY,str(O/'infer_compact.py'),'--parent-package',str(O/'parent_package'),'--assets',str(O/'assets'),'--input',str(images),'--output',str(dest),'--gpus','0']
   with (O/f'fresh_{split}.log').open('a') as log:
    child=subprocess.Popen(cmd,stdout=log,stderr=log);dump('fresh_active_child.json',{'pid':child.pid,'split':split});rc=child.wait()
   if rc:raise RuntimeError('fresh inference failed '+split)
  if split=='valid':
   pal=np.array([[25,76,153],[76,178,76],[153,127,76],[204,204,204]],np.uint8);rows={};difference={}
   for p in sorted((dest/'fusion/predicted_labels').glob('*.png')):
    n=p.stem;pred=np.array(Image.open(p));rgb=np.array(Image.open(K/f'data/valid/masks/{n}_mask.png').convert('RGB'));gt=np.full(pred.shape,255,np.uint8)
    for c,col in enumerate(pal):gt[np.all(rgb==col,-1)]=c
    assert (gt<4).all();ci=[float(((pred==c)&(gt==c)).sum()/u) if (u:=((pred==c)|(gt==c)).sum()) else 0. for c in range(4)];rows[n]={'class_iou':ci,'miou':float(np.mean(ci))};difference[n]=int((pred!=np.asarray(Image.open(O/f'valid20/predicted_labels/{n}.png'))).sum())
   assert len(rows)==20;score=100*float(np.mean([r['miou'] for r in rows.values()]));dump('fresh_valid_independent.json',{'percent':score,'expected_cached_percent':80.68255541753359,'delta_vs_cached_pp':score-80.68255541753359,'pixel_difference_vs_cached':difference,'per_image':rows})
 dump('fresh_completion.json',{'time':time.time(),'full_GPU_source_inference_valid20_test10':True,'test_GT_available':False});dump('fresh_status.json',{'stage':'complete','time':time.time()})
if __name__=='__main__':
 try:main()
 except Exception:dump('fresh_failure.json',{'trace':traceback.format_exc()});raise
