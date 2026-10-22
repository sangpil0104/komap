import os,sys,time,json,hashlib,subprocess,traceback
from pathlib import Path
import numpy as np
from PIL import Image
K=Path.home()/'komap';O=K/'codex_lite16_b4_local_20261012';B=K/'Model349_lite20_b4ws_bundle';P=B/'Model349_fusion805';PY=sys.executable
TAGS=['control','b4_13','b4_14']
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def busy():
 for pid,fragment in [(2513950,'chain_after5.sh'),(2833208,'run_e.sh')]:
  ps=subprocess.run(['ps','-p',str(pid),'-o','args='],capture_output=True,text=True)
  if fragment in ps.stdout:return True
 return bool(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip())
def run(cmd,log,env=None):
 with log.open('a') as f:
  child=subprocess.Popen(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,env=env)
  dump(O/'active_child.json',{'pid':child.pid,'stage':log.name,'time':time.time()})
  if child.wait()!=0:raise RuntimeError('failed '+str(log))
def metrics(folder,split,reference):
 pal=np.array([[25,76,153],[76,178,76],[153,127,76],[204,204,204]],np.uint8);rows=[]
 for ip in sorted((K/f'data/{split}/images').glob('*_image.png')):
  n=ip.stem.removesuffix('_image');pred=np.array(Image.open(folder/f'{n}.png'));ref=np.array(Image.open(reference/f'{n}.png'));assert pred.shape==ref.shape and np.all(pred<4)
  row={'image':n,'changed_pixels':int((pred!=ref).sum()),'total_pixels':pred.size,'reference_sha256':sha(reference/f'{n}.png'),'prediction_sha256':sha(folder/f'{n}.png')}
  if split=='valid':
   rgb=np.array(Image.open(K/f'data/valid/masks/{n}_mask.png').convert('RGB'));gt=np.full(pred.shape,255,np.uint8)
   for c,col in enumerate(pal):gt[np.all(rgb==col,-1)]=c
   assert np.all(gt<4);ci=[]
   for c in range(4):
    intersection=((pred==c)&(gt==c)).sum();union=((pred==c)|(gt==c)).sum();ci.append(float(intersection/union) if union else 0.)
   row.update(class_iou=ci,image_miou=float(np.mean(ci)))
  rows.append(row)
 return {'images':rows,'count':len(rows),'changed_pixels':sum(r['changed_pixels'] for r in rows),'total_pixels':sum(r['total_pixels'] for r in rows),**({'image_miou':float(np.mean([r['image_miou'] for r in rows]))} if split=='valid' else {})}
def main():
 if (O/'fresh_completion.json').exists():return
 dump(O/'queue_status.json',{'state':'waiting for current HR seed chain and idle GPU','time':time.time(),'waiter_pid':os.getpid()})
 stable=0
 while stable<3:
  stable=0 if busy() else stable+1
  if stable<3:time.sleep(20)
 dump(O/'queue_status.json',{'state':'running fresh validation','time':time.time(),'waiter_pid':os.getpid()})
 cfg=json.loads((B/'Model349_fusion805_lite23_b4ws/selected_config_lite18h.json').read_text());dump(O/'config.json',cfg)
 result={};input_hashes={}
 for split in ['valid','test']:
  work=O/'fresh'/split;work.mkdir(parents=True,exist_ok=True)
  release=K/f'l18h/e2e_{split}/release';manifest=json.loads((release/'input_manifest.json').read_text())
  expected={r['name']:r['sha256'] for r in manifest['images']};actual={p.stem.removesuffix('_image'):sha(p) for p in (K/f'data/{split}/images').glob('*_image.png')};assert expected==actual,'R input images changed'
  import yaml
  source_cfg=yaml.safe_load((B/'Model349_fusion805_lite23_b4ws/selected_R_lite18h.yaml').read_text());assert manifest['config']==source_cfg,'R inference config changed'
  assert manifest['calibration_sha256']==sha(P/'Model349_exh14'/source_cfg['calibration'])
  input_hashes[split]=actual
  expert=work/'expert'
  if not (expert/'completion.json').exists():
   # Never start a GPU stage if a new compute job has taken the GPU meanwhile.
   while busy():time.sleep(20)
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',NO_ALBUMENTATIONS_UPDATE='1')
   run([PY,'-u',O/'fresh_expert_triplet.py','--parent-package',P,'--assets',K/'weights/all/assets','--input',K/f'data/{split}/images','--output',expert,'--config',O/'config.json','--device','cuda'],work/'expert.log',env)
  for tag in TAGS:
   fusion=work/tag/'fusion';fusion.parent.mkdir(exist_ok=True)
   if not (fusion/'completion.json').exists():run([PY,P/'infer_fusion.py','--input',K/f'data/{split}/images','--release-labels',release/'predicted_labels','--expert-probabilities',expert/tag,'--refiner',K/'weights/all/assets/refiner/epoch8.pt','--output',fusion,'--device','cpu'],work/tag/'fusion.log')
   result.setdefault(tag,{})[split]=metrics(fusion/'predicted_labels',split,K/f'l18h/e2e_{split}/fusion/predicted_labels')
   dump(work/tag/'metrics.json',result[tag][split])
  assert result['control'][split]['changed_pixels']==0,'Fresh full-fold control differs from original lite18h'
 summary={}
 base=result['control']['valid'];b={r['image']:r for r in base['images']}
 for tag in TAGS:
  valid=result[tag]['valid'];d={r['image']:100*(r['image_miou']-b[r['image']]['image_miou']) for r in valid['images']};cd={r['image']:100*(np.array(r['class_iou'])-np.array(b[r['image']]['class_iou'])) for r in valid['images']}
  summary[tag]={'valid20_percent':valid['image_miou']*100,'mean_delta_pp':float(np.mean(list(d.values()))),'worst_image':min(d,key=d.get),'worst_image_delta_pp':min(d.values()),'mean_class_delta_pp':np.mean(list(cd.values()),0).tolist(),'worst_image_class_delta_pp':min(float(v.min()) for v in cd.values()),'valid_changed_pixels':valid['changed_pixels'],'test_changed_pixels':result[tag]['test']['changed_pixels'],'test_total_pixels':result[tag]['test']['total_pixels'],'passes_mean_and_worst_image':float(np.mean(list(d.values())))>=0 and min(d.values())>=-.10,'valid_selected':tag!='control','test_accuracy_claimed':False}
 dump(O/'fresh_result.json',{'summary':summary,'all_metrics':result,'input_image_sha256':input_hashes,'R_reused_after_SHA_config_check':True,'control_exact':True,'release_replaced':False})
 dump(O/'fresh_completion.json',{'time':time.time(),'result':'fresh_result.json','release_replaced':False});dump(O/'queue_status.json',{'state':'complete','time':time.time()})
 print(json.dumps(summary,indent=2),flush=True)
try:main()
except Exception:
 dump(O/'fresh_failure.json',{'time':time.time(),'error':traceback.format_exc()});dump(O/'queue_status.json',{'state':'failed','time':time.time()});raise
