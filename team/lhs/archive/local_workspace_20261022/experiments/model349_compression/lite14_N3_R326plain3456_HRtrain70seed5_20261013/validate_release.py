from pathlib import Path
import os,sys,json,time,subprocess,traceback,hashlib
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent;K=Path.home()/'komap';OLD=K/'lite15_N3_R326plain3456_adopted_20261013'
def dump(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def iou(pred,gt):
 return np.array([((pred==c)&(gt==c)).sum()/((pred==c)|(gt==c)).sum() if ((pred==c)|(gt==c)).any() else 0. for c in range(4)])
def main():
 if (ROOT/'completion.json').exists():return
 subprocess.run([sys.executable,ROOT/'build_server.py'],check=True);subprocess.run([sys.executable,ROOT/'verify_package.py'],check=True)
 m=json.load(open(ROOT/'selected_assets.json'));frozen={}
 for row in m['checkpoints']+m['metadata']:
  p=(ROOT/'parent_package/Model349_exh14' if row['namespace']=='R' else ROOT/'assets')/row['path'];frozen[str(p)]=row['sha256']
 # Use the exact existing R source release. Only E has changed.
 for split in ['valid','test']:
  out=ROOT/f'fresh_{split}'
  if (out/'completion.json').exists():continue
  if out.exists():
   assert (out/'expert/completion.json').exists(),'partial expert inference requires inspection'
   if not (out/'fusion/completion.json').exists():
    assert not (out/'fusion').exists(),'partial fusion requires inspection'
    dump('status.json',dict(stage=f'resuming CPU fusion {split}',time=time.time()))
    cmd=[sys.executable,ROOT/'parent_package/infer_fusion.py','--input',K/f'data/{split}/images','--release-labels',OLD/f'fresh_{split}/release/predicted_labels','--expert-probabilities',out/'expert','--refiner',ROOT/'assets/refiner/epoch8.pt','--output',out/'fusion','--device','cpu']
    with (ROOT/f'fresh_{split}.log').open('a') as log:subprocess.run(list(map(str,cmd)),check=True,stdout=log,stderr=subprocess.STDOUT)
   (out/'completion.json').write_text(json.dumps(dict(gt_read=False,config=json.load(open(ROOT/'selected_config.json')),fresh_expert_inference=True,release_supplied=True,resumed_CPU_fusion=True)))
   continue
  while True:
   pids=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
   if not pids:break
   dump('status.json',dict(stage='waiting GPU0 idle',time=time.time()));time.sleep(10)
  dump('status.json',dict(stage=f'source E/F inference {split}',time=time.time(),pid=os.getpid()))
  cmd=[sys.executable,'-u',ROOT/'infer_compact.py','--parent-package',ROOT/'parent_package','--assets',ROOT/'assets','--input',K/f'data/{split}/images','--output',out,'--release',OLD/f'fresh_{split}/release','--gpus','0']
  with (ROOT/f'fresh_{split}.log').open('a') as log:
   child=subprocess.Popen(list(map(str,cmd)),stdout=log,stderr=subprocess.STDOUT)
   dump('active_child.json',dict(pid=child.pid,split=split,time=time.time()));assert child.wait()==0,'source inference failed '+split
 import yaml
 pal=np.array(yaml.safe_load((ROOT/'selected_R_lite16_N3.yaml').read_text())['palette'],np.uint8)
 valid={};maskdiff={};inputsha={};bases=[];expected=K/'codex_HR392_train70_seed345_lite15_20261013/predictions/seed5_pair_s2f1_s2f4'
 for f in sorted((ROOT/'fresh_valid/fusion/predicted_labels').glob('*.png')):
  n=f.stem;pred=np.asarray(Image.open(f));base=np.asarray(Image.open(OLD/f'fresh_valid/fusion/predicted_labels/{n}.png'));rgb=np.asarray(Image.open(K/f'data/valid/masks/{n}_mask.png').convert('RGB'));gt=np.full(rgb.shape[:2],255,np.uint8)
  for c,col in enumerate(pal):gt[np.all(rgb==col,-1)]=c
  assert (gt<4).all();ci=iou(pred,gt);bc=iou(base,gt);delta=100*(ci-bc).mean();bases.append(100*bc.mean())
  valid[n]=dict(percent=100*float(ci.mean()),delta_pp=float(delta),class_delta_pp=(100*(ci-bc)).tolist());maskdiff[n]=int((pred!=np.asarray(Image.open(expected/f'{n}.png'))).sum())
  ip=K/f'data/valid/images/{n}_image.png';inputsha[str(ip)]=sha(ip)
 assert len(valid)==20 and abs(np.mean(bases)-80.68255541753359)<1e-9
 score=float(np.mean([x['percent'] for x in valid.values()]));worst=min(valid,key=lambda n:valid[n]['delta_pp'])
 summary=dict(valid20_percent=score,delta_vs_lite15_pp=score-80.68255541753359,worst_image=worst,worst_image_delta_pp=valid[worst]['delta_pp'],better=sum(x['delta_pp']>0 for x in valid.values()),worse=sum(x['delta_pp']<0 for x in valid.values()),cached_candidate_mask_difference=maskdiff,cached_candidate_exact20=not any(maskdiff.values()),valid_development_score=True)
 dump('valid20_comparison.json',dict(summary=summary,per_image=valid))
 test=[]
 for f in sorted((ROOT/'fresh_test/fusion/predicted_labels').glob('*.png')):
  n=f.stem;pred=np.asarray(Image.open(f));base=np.asarray(Image.open(OLD/f'fresh_test/fusion/predicted_labels/{n}.png'));diff=pred!=base
  test.append(dict(image=n,changed_pixels=int(diff.sum()),total_pixels=diff.size,changed_percent=100*float(diff.mean()),class_area_delta_px=[int((pred==c).sum()-(base==c).sum()) for c in range(4)],transitions=np.bincount((base*4+pred)[diff],minlength=16).reshape(4,4).tolist()))
  ip=K/f'data/test/images/{n}_image.png';inputsha[str(ip)]=sha(ip)
 assert len(test)==10
 total=sum(x['total_pixels'] for x in test);changed=sum(x['changed_pixels'] for x in test)
 dump('test10_comparison.json',dict(changed_pixels=changed,total_pixels=total,changed_percent=100*changed/total,per_image=test,test_GT_used=False,test_accuracy_claimed=False))
 for p,h in frozen.items():assert sha(Path(p))==h
 dump('completion.json',dict(checkpoint_count=14,R=8,E=5,F=1,source_E_F_inference_valid20_test10=True,unchanged_R_fresh_outputs_reused=True,selected_checkpoint_sha256='2fe3a00379d2f9c27a9ad52f9d9b358080ce0dbaa42c434d48e4796c7e7d7fee',valid_summary=summary,input_checkpoint_sha_unchanged=True,input_sha256=inputsha,test_GT_used=False,old_lite15_preserved=True,time=time.time()))
 dump('ADOPTED_BASELINE.json',dict(name=ROOT.name,user_requested=True,architecture={'R':8,'E':5,'F':1,'total':14},valid20_percent=score,valid_is_development_score=True,selection='seed5 chosen after valid comparison of five train70 seeds',previous=OLD.name,old_release_preserved=True,test_accuracy_claimed=False))
 dump('status.json',dict(stage='complete',time=time.time()));print('LITE14_COMPLETE',score,changed,total,flush=True)
if __name__=='__main__':
 try:main()
 except Exception:dump('failure.json',dict(trace=traceback.format_exc(),time=time.time()));raise
