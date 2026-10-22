import os,sys,time,json,hashlib,subprocess,traceback,shutil,statistics,pickle,importlib.util
from pathlib import Path
from multiprocessing import get_context
import numpy as np
from PIL import Image
K=Path.home()/'komap';O=K/'codex_r326_ema_seed4_20261012';B=K/'Model349_lite20_b4ws_bundle';P=B/'Model349_fusion805';X=P/'Model349_exh14';PY=sys.executable
SEEDS=[(3455,3456),(3457,3458),(3459,3460),(3461,3462)]
VARIANTS={'plain':'last.pt','ema':'ema.pt','plain_bn':'plain_bn.pt','ema_bn':'ema_bn.pt'}
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def gpu_busy():
 pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines()
 return any(x.strip() and int(x.strip())!=os.getpid() for x in pids)
def wait_gpu():
 stable=0
 while stable<3:
  stable=0 if gpu_busy() else stable+1
  if stable<3:time.sleep(20)
def run(cmd,log,cwd=None,env=None):
 with log.open('a') as f:
  child=subprocess.Popen(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,cwd=cwd,env=env)
  dump(O/'active_child.json',{'pid':child.pid,'stage':str(log),'time':time.time()})
  if child.wait()!=0:raise RuntimeError('failed '+str(log))
def verify_data():
 manifest=json.loads((B/'train_route/bundle_assets/training/model303_progress.json').read_text())
 roots=K/'eseed/splits/model329_cv';hashes={}
 for fold in (1,2):
  record=next(r for r in manifest['completed_folds'] if r['fold']==fold)
  for split,key,countkey in [(f'train_fold{fold}','train_groups','train_image_count'),(f'oof_fold{fold}','heldout_groups','heldout_image_count')]:
   paths=sorted((roots/split/'images').glob('*_image.png'));assert len(paths)==record[countkey]
   assert {p.name.split('_')[0] for p in paths}==set(record[key]),'Model303 group mismatch'
   for ip in paths:
    mp=roots/split/'masks'/ip.name.replace('_image.png','_mask.png');assert mp.exists()
    for p in (ip,mp):hashes[str(p.resolve())]=sha(p)
  assert not set(record['train_groups'])&set(record['heldout_groups'])
 return roots,hashes
def eval_one(name):
 import torch,yaml
 torch.set_num_threads(2);sys.path.insert(0,str(X));sys.path.insert(0,str(X/'src'))
 import infer_lite
 spec=importlib.util.spec_from_file_location('fu',P/'infer_fusion.py');fu=importlib.util.module_from_spec(spec);spec.loader.exec_module(fu)
 m=fu.Residual().eval();m.load_state_dict(torch.load(K/'weights/all/assets/refiner/epoch8.pt',map_location='cpu',weights_only=False)['state_dict'])
 orig=K/'l18h/e2e_valid/release';man=json.loads((orig/'input_manifest.json').read_text());cfg=man['config']
 with (orig/f'probabilities/base/{name}.pkl').open('rb') as f:item=pickle.load(f)
 sources={c:np.load(orig/f'probabilities/{c}/{name}.npy').astype(np.float32) for c in cfg['inference']['active_components'] if c!='base'}
 expert=np.load(K/f'codex_lite16_b4_local_20261012/fresh/valid/expert/b4_13/{name}.npy').astype(np.float32)
 ip=K/f'data/valid/images/{name}_image.png';gray=torch.from_numpy(fu.prepare_gray(np.array(Image.open(ip).convert('L'),dtype=np.float32)/255.)[None,None])
 with torch.no_grad():fv,_=m(torch.from_numpy(np.log(np.maximum(expert,1e-7))[None]),gray)
 ex=fu.labels(expert);refined=fu.labels(fv.numpy()[0]);pal=np.array(cfg['palette'],np.uint8)
 rgb=np.array(Image.open(K/f'data/valid/masks/{name}_mask.png').convert('RGB'));gt=np.full(rgb.shape[:2],255,np.uint8)
 for c,col in enumerate(pal):gt[np.all(rgb==col,-1)]=c
 assert np.all(gt<4)
 original_folds={f:np.load(K/f'add/pr/valid/326_f{f}/{name}.npy').astype(np.float32)[[1,2,0,3]] for f in (1,2)}
 configurations={'control':sources['326']}
 for s in range(4):
  for v in VARIANTS:
   parts=[np.load(O/f'probabilities/set{s}/fold{f}/{v}/{name}.npy').astype(np.float32) for f in (1,2)]
   configurations[f'set{s}_{v}']=np.mean(parts,0).astype(np.float16).astype(np.float32)
   if v=='plain':
    configurations[f'set{s}_fold1_only']=((parts[0]+original_folds[2])/2).astype(np.float16).astype(np.float32)
    configurations[f'set{s}_fold2_only']=((original_folds[1]+parts[1])/2).astype(np.float16).astype(np.float32)
 rows={}
 for tag,prob in configurations.items():
  src=dict(sources);src['326']=prob;parent=infer_lite.predict_fixed(item,src,cfg)
  pred=np.where(refined==2,refined,np.where(np.isin(ex,[0,2]),ex,parent)).astype(np.uint8)
  ci=[]
  for c in range(4):
   inter=((pred==c)&(gt==c)).sum();union=((pred==c)|(gt==c)).sum();ci.append(float(inter/union) if union else 0.)
  rows[tag]={'image_miou':float(np.mean(ci)),'class_iou':ci}
  dest=O/'lite16_evaluation'/tag/'predicted_labels';dest.mkdir(parents=True,exist_ok=True);Image.fromarray(pred).save(dest/f'{name}.png')
 expected=np.array(Image.open(K/f'codex_lite16_b4_local_20261012/fresh/valid/b4_13/fusion/predicted_labels/{name}.png'))
 assert np.array_equal(np.array(Image.open(O/f'lite16_evaluation/control/predicted_labels/{name}.png')),expected),'Control mismatch'
 return name,rows
def main():
 if (O/'lite16_evaluation_completion.json').exists():return
 dump(O/'lite16_evaluation_status.json',{'state':'waiting for paired training and original lite18h evaluation','time':time.time(),'pid':os.getpid()})
 while not (O/'completion.json').exists():
  if (O/'failure.json').exists():raise RuntimeError('Original paired experiment failed; no silent resumption')
  time.sleep(30)
 dump(O/'lite16_evaluation_status.json',{'state':'CPU evaluation','time':time.time(),'pid':os.getpid()})
 names=[r['name'] for r in json.loads((K/'l18h/e2e_valid/release/input_manifest.json').read_text())['images']]
 with get_context('spawn').Pool(4) as pool:data=dict(pool.map(eval_one,names))
 dump(O/'lite16_per_image.json',data)
 scores={tag:float(np.mean([data[n][tag]['image_miou'] for n in names])*100) for tag in next(iter(data.values()))};base=scores['control'];summary={}
 assert abs(base-80.55464328764197)<1e-9,'Adopted lite16 control mismatch'
 for v in VARIANTS:
  values=[scores[f'set{s}_{v}'] for s in range(4)]
  delta=[100*(data[n][f'set{s}_{v}']['image_miou']-data[n]['control']['image_miou']) for s in range(4) for n in names]
  summary[v]={'scores_percent':values,'mean_percent':statistics.mean(values),'sample_sd_pp':statistics.stdev(values),'range_pp':max(values)-min(values),'mean_vs_lite16_pp':statistics.mean(values)-base,'worst_image_across_seeds_pp':min(delta)}
 comparisons={}
 for a,b in [('ema','plain'),('plain_bn','plain'),('ema_bn','plain_bn')]:
  ds=[scores[f'set{s}_{a}']-scores[f'set{s}_{b}'] for s in range(4)];comparisons[a+'_vs_'+b]={'paired_seed_deltas_pp':ds,'mean_delta_pp':statistics.mean(ds),'sample_sd_delta_pp':statistics.stdev(ds),'SD_ratio':summary[a]['sample_sd_pp']/summary[b]['sample_sd_pp'] if summary[b]['sample_sd_pp'] else None}
 dump(O/'lite16_result.json',{'baseline':'lite16_b413','baseline_percent':base,'summary':summary,'paired_comparisons':comparisons,'all_scores':scores,'seeds':SEEDS,'valid_selected_seed':False,'release_replaced':False,'GPU_training_repeated':False})
 dump(O/'lite16_evaluation_completion.json',{'time':time.time(),'result':'lite16_result.json'});dump(O/'lite16_evaluation_status.json',{'state':'complete','time':time.time()});print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  dump(O/'lite16_evaluation_failure.json',{'time':time.time(),'error':traceback.format_exc()});dump(O/'lite16_evaluation_status.json',{'state':'failed','time':time.time()});raise
