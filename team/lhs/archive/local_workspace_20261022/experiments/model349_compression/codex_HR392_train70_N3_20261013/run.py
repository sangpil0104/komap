import os,sys,time,json,subprocess,hashlib,traceback,importlib.util
from pathlib import Path
K=Path.home()/'komap';O=K/'codex_HR392_train70_N3_20261013';P=K/'Model349_lite20_b4ws_bundle/Model349_fusion805';F=K/'codex_lite16_N3_L2_fresh_valid_20261013';Q=K/'codex_N3_L2_then_resume_queue_20261013'
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def status(s,**kw):dump(O/'status.json',dict(stage=s,pid=os.getpid(),time=time.time(),**kw))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def idle():
 stable=0
 while stable<3:
  ids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines();stable=0 if any(x.strip() and int(x)!=os.getpid() for x in ids) else stable+1
  if stable<3:time.sleep(10)
def evaluate(s,out):
 import numpy as np,torch
 from PIL import Image
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;sys.path.insert(0,str(P/'model397_reproduction_bundle/model392'))
 from komap_seg.checkpoint import load_model_and_preprocessing,resolve_preprocessing
 from komap_seg.dataset import KomapFullDataset,effective_input_scale,scaled_overlap
 from komap_seg.inference import predict_probs_tta
 dest=O/f'evaluation/seed{s}';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'result.json').exists():return json.loads((dest/'result.json').read_text())
 idle();status('train70 valid20 flip4 inference',seed=s);model,saved=load_model_and_preprocessing(out/'last.pt',torch.device('cuda'));prep=resolve_preprocessing(saved);assert prep['normalize_width'];ds=KomapFullDataset(K/'data/valid/images',masks_dir=None);prob=dest/'probabilities';prob.mkdir(exist_ok=True)
 for i,pair in enumerate(ds.pairs):
  n=pair['stem'];f=prob/f'{n}.npy'
  if f.exists():continue
  sample=ds[i];scale=effective_input_scale(prep['input_scale'],prep['target_width'],sample['width']);v=predict_probs_tta(model,sample['image'],patch_size=512,overlap=scaled_overlap(512,2.,170,scale),batch_size=2,tta=True,input_scale=scale,device=torch.device('cuda'),tta_mode='flip4').numpy().astype(np.float32);assert np.isfinite(v).all();np.save(f,v);print(s,n,flush=True)
 del model;torch.cuda.empty_cache();spec=importlib.util.spec_from_file_location('fu',P/'infer_fusion.py');fu=importlib.util.module_from_spec(spec);spec.loader.exec_module(fu);ref=fu.Residual().eval();ref.load_state_dict(torch.load(K/'weights/all/assets/refiner/epoch8.pt',map_location='cpu',weights_only=False)['state_dict']);man=json.loads((F/'R_fresh/input_manifest.json').read_text());pal=np.array(man['config']['palette'],np.uint8);data={};rt=lambda a:a.astype(np.float16).astype(np.float32)
 for item in man['images']:
  n=item['name'];load=lambda tag:np.load(F/f'fresh_probabilities/{tag}/{n}.npy').astype(np.float32)[[1,2,0,3]];b=load('b4ws');parts={'N3':load('N3'),'s1f4':load('hr_s1_f4'),'s2f1':load('hr_s2_f1'),'s2f4':load('hr_s2_f4')};new=np.load(prob/f'{n}.npy')[[1,2,0,3]];parent=np.asarray(Image.open(F/f'R_fresh/predicted_labels/{n}.png'));base=np.asarray(Image.open(F/f'N3/fusion/predicted_labels/{n}.png'));rgb=np.asarray(Image.open(K/f'data/valid/masks/{n}_mask.png').convert('RGB'));gt=np.full(rgb.shape[:2],255,np.uint8)
  for c,col in enumerate(pal):gt[np.all(rgb==col,-1)]=c
  assert (gt<4).all();gray=torch.from_numpy(fu.prepare_gray(np.asarray(Image.open(K/f'data/valid/images/{n}_image.png').convert('L'),np.float32)/255)[None,None])
  def score(p):return [float(((p==c)&(gt==c)).sum()/u) if (u:=((p==c)|(gt==c)).sum()) else 0. for c in range(4)]
  bc=score(base)
  for targets in [('s1f4',),('s2f1',),('s2f4',),('s2f1','s2f4'),('s1f4','s2f1','s2f4')]:
   p=dict(parts);p.update({t:new for t in targets});e=.5*b+.25*rt((p['N3']+p['s1f4'])/2)+.25*rt((p['s2f1']+p['s2f4'])/2)
   with torch.no_grad():v,_=ref(torch.from_numpy(np.log(np.maximum(e,1e-7))[None]),gray)
   ex=fu.labels(e);fl=fu.labels(v.numpy()[0]);pred=np.where(fl==2,2,np.where(np.isin(ex,[0,2]),ex,parent)).astype(np.uint8);ci=score(pred);tag='_'.join(targets);data.setdefault(tag,{})[n]={'miou':float(np.mean(ci)),'delta_pp':100*(float(np.mean(ci))-float(np.mean(bc))),'class_delta_pp':(100*(np.array(ci)-bc)).tolist(),'fixed_pixels':int(((base!=gt)&(pred==gt)).sum()),'new_error_pixels':int(((base==gt)&(pred!=gt)).sum())};folder=dest/tag/'predicted_labels';folder.mkdir(parents=True,exist_ok=True);Image.fromarray(pred).save(folder/f'{n}.png')
 summary={}
 for tag,rows in data.items():
  ds=[v['delta_pp'] for v in rows.values()];summary[tag]={'percent':100*float(np.mean([v['miou'] for v in rows.values()])),'delta_vs_N3_pp':float(np.mean(ds)),'worst_image_delta_pp':min(ds),'worst_image':min(rows,key=lambda n:rows[n]['delta_pp']),'worst_class_delta_pp':min(min(v['class_delta_pp']) for v in rows.values()),'better':sum(v>0 for v in ds),'worse':sum(v<0 for v in ds),'checkpoint_count':16-len(tag.split('_'))+1,'pass_mean_local':bool(np.mean(ds)>=0 and min(ds)>=-.1)}
 result={'seed':s,'preprocessing':prep,'summary':summary,'per_image':data,'baseline_percent':80.62846148459062,'new_candidate_fresh':True,'fixed_common_N3_R_E_reused':True,'valid_development_score':True,'no_valid_best_seed_selection':True};dump(dest/'result.json',result);return result
def main():
 if (O/'completion.json').exists():return
 status('waiting original ANA queue through HR392 set5 evaluation')
 while not (Q/'completion.json').exists():
  if (Q/'failure.json').exists():raise RuntimeError('preceding ANA queue failed')
  time.sleep(20)
 import torch
 raw=json.loads((K/'codex_L2_seed4_20261013/raw_data_reference_sha.json').read_text());hashes={str(K/'data'/p):h for p,h in raw.items()}
 for p,h in hashes.items():assert sha(p)==h,p
 script=P/'model397_reproduction_bundle/model392/train.py';hashes[str(script)]=sha(script);dump(O/'inputs_sha.json',hashes);results=[]
 for seed in [1,2]:
  out=O/f'training/seed{seed}';out.mkdir(parents=True,exist_ok=True)
  if not (out/'completion.json').exists():
   idle();status('train70 HR392 training',seed=seed)
   cmd=[sys.executable,'-u',str(script),'--data-root',str(K/'data'),'--output-dir',str(out),'--device','cuda:0','--architecture','UnetPlusPlus','--encoder-name','tu-hrnet_w48','--encoder-weights','imagenet','--patch-size','512','--input-scale','2','--normalize-width','--samples-per-image','12','--batch-size','4','--accumulation-steps','2','--num-workers','4','--epochs','70','--lr','3e-4','--encoder-lr-mult','1','--weight-decay','1e-4','--boundary-head-weight','1','--rng-mode','legacy','--seed',str(seed),'--val-every','0']
   if (out/'train_state.pt').exists():cmd.append('--resume')
   dump(out/'recipe.json',{'command':cmd,'train70':True,'no_valid_epoch_selection':True,'ImageNet_cached_init':True})
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1',NO_ALBUMENTATIONS_UPDATE='1')
   with (out/'train.log').open('a') as f:
    child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env);dump(O/'active_child.json',{'pid':child.pid,'seed':seed,'time':time.time()});assert child.wait()==0,'train70 training failed'
   state=torch.load(out/'train_state.pt',map_location='cpu',weights_only=False);assert state['epoch']==70;dump(out/'completion.json',{'epoch':70,'checkpoint_sha':sha(out/'last.pt')})
  results.append(evaluate(seed,out))
 for p,h in hashes.items():assert sha(p)==h,p
 dump(O/'result.json',{'results':results,'seeds':[1,2],'release_replaced':False,'input_sha_unchanged':True});dump(O/'completion.json',{'time':time.time(),'result':'result.json'});status('complete')
if __name__=='__main__':
 try:main()
 except Exception:dump(O/'failure.json',{'time':time.time(),'trace':traceback.format_exc()});status('failed');raise
