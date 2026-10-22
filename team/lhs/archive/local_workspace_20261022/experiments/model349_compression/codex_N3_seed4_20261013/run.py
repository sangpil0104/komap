import os,sys,json,time,subprocess,hashlib,traceback,statistics,importlib.util
from pathlib import Path
K=Path.home()/'komap';O=K/'codex_N3_seed4_20261013';S=O/'source';PY=sys.executable;P=K/'Model349_lite20_b4ws_bundle/Model349_fusion805';FRESH=K/'codex_lite16_b4_local_20261012/fresh/valid';SEEDS=[414,415,416,417]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def idle():
 stable=0
 while stable<3:
  ids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines();stable=0 if any(x.strip() and int(x)!=os.getpid() for x in ids) else stable+1
  if stable<3:time.sleep(10)
def train(seed):
 out=O/f'training/seed{seed}';out.mkdir(parents=True,exist_ok=True)
 if (out/'completion.json').exists():return out
 idle();dump(O/'queue_status.json',{'state':'training','seed':seed,'pid':os.getpid(),'time':time.time()})
 cmd=[PY,'-u',S/'model336_hrnet_best/train.py','--data-root',S/'data_zcdf_png','--output-dir',out,'--device','cuda:0','--architecture','UnetPlusPlus','--encoder-name','tu-hrnet_w48','--patch-size','512','--input-scale','2','--samples-per-image','12','--batch-size','4','--accumulation-steps','2','--num-workers','4','--epochs','25','--lr','2e-4','--encoder-lr-mult','.5','--weight-decay','1e-4','--boundary-head-weight','1','--seed',seed,'--val-every','0','--init-checkpoint',S/'outputs/model336_hrnet_best/last.pt']
 if (out/'train_state.pt').exists():cmd.append('--resume')
 dump(out/'recipe.json',{'seed':seed,'command':list(map(str,cmd)),'parent_sha':sha(S/'outputs/model336_hrnet_best/last.pt'),'valid_epoch_selection':False})
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',NO_ALBUMENTATIONS_UPDATE='1',HF_HUB_OFFLINE='1')
 with (out/'train.log').open('a') as f:
  child=subprocess.Popen(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,cwd=S/'model336_hrnet_best',env=env);dump(O/'active_child.json',{'pid':child.pid,'seed':seed,'time':time.time()});assert child.wait()==0,'training failed'
 import torch
 state=torch.load(out/'train_state.pt',map_location='cpu',weights_only=False);assert state['epoch']==25
 dump(out/'completion.json',{'epoch':25,'checkpoint_sha':sha(out/'last.pt')});return out
def evaluate(seed,out):
 import numpy as np,torch
 from PIL import Image
 sys.path[:0]=[str(P/'model397_reproduction_bundle/model392')]
 from komap_seg.checkpoint import load_model_and_preprocessing
 from komap_seg.dataset import KomapFullDataset
 from komap_seg.inference import predict_probs_tta
 spec=importlib.util.spec_from_file_location('fu',P/'infer_fusion.py');fu=importlib.util.module_from_spec(spec);spec.loader.exec_module(fu)
 dest=O/f'evaluation/seed{seed}';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'result.json').exists():return json.loads((dest/'result.json').read_text())
 idle();torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;model,_=load_model_and_preprocessing(out/'last.pt',torch.device('cuda'));ds=KomapFullDataset(S/'data_zcdf_png/valid/images',masks_dir=None)
 prob=dest/'probabilities';prob.mkdir(exist_ok=True)
 for i,pair in enumerate(ds.pairs):
  n=pair['stem'];f=prob/f'{n}.npy'
  if f.exists():continue
  sample=ds[i];v=predict_probs_tta(model,sample['image'],patch_size=512,overlap=170,batch_size=2,tta=False,input_scale=2,device=torch.device('cuda')).numpy().astype(np.float16).astype(np.float32)[[1,2,0,3]];assert np.isfinite(v).all();np.save(f,v);print(seed,n,flush=True)
 del model;torch.cuda.empty_cache()
 refiner=fu.Residual().eval();refiner.load_state_dict(torch.load(K/'weights/all/assets/refiner/epoch8.pt',map_location='cpu',weights_only=False)['state_dict']);man=json.loads((K/'l18h/e2e_valid/release/input_manifest.json').read_text());pal=np.array(man['config']['palette'],np.uint8);rows={}
 for item in man['images']:
  n=item['name'];e=np.load(FRESH/f'expert/b4_13/{n}.npy').astype(np.float32);v=np.load(prob/f'{n}.npy');old=np.load(K/f'add/pe/valid/hr392_s1_f2/{n}.npy').astype(np.float32)[[1,2,0,3]];mix=np.maximum(e+.125*(v-old),1e-7);mix/=mix.sum(0,keepdims=True)
  parent=np.array(Image.open(K/f'l18h/e2e_valid/release/predicted_labels/{n}.png'));base=np.array(Image.open(FRESH/f'b4_13/fusion/predicted_labels/{n}.png'));raw=Image.open(K/f'data/valid/images/{n}_image.png');gray=torch.from_numpy(fu.prepare_gray(np.array(raw.convert('L'),dtype=np.float32)/255.)[None,None])
  with torch.no_grad():fv,_=refiner(torch.from_numpy(np.log(np.maximum(mix,1e-7))[None]),gray)
  ex=fu.labels(mix);fl=fu.labels(fv.numpy()[0]);pred=np.where(fl==2,2,np.where(np.isin(ex,[0,2]),ex,parent)).astype(np.uint8)
  gt_rgb=np.array(Image.open(K/f'data/valid/masks/{n}_mask.png').convert('RGB'));gt=np.full(gt_rgb.shape[:2],255,np.uint8)
  for c,col in enumerate(pal):gt[np.all(gt_rgb==col,-1)]=c
  assert (gt<4).all()
  def score(p):
   ci=[]
   for c in range(4):
    u=((p==c)|(gt==c)).sum();ci.append(float(((p==c)&(gt==c)).sum()/u) if u else 0.)
   return ci,float(np.mean(ci))
  ci,mi=score(pred);bci,bmi=score(base);sci,smi=score(fu.labels(v));rows[n]={'miou':mi,'class_iou':ci,'delta_pp':100*(mi-bmi),'class_delta_pp':(100*(np.array(ci)-bci)).tolist(),'single_N3_miou':smi}
  labels=dest/'predicted_labels';labels.mkdir(exist_ok=True);Image.fromarray(pred).save(labels/f'{n}.png')
 result={'seed':seed,'percent':float(np.mean([r['miou'] for r in rows.values()])*100),'baseline_percent':80.55464328764197,'single_N3_percent':float(np.mean([r['single_N3_miou'] for r in rows.values()])*100),'worst_image_delta_pp':min(r['delta_pp'] for r in rows.values()),'worst_class_delta_pp':min(min(r['class_delta_pp']) for r in rows.values()),'mean_class_delta_pp':np.mean([r['class_delta_pp'] for r in rows.values()],axis=0).tolist(),'better':sum(r['delta_pp']>0 for r in rows.values()),'worse':sum(r['delta_pp']<0 for r in rows.values()),'per_image':rows,'E_HR_s1f2_replaced':True,'checkpoint_count':16,'valid_development_score':True,'cache_delta_formula_not_full_E_fresh':True};dump(dest/'result.json',result);return result
def main():
 if (O/'completion.json').exists():return
 sources=[S/'outputs/model336_hrnet_best/last.pt']+list((S/'model336_hrnet_best').rglob('*.py'))+list((S/'data_zcdf_png/train').rglob('*.png'))+list((S/'data_zcdf_png/valid').rglob('*.png'));assert len(list((S/'data_zcdf_png/train/images').glob('*.png')))==70;assert len(list((S/'data_zcdf_png/valid/images').glob('*.png')))==20
 hashes={str(p):sha(p) for p in sources};dump(O/'input_sha.json',hashes);results=[]
 for seed in SEEDS:
  out=train(seed);dump(O/'queue_status.json',{'state':'valid20 inference/evaluation','seed':seed,'pid':os.getpid()});results.append(evaluate(seed,out))
 for p,h in hashes.items():assert sha(Path(p))==h
 scores=[r['percent'] for r in results];dump(O/'result.json',{'seeds':SEEDS,'results':results,'mean_percent':statistics.mean(scores),'sample_sd_pp':statistics.stdev(scores),'range_pp':max(scores)-min(scores),'mean_vs_lite16_pp':statistics.mean(scores)-80.55464328764197,'no_valid_seed_selection':True,'input_sha_unchanged':True,'release_replaced':False});dump(O/'completion.json',{'time':time.time(),'result':'result.json'});dump(O/'queue_status.json',{'state':'complete','time':time.time()})
if __name__=='__main__':
 try:main()
 except Exception:dump(O/'failure.json',{'time':time.time(),'trace':traceback.format_exc()});raise
