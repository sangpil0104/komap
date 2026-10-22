import os,sys,json,time,hashlib,subprocess,statistics,traceback,importlib.util
from pathlib import Path
K=Path.home()/'komap';O=K/'codex_L2_seed4_20261013';P=K/'Model349_lite20_b4ws_bundle/Model349_fusion805';F=K/'codex_lite16_N3_L2_fresh_valid_20261013';S=O/'source';PARENT=K/'codex_N3_seed4_20261013/source/outputs/model336_hrnet_best/last.pt';SEEDS=[361,362,363,364]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def status(stage,**kw):dump(O/'status.json',dict(stage=stage,time=time.time(),pid=os.getpid(),**kw))
def idle():
 stable=0
 while stable<3:
  ids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines();stable=0 if any(x.strip() and int(x)!=os.getpid() for x in ids) else stable+1
  if stable<3:time.sleep(10)
def train(s):
 import torch
 out=O/f'training/seed{s}';out.mkdir(parents=True,exist_ok=True)
 if (out/'completion.json').exists():return out
 idle();status('L2 training',seed=s)
 cmd=[sys.executable,'-u',str(S/'model_eut_specialist/train_eut.py'),'--data-root',str(K/'data'),'--output-dir',str(out),'--device','cuda:0','--architecture','UnetPlusPlus','--encoder-name','tu-hrnet_w48','--patch-size','512','--input-scale','2','--samples-per-image','12','--batch-size','4','--accumulation-steps','2','--num-workers','4','--epochs','12','--lr','1e-4','--encoder-lr-mult','.3','--weight-decay','1e-4','--eut-loss','lovasz','--eut-class-weights','1','1.5','1','2','--boundary-pixel-weight','0','--boundary-head-weight','1','--seed',str(s),'--val-every','0','--init-checkpoint',str(PARENT)]
 if (out/'train_state.pt').exists():cmd.append('--resume')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',PYTHONPATH=str(K/'codex_N3_seed4_20261013/source/model336_hrnet_best')+':'+str(S/'model_eut_specialist'),NO_ALBUMENTATIONS_UPDATE='1',HF_HUB_OFFLINE='1')
 dump(out/'recipe.json',{'command':cmd,'parent_sha':sha(PARENT),'seed':s,'validation_epoch_selection':False})
 with (out/'train.log').open('a') as f:
  child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=S/'model_eut_specialist');dump(O/'active_child.json',{'pid':child.pid,'seed':s,'time':time.time()});assert child.wait()==0,'L2 training failed'
 state=torch.load(out/'train_state.pt',map_location='cpu',weights_only=False);assert state['epoch']==12
 dump(out/'completion.json',{'epoch':12,'checkpoint_sha':sha(out/'last.pt')});return out
def evaluate(s,out):
 import torch,numpy as np
 from PIL import Image
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 sys.path.insert(0,str(P/'model397_reproduction_bundle/model392'))
 from komap_seg.checkpoint import load_model_and_preprocessing
 from komap_seg.dataset import KomapFullDataset
 from komap_seg.inference import predict_probs_tta
 dest=O/f'evaluation/seed{s}';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'result.json').exists():return json.loads((dest/'result.json').read_text())
 idle();status('L2 valid20 inference',seed=s);model,_=load_model_and_preprocessing(out/'last.pt',torch.device('cuda'));ds=KomapFullDataset(K/'data/valid/images',masks_dir=None);prob=dest/'probabilities';prob.mkdir(exist_ok=True)
 for i,pair in enumerate(ds.pairs):
  n=pair['stem'];f=prob/f'{n}.npy'
  if f.exists():continue
  sample=ds[i];v=predict_probs_tta(model,sample['image'],patch_size=512,overlap=170,batch_size=2,tta=False,input_scale=2,device=torch.device('cuda')).numpy().astype(np.float16).astype(np.float32)[[1,2,0,3]];assert np.isfinite(v).all();np.save(f,v);print(s,n,flush=True)
 del model;torch.cuda.empty_cache()
 spec=importlib.util.spec_from_file_location('fu',P/'infer_fusion.py');fu=importlib.util.module_from_spec(spec);spec.loader.exec_module(fu);ref=fu.Residual().eval();ref.load_state_dict(torch.load(K/'weights/all/assets/refiner/epoch8.pt',map_location='cpu',weights_only=False)['state_dict'])
 man=json.loads((F/'R_fresh/input_manifest.json').read_text());pal=np.array(man['config']['palette'],np.uint8);data={'L2':{},'both':{}};single=[];rt=lambda a:a.astype(np.float16).astype(np.float32)
 for item in man['images']:
  n=item['name'];load=lambda tag:np.load(F/f'fresh_probabilities/{tag}/{n}.npy').astype(np.float32)[[1,2,0,3]]
  b=load('b4ws');h1=rt((load('hr_s1_f2')+load('hr_s1_f4'))/2);newh1=rt((load('N3')+load('hr_s1_f4'))/2);l2=np.load(prob/f'{n}.npy');h2=rt((l2+load('hr_s2_f4'))/2)
  parent=np.array(Image.open(F/f'R_fresh/predicted_labels/{n}.png'));base=np.array(Image.open(F/f'baseline/fusion/predicted_labels/{n}.png'));gray=torch.from_numpy(fu.prepare_gray(np.array(Image.open(K/f'data/valid/images/{n}_image.png').convert('L'),dtype=np.float32)/255.)[None,None])
  rgb=np.array(Image.open(K/f'data/valid/masks/{n}_mask.png').convert('RGB'));gt=np.full(rgb.shape[:2],255,np.uint8)
  for c,col in enumerate(pal):gt[np.all(rgb==col,-1)]=c
  assert (gt<4).all()
  def score(pred):
   return [float(((pred==c)&(gt==c)).sum()/u) if (u:=((pred==c)|(gt==c)).sum()) else 0. for c in range(4)]
  bc=score(base);single.append(float(np.mean(score(fu.labels(l2)))))
  for tag,hh1 in [('L2',h1),('both',newh1)]:
   e=.5*b+.25*hh1+.25*h2
   with torch.no_grad():v,_=ref(torch.from_numpy(np.log(np.maximum(e,1e-7))[None]),gray)
   ex=fu.labels(e);fl=fu.labels(v.numpy()[0]);pred=np.where(fl==2,2,np.where(np.isin(ex,[0,2]),ex,parent)).astype(np.uint8);ci=score(pred);di=100*(float(np.mean(ci))-float(np.mean(bc)));data[tag][n]={'miou':float(np.mean(ci)),'delta_pp':di,'class_delta_pp':(100*(np.array(ci)-bc)).tolist()};folder=dest/tag/'predicted_labels';folder.mkdir(parents=True,exist_ok=True);Image.fromarray(pred).save(folder/f'{n}.png')
 result={'seed':s,'single_L2_percent':100*float(np.mean(single)),'variants':{}}
 for tag,rows in data.items():result['variants'][tag]={'percent':100*float(np.mean([r['miou'] for r in rows.values()])),'worst_image_delta_pp':min(r['delta_pp'] for r in rows.values()),'worst_class_delta_pp':min(min(r['class_delta_pp']) for r in rows.values()),'better':sum(r['delta_pp']>0 for r in rows.values()),'worse':sum(r['delta_pp']<0 for r in rows.values()),'per_image':rows}
 result.update(checkpoint_count=16,GT_test_used=False,fresh_member=True,common_fresh_R_E_reused=True,direct_family_aggregation=True,valid_development_score=True,N3_fixed_seed=414);dump(dest/'result.json',result);return result
def main():
 if (O/'completion.json').exists():return
 assert (F/'completion.json').exists(),'Fresh validation is prerequisite'
 r=json.loads((F/'result.json').read_text());assert r['results']['baseline']['changed_pixels']==0,'Fresh baseline mismatch requires investigation'
 reference=json.loads((O/'raw_data_reference_sha.json').read_text());assert len(reference)==180;hashes={str(K/'data'/p):h for p,h in reference.items()}
 for p,h in hashes.items():assert sha(p)==h,p
 hashes.update({str(p):sha(p) for p in [PARENT]+list((S/'model_eut_specialist').glob('*.py'))});dump(O/'inputs_sha.json',hashes)
 results=[]
 for s in SEEDS:results.append(evaluate(s,train(s)))
 summary={}
 for tag in ['L2','both']:
  vals=[r['variants'][tag]['percent'] for r in results];summary[tag]={'scores':vals,'mean_percent':statistics.mean(vals),'sample_sd_pp':statistics.stdev(vals),'range_pp':max(vals)-min(vals),'worst_image_delta_pp':min(r['variants'][tag]['worst_image_delta_pp'] for r in results)}
 for p,h in hashes.items():assert sha(p)==h,p
 dump(O/'result.json',{'results':results,'summary':summary,'input_sha_unchanged':True,'release_replaced':False,'test_L2_seed':361,'test_N3_seed':414,'no_valid_best_seed_selection':True});dump(O/'completion.json',{'time':time.time(),'result':'result.json'});status('complete')
if __name__=='__main__':
 try:main()
 except Exception:dump(O/'failure.json',{'time':time.time(),'trace':traceback.format_exc()});raise
