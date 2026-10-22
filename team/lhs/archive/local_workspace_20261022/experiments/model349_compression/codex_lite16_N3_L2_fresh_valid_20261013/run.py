import os,sys,json,time,subprocess,hashlib,traceback
from pathlib import Path
K=Path.home()/'komap'; O=K/'codex_lite16_N3_L2_fresh_valid_20261013'; P=K/'Model349_lite20_b4ws_bundle/Model349_fusion805'; A=K/'weights/all/assets'; D=K/'data/valid/images'; AD=K/'lite16_b413_adopted_20261012'
def dump(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def status(v):dump(O/'status.json',dict(v,time=time.time(),pid=os.getpid()))
def idle():
 stable=0
 while stable<3:
  ids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines();stable=0 if any(x.strip() and int(x)!=os.getpid() for x in ids) else stable+1
  if stable<3:time.sleep(10)
def run(script,*args):
 child=subprocess.Popen([sys.executable,str(script),*map(str,args)]);dump(O/'active_child.json',{'pid':child.pid,'script':str(script),'time':time.time()});assert child.wait()==0,str(script)
def main():
 if (O/'completion.json').exists():return
 import numpy as np,torch,yaml
 from PIL import Image
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 man=json.loads((K/'l18h/e2e_valid/release/input_manifest.json').read_text());names=[r['name'] for r in man['images']];assert len(names)==20
 manifest=json.loads((AD/'selected_assets_lite16_b413.json').read_text());files=[D/f'{n}_image.png' for n in names]+[K/f'data/valid/masks/{n}_mask.png' for n in names]
 for row in manifest['checkpoints']+manifest['metadata']:
  p=(P/'Model349_exh14' if row['namespace']=='R' else A)/row['path'];assert sha(p)==row['sha256'];files.append(p)
 n3=K/'codex_N3_seed4_20261013/training/seed414/last.pt';zc=K/'codex_N3_seed4_20261013/source/data_zcdf_png/valid/images';l2=K/'codex_lite16_N3_L2_test_20261012/assets/outputs/model_eut_L2/last.pt'
 files += [n3,l2,AD/'selected_R_lite16_b413.yaml']+[zc/f'{n}_image.png' for n in names]
 hashes={str(p):sha(p) for p in files};dump(O/'inputs_sha.json',hashes)
 assert all(sha(D/f'{r["name"]}_image.png')==r['sha256'] for r in man['images'])
 release=O/'R_fresh'
 if not (release/'completion.json').exists():
  assert not release.exists(),'Partial R exists: inspect before retry'
  idle();status({'stage':'fresh R9 inference'})
  run(P/'Model349_exh14/infer_lite.py','--input',D,'--output',release,'--config',AD/'selected_R_lite16_b413.yaml','--gpus','0')
 sys.path[:0]=[str(P/'model397_reproduction_bundle/model329'),str(P/'model397_reproduction_bundle/model392')]
 from common import load_member,load_gray,predict_probs
 from komap_seg.checkpoint import load_model_and_preprocessing,resolve_preprocessing
 from komap_seg.dataset import KomapFullDataset,effective_input_scale,scaled_overlap
 from komap_seg.inference import predict_probs_tta
 device=torch.device('cuda'); ds=KomapFullDataset(D,masks_dir=None)
 bdir=O/'fresh_probabilities/b4ws';bdir.mkdir(parents=True,exist_ok=True)
 if not (bdir/'completion.json').exists():
  idle();status({'stage':'fresh E B4ws fold1/3'})
  members=[load_member(A/f'model329_ws/fold{f}','cuda','last.pt') for f in [1,3]]
  for pair in ds.pairs:
   n=pair['stem'];p=bdir/f'{n}.npy'
   if not p.exists():np.save(p,predict_probs(members,load_gray(pair['image']),tta=True,device='cuda').cpu().numpy().astype(np.float16).astype(np.float32));print('B4ws',n,flush=True)
  del members;torch.cuda.empty_cache();dump(bdir/'completion.json',{'images':20,'fresh':True,'folds':[1,3]})
 jobs=[(f'hr_s{s}_f{f}',A/f'model392/seed{s}/fold{f}/last.pt',D,True) for s,f in [(1,2),(1,4),(2,1),(2,4)]]+[('N3',n3,zc,False),('L2',l2,D,False)]
 for tag,ck,inp,tta in jobs:
  dest=O/f'fresh_probabilities/{tag}';dest.mkdir(parents=True,exist_ok=True)
  if (dest/'completion.json').exists():continue
  idle();status({'stage':'fresh E model inference','model':tag});model,saved=load_model_and_preprocessing(ck,device);prep=resolve_preprocessing(saved);ds1=KomapFullDataset(inp,masks_dir=None);assert {p['stem'] for p in ds1.pairs}==set(names)
  for i,pair in enumerate(ds1.pairs):
   n=pair['stem'];f=dest/f'{n}.npy'
   if f.exists():continue
   sample=ds1[i];scale=effective_input_scale(prep['input_scale'],prep['target_width'],sample['width'])
   v=predict_probs_tta(model,sample['image'],patch_size=512,overlap=scaled_overlap(512,2.,170,scale),batch_size=2,tta=tta,input_scale=scale,device=device,tta_mode='flip4').numpy().astype(np.float32)
   if tag in ['N3','L2']:v=v.astype(np.float16).astype(np.float32)
   assert np.isfinite(v).all() and np.max(np.abs(v.sum(0)-1))<.03;np.save(f,v);print(tag,n,flush=True)
  del model;torch.cuda.empty_cache();dump(dest/'completion.json',{'images':20,'fresh':True,'checkpoint_sha':sha(ck),'tta':tta,'preprocessing':prep})
 status({'stage':'direct family aggregation and F inference'})
 rt=lambda p:p.astype(np.float16).astype(np.float32)
 pal=np.asarray(man['config']['palette'],np.uint8);all_results={}
 for variant in ['baseline','N3','L2']:
  edir=O/f'{variant}/expert';edir.mkdir(parents=True,exist_ok=True)
  for n in names:
   load=lambda tag:np.load(O/f'fresh_probabilities/{tag}/{n}.npy').astype(np.float32)
   h1=rt(.5*load('N3' if variant=='N3' else 'hr_s1_f2')+.5*load('hr_s1_f4'))
   h2=rt(.5*load('L2' if variant=='L2' else 'hr_s2_f1')+.5*load('hr_s2_f4'))
   e=(.5*load('b4ws')+.25*h1+.25*h2)[[1,2,0,3]];assert np.isfinite(e).all();np.save(edir/f'{n}.npy',e)
  fusion=O/f'{variant}/fusion'
  if not (fusion/'completion.json').exists():
   assert not fusion.exists(),'Partial F exists: inspect before retry'
   status({'stage':'fresh F inference','variant':variant});run(P/'infer_fusion.py','--input',D,'--release-labels',release/'predicted_labels','--expert-probabilities',edir,'--refiner',A/'refiner/epoch8.pt','--output',fusion,'--device','cpu','--strict-valid')
  rows={}
  for n in names:
   pred=np.asarray(Image.open(fusion/f'predicted_labels/{n}.png'));old=np.asarray(Image.open(K/f'codex_lite16_b4_local_20261012/fresh/valid/b4_13/fusion/predicted_labels/{n}.png'));rgb=np.asarray(Image.open(K/f'data/valid/masks/{n}_mask.png').convert('RGB'));gt=np.full(rgb.shape[:2],255,np.uint8)
   for c,col in enumerate(pal):gt[np.all(rgb==col,-1)]=c
   assert (gt<4).all()
   def score(v):
    return [float(((v==c)&(gt==c)).sum()/((v==c)|(gt==c)).sum()) if ((v==c)|(gt==c)).any() else 0. for c in range(4)]
   ci=score(pred);bc=score(old);rows[n]={'miou':float(np.mean(ci)),'class_iou':ci,'delta_pp':float(100*(np.mean(ci)-np.mean(bc))),'class_delta_pp':(100*(np.array(ci)-bc)).tolist(),'changed_pixels':int((pred!=old).sum())}
  result={'percent':100*float(np.mean([r['miou'] for r in rows.values()])),'worst_image_delta_pp':min(r['delta_pp'] for r in rows.values()),'worst_class_delta_pp':min(min(r['class_delta_pp']) for r in rows.values()),'better':sum(r['delta_pp']>0 for r in rows.values()),'worse':sum(r['delta_pp']<0 for r in rows.values()),'changed_pixels':sum(r['changed_pixels'] for r in rows.values()),'per_image':rows,'checkpoint_count':16,'fresh_R_E_F':True,'valid_development_score':True}
  dump(O/f'{variant}/result.json',result);all_results[variant]=result;print('RESULT',variant,result['percent'],flush=True)
 for p,h in hashes.items():assert sha(p)==h,p
 dump(O/'result.json',{'results':all_results,'input_sha_unchanged':True,'source_sha':sha(__file__),'release_replaced':False,'N3_seed':414,'selection':'predeclared reproduction seed, not best valid seed','baseline_previous':80.55464328764197})
 dump(O/'completion.json',{'time':time.time(),'result':'result.json','baseline_exact':all_results['baseline']['changed_pixels']==0});status({'stage':'complete'})
if __name__=='__main__':
 try:main()
 except Exception:dump(O/'failure.json',{'time':time.time(),'trace':traceback.format_exc()});raise
