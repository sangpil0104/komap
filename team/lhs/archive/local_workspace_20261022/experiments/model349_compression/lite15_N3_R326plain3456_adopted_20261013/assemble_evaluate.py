from pathlib import Path
import os,sys,json,shutil,hashlib,time,pickle,importlib.util,traceback
import numpy as np
K=Path.home()/'komap';O=K/'lite15_N3_R326plain3456_adopted_20261013';A=K/'lite16_N3_adopted_20261013';P=K/'Model349_lite20_b4ws_bundle/Model349_fusion805';X=P/'Model349_exh14';T=K/'codex_r326_ema_seed4_20261012/training/set0/fold2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
def link(p,q):
 q.parent.mkdir(parents=True,exist_ok=True)
 if not q.exists():q.symlink_to(p.resolve(),target_is_directory=p.is_dir())
def main():
 if (O/'completion.json').exists():return
 import yaml,torch
 from PIL import Image
 assert sha(T/'last.pt')=='06602d7378249de97a0055923861c70d053b7fcbb374597dbd6187e9688605c6'
 for f in ['infer_compact.py','infer_expert.py','zcdf_input.py']:shutil.copy2(A/f,O/f)
 code=(O/'infer_compact.py').read_text();code=code.replace("assert saved['route']==rconfig['route']", "assert saved['inference']['b4_fold_ids']==rconfig['inference']['b4_fold_ids'], 'Reused R fold selection differs'\n  assert saved['route']==rconfig['route']");(O/'infer_compact.py').write_text(code)
 shutil.copytree(A/'parent_metadata',O/'parent_metadata',dirs_exist_ok=True)
 cfg=json.load(open(A/'selected_config.json'));cfg['key']='lite15_N3_R326plain3456';cfg['status']='User adopted plain seed3456 replacement; original lite16-N3 preserved';dump(O/'selected_config.json',cfg)
 rc=yaml.safe_load((A/'selected_R_lite16_N3.yaml').read_text());rc['inference']['b4_fold_ids']['326']=[2];(O/'selected_R_lite16_N3.yaml').write_text(yaml.safe_dump(rc,allow_unicode=True,sort_keys=False))
 assets=json.load(open(A/'selected_assets.json'));oldbase='weights/model326_hybrid_fullstats_group_oof/'
 for typ in ['checkpoints','metadata']:assets[typ]=[r for r in assets[typ] if not(r['namespace']=='R' and r['path'].startswith(oldbase))]
 assets['checkpoints'].append({'namespace':'R','path':oldbase+'fold2/last.pt','bytes':(T/'last.pt').stat().st_size,'sha256':sha(T/'last.pt'),'source_variant':'plain seed3456 fold2'})
 assets['metadata'].append({'namespace':'R','path':oldbase+'fold2/args.json','sha256':sha(T/'args.json')})
 dest=O/'parent_metadata/R'/oldbase/'fold2/args.json';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(T/'args.json',dest)
 assets['count']=len(assets['checkpoints']);assets['bytes']=sum(x['bytes'] for x in assets['checkpoints']);assert assets['count']==15;dump(O/'selected_assets.json',assets)
 parent=O/'parent_package';overlay=parent/'Model349_exh14';overlay.mkdir(parents=True,exist_ok=True)
 for p in X.iterdir():
  if p.name=='weights':continue
  if p.is_file() and p.suffix=='.py':shutil.copy2(p,overlay/p.name)
  else:link(p,overlay/p.name)
 link(P/'infer_fusion.py',parent/'infer_fusion.py');link(A/'assets',O/'assets')
 for row in assets['checkpoints']+assets['metadata']:
  if row['namespace']!='R':continue
  target=overlay/row['path']
  source=(T/'last.pt' if row['path'].endswith('fold2/last.pt') else T/'args.json') if row['path'].startswith(oldbase) else X/row['path']
  link(source,target);assert sha(target)==row['sha256']
 dump(O/'assembly.json',{'checkpoint_count':15,'R_checkpoint_count':8,'E_checkpoint_count':6,'F_checkpoint_count':1,'plain_sha':sha(T/'last.pt'),'original_release_preserved':True,'GPU_fresh_package_smoke_test':False})
 torch.set_num_threads(2);sys.path[:0]=[str(X),str(X/'src')];import infer_lite
 sp=importlib.util.spec_from_file_location('fu',P/'infer_fusion.py');fu=importlib.util.module_from_spec(sp);sp.loader.exec_module(fu);ref=fu.Residual().eval();ref.load_state_dict(torch.load(K/'weights/all/assets/refiner/epoch8.pt',map_location='cpu',weights_only=False)['state_dict'])
 old=K/'l18h/e2e_valid/release';man=json.load(open(old/'input_manifest.json'));pal=np.array(man['config']['palette'],np.uint8);check=json.load(open(K/'codex_lite16N3_B4_subsets_20261013/EMA_single_per_image.json'))['set0_fold2_plain'];rows={}
 for row in man['images']:
  n=row['name'];ip=K/f'data/valid/images/{n}_image.png';assert sha(ip)==row['sha256']
  with (old/f'probabilities/base/{n}.pkl').open('rb') as f:item=pickle.load(f)
  src={c:np.load(old/f'probabilities/{c}/{n}.npy').astype(np.float32) for c in man['config']['inference']['active_components'] if c!='base'}
  probpath=K/f'codex_r326_ema_seed4_20261012/probabilities/set0/fold2/plain/{n}.npy';src['326']=np.load(probpath).astype(np.float16).astype(np.float32)
  rr=infer_lite.predict_fixed(item,src,man['config']);e=np.load(K/f'codex_lite16_N3_L2_fresh_valid_20261013/N3/expert/{n}.npy').astype(np.float32);gray=torch.from_numpy(fu.prepare_gray(np.array(Image.open(ip).convert('L'),np.float32)/255)[None,None])
  with torch.no_grad():fv,_=ref(torch.from_numpy(np.log(np.maximum(e,1e-7))[None]),gray)
  rdir=O/'valid20/R_release/predicted_labels';rdir.mkdir(parents=True,exist_ok=True);Image.fromarray(rr).save(rdir/f'{n}.png')
  pred=np.where(fu.labels(fv.numpy()[0])==2,2,np.where(np.isin(fu.labels(e),[0,2]),fu.labels(e),rr)).astype(np.uint8)
  rgb=np.array(Image.open(K/f'data/valid/masks/{n}_mask.png').convert('RGB'));gt=np.full(rgb.shape[:2],255,np.uint8)
  for c,col in enumerate(pal):gt[np.all(rgb==col,-1)]=c
  assert (gt<4).all()
  def score(p):return [float(((p==c)&(gt==c)).sum()/u) if(u:=((p==c)|(gt==c)).sum()) else 0. for c in range(4)]
  ci=score(pred);assert abs(np.mean(ci)-check[n]['miou'])<1e-12
  b=np.asarray(Image.open(K/f'codex_lite16_N3_L2_fresh_valid_20261013/N3/fusion/predicted_labels/{n}.png'));o=np.asarray(Image.open(K/f'codex_lite16_b4_local_20261012/fresh/valid/b4_13/fusion/predicted_labels/{n}.png'))
  rows[n]={'candidate_percent':100*float(np.mean(ci)),'lite16_N3_percent':100*float(np.mean(score(b))),'original_lite16_percent':100*float(np.mean(score(o))),'class_iou':ci,'changed_pixels_vs_N3':int((pred!=b).sum()),'probability_sha256':sha(probpath)}
  for label,img in [('predicted_labels',Image.fromarray(pred)),('predicted_color',Image.fromarray(pal[pred]))]:
   dest=O/'valid20'/label;dest.mkdir(parents=True,exist_ok=True);img.save(dest/f'{n}.png')
  dump(O/'progress.json',{'images':len(rows),'total':20,'time':time.time()});print(n,flush=True)
 newman=dict(man);newman['config']=rc;dump(O/'valid20/R_release/input_manifest.json',newman)
 dump(O/'per_image_independent.json',rows);dump(O/'completion.json',{'time':time.time(),'images':20,'score_matches_screen_all20':True,'raw_image_sha_unchanged':True,'candidate_percent':np.mean([v['candidate_percent'] for v in rows.values()]),'original_lite16_percent':np.mean([v['original_lite16_percent'] for v in rows.values()]),'baseline_N3_percent':np.mean([v['lite16_N3_percent'] for v in rows.values()]),'full_GPU_fresh':False})
if __name__=='__main__':
 try:main()
 except Exception:dump(O/'failure.json',{'trace':traceback.format_exc()});raise
