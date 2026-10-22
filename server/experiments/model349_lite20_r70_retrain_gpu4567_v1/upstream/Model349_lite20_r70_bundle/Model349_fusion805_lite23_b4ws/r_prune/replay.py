"""GT-free R Al-only path pruning; E25 fixed, parent control parity enforced."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import sys,json,hashlib,time,pickle,copy,importlib.util,shutil
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parent;REMOTE=Path('/workspace/model349_compress');OLD=REMOTE/'fusion805_compression_20261010'
RROOT=REMOTE/'exh14_base_search_20261008';RBASE=RROOT/'runs/control14_valid20';BUNDLE=RROOT/'bundles/control14'
REF=REMOTE/'accuracy81_oof_zgray_20261009';HERE=REMOTE/'accuracy81_candidate_oracle_20261009';BASE=HERE/'publication805_v1/canonical_labels'
DATA=Path('/workspace/komap_model1');SRC=DATA/'outputs/model401_model397_blend_valid_once';B4=DATA/'outputs/model329_fullstats_width4000_uniform_group_oof/valid_ensemble/probabilities.npz'
CONFIGS=[{'key':'original27','checkpoint_count_stage2':27,'drop':[]},{'key':'reference25','checkpoint_count_stage2':25,'drop':[]},{'key':'r_no347','checkpoint_count_stage2':24,'drop':['347']},{'key':'r_no322','checkpoint_count_stage2':23,'drop':['322']},{'key':'r_no322_347','checkpoint_count_stage2':22,'drop':['322','347']}]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def labels(p):return np.array([2,0,1,3],dtype=np.uint8)[p[[2,0,1,3]].argmax(0)]
def main():
 assert not (ROOT/'runs').exists();sys.path.insert(0,str(BUNDLE/'src'))
 spec=importlib.util.spec_from_file_location('frozen_R',BUNDLE/'infer_lite.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
 sys.path.insert(0,str(REF));from residual_model import Residual;from gray_input import prepare_gray
 manifest=json.loads((RBASE/'input_manifest.json').read_text());config=manifest['config'];palette=np.asarray(config['palette'],dtype=np.uint8)
 recipe={}
 for cfg in CONFIGS[1:]:
  c=copy.deepcopy(config)
  if '347' in cfg['drop']:c['route']['gamma347']=[0.,0.,0.,0.]
  if '322' in cfg['drop']:c['route']['base']['al_gamma']=0.
  c['inference']['active_components']=[component for component in runner.active_components(c) if component not in cfg['drop']];recipe[cfg['key']]=c
 (ROOT/'R_configs.json').write_text(json.dumps(recipe,indent=2)+'\n')
 receipt=json.loads((REF/'training/completion.json').read_text());checkpoint=Path(receipt['final_checkpoint']);frozen={str(checkpoint):sha(checkpoint),str(B4):sha(B4),str(RBASE/'input_manifest.json'):sha(RBASE/'input_manifest.json')};assert frozen[str(checkpoint)]==receipt['final_sha256']
 torch.set_num_threads(2);model=Residual().eval();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict'])
 names=[row['name'] for row in manifest['images']];rows=[];start=time.time()
 for cfg in CONFIGS:
  for stage in [1,2]:
   for kind in ['predicted_labels','predicted_masks']:(ROOT/f"runs/{cfg['key']}/stage{stage}/{kind}").mkdir(parents=True)
  if cfg['key']!='original27':(ROOT/f"runs/{cfg['key']}/release/predicted_labels").mkdir(parents=True)
 with np.load(B4,allow_pickle=False) as archive:
  for name in names:
   parts=[archive[name].astype(np.float32)]
   with np.load(SRC/f'hrnet_seed1_mean_valid/{name}.npz',allow_pickle=False) as z:parts.append(z[name].astype(np.float32))
   shard=next(OLD/f'fold_maps/shard{i}' for i in [0,1] if (OLD/f'fold_maps/shard{i}/{name}_fold1.npy').exists())
   parts.append(sum(np.load(shard/f'{name}_fold{f}.npy',allow_pickle=False).astype(np.float32)*.5 for f in [1,4]).astype(np.float16).astype(np.float32))
   prob=(.5*parts[0]+.25*parts[1]+.25*parts[2])[[1,2,0,3]];expert=labels(prob)
   image=next(p for p in (DATA/'data/valid/images').iterdir() if p.stem.removesuffix('_image')==name);gray=prepare_gray(np.asarray(Image.open(image).convert('L'),dtype=np.float32)/255.)
   with torch.no_grad():value,delta=model(torch.from_numpy(np.log(np.maximum(prob,1e-7))[None]),torch.from_numpy(gray[None,None]))
   refined=labels(value.numpy()[0])
   basepath=RBASE/f'probabilities/base/{name}.pkl';frozen[str(basepath)]=sha(basepath)
   with basepath.open('rb') as f:item=pickle.load(f)
   sources={}
   for component in runner.active_components(config):
    if component=='base':continue
    path=RBASE/f'probabilities/{component}/{name}.npy';frozen[str(path)]=sha(path);sources[component]=np.load(path,allow_pickle=False).astype(np.float32)
   canonical=np.asarray(Image.open(BASE/f'{name}.png'))
   for cfg in CONFIGS:
    key=cfg['key']
    if key=='original27':
     for stage in [1,2]:
      old=ROOT.parent/f'seed2_only/runs/original27/stage{stage}'
      for kind,filename in [('predicted_labels',f'{name}.png'),('predicted_masks',f'{name}_mask.png')]:shutil.copy2(old/kind/filename,ROOT/f'runs/{key}/stage{stage}/{kind}/{filename}')
     continue
    c=recipe[key];selected={component:sources[component] for component in runner.active_components(c) if component!='base'}
    parent=runner.predict_fixed(item,selected,c).astype(np.uint8)
    if key=='reference25':assert np.array_equal(parent,canonical),('R control parity',name,int(np.count_nonzero(parent!=canonical)))
    Image.fromarray(parent).save(ROOT/f'runs/{key}/release/predicted_labels/{name}.png')
    s1=np.where(np.isin(expert,[0,2]),expert,parent).astype(np.uint8);s2=np.where(refined==2,refined,s1).astype(np.uint8)
    if key=='reference25':
     for stage,pred in [(1,s1),(2,s2)]:assert np.array_equal(pred,np.asarray(Image.open(OLD/f'subset_search/runs/hr2_keep_14/stage{stage}/predicted_labels/{name}.png')))
    for stage,pred in [(1,s1),(2,s2)]:
     dest=ROOT/f'runs/{key}/stage{stage}';Image.fromarray(pred).save(dest/f'predicted_labels/{name}.png');Image.fromarray(palette[pred]).save(dest/f'predicted_masks/{name}_mask.png')
   rows.append({'name':name});print(len(rows),name,round(time.time()-start,1),flush=True)
 assert len(rows)==20 and all(sha(p)==h for p,h in frozen.items())
 (ROOT/'replay_completion.json').write_text(json.dumps({'names':names,'configs':CONFIGS,'rows':rows,'input_sha256':frozen,'gt_read':False,'test_read':False,'gpu_used':False,'R_control_exact':True,'reference25_exact':True,'whole_model_raw_inference':False,'elapsed_seconds':time.time()-start},indent=2)+'\n')
if __name__=='__main__':
 import traceback
 try:main()
 except BaseException:(ROOT/'failure.txt').write_text(traceback.format_exc());raise
