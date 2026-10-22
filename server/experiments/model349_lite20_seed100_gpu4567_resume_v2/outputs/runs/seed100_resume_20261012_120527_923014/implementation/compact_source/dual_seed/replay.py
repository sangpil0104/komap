"""GT-free joint HR subset replay; two disjoint image shards, frozen rules."""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import sys,json,hashlib,time,argparse,itertools
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parent
REMOTE=Path('/workspace/model349_compress');OLD=REMOTE/'fusion805_compression_20261010'
REF=REMOTE/'accuracy81_oof_zgray_20261009';HERE=REMOTE/'accuracy81_candidate_oracle_20261009'
PACK=HERE/'publication805_v1/Model349_fusion805';BASE=HERE/'publication805_v1/canonical_labels'
DATA=Path('/workspace/komap_model1');SRC=DATA/'outputs/model401_model397_blend_valid_once'
B4=DATA/'outputs/model329_fullstats_width4000_uniform_group_oof/valid_ensemble/probabilities.npz'
SUBSETS1=[list(c) for count in [1,2,3,4] for c in itertools.combinations([1,2,3,4],count)]
R_LAYOUTS={'canonical14':14,'no347':13,'no322':12,'no322_347':11}
CONFIGS=[{'key':'original27','hr1':[1,2,3,4],'hr2':[1,2,3,4],'weights':[.5,.25,.25],'R_mode':'canonical14','R_checkpoint_count':14}, {'key':'reference25','hr1':[1,2,3,4],'hr2':[1,4],'weights':[.5,.25,.25],'R_mode':'canonical14','R_checkpoint_count':14}, {'key':'r_replay25','hr1':[1,2,3,4],'hr2':[1,4],'weights':[.5,.25,.25],'R_mode':'cache14','R_checkpoint_count':14}]
for mode,count in R_LAYOUTS.items():
 if mode!='canonical14':CONFIGS.append({'key':'referenceE_R_'+mode,'hr1':[1,2,3,4],'hr2':[1,4],'weights':[.5,.25,.25],'R_mode':mode,'R_checkpoint_count':count})
for keep1 in SUBSETS1:
 for keep2 in [[],[1],[4],[1,4]]:
  if len(keep1)+len(keep2)>5:continue
  for mode,count in R_LAYOUTS.items():CONFIGS.append({'key':'s1_'+''.join(map(str,keep1))+'_s2_'+(''.join(map(str,keep2)) or 'none')+'_R_'+mode,'hr1':keep1,'hr2':keep2,'weights':[.5,.25,.25] if keep2 else [.5,.5,0],'R_mode':mode,'R_checkpoint_count':count})
GROUPS={}
for cfg in CONFIGS:GROUPS.setdefault((tuple(cfg['hr1']),tuple(cfg['hr2']),tuple(cfg['weights'])),[]).append(cfg)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def labels(p):return np.array([2,0,1,3],dtype=np.uint8)[p[[2,0,1,3]].argmax(0)]
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,choices=[0,1],required=True);a=parser.parse_args()
 assert not (ROOT/f'replay_shard{a.shard}_completion.json').exists()
 sys.path.insert(0,str(REF));from residual_model import Residual;from gray_input import prepare_gray
 import yaml
 palette=np.asarray(yaml.safe_load((PACK/'Model349_exh14/configs/model349_lite.yaml').read_text())['palette'],dtype=np.uint8)
 receipt=json.loads((REF/'training/completion.json').read_text());checkpoint=Path(receipt['final_checkpoint']);frozen={str(checkpoint):sha(checkpoint),str(B4):sha(B4)}
 assert frozen[str(checkpoint)]==receipt['final_sha256'] and frozen[str(B4)]=='80ebfddad10abb1bb5f1d7af8472f477fd5b6435b3c7fef0bfa51e21e1314b05'
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 model=Residual().to('cuda').eval();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict'])
 assert (ROOT.parent/'r_prune_v2/replay_completion.json').exists()
 names=[r['name'] for r in json.loads((PACK/'config/input_images.json').read_text())];rows=[];started=time.time()
 for cfg in CONFIGS:
  for stage in [1,2]:
   for kind in ['predicted_labels','predicted_masks']:(ROOT/f"runs/{cfg['key']}/stage{stage}/{kind}").mkdir(parents=True,exist_ok=True)
 with np.load(B4,allow_pickle=False) as archive:
  assert set(archive.files)==set(names)
  for j,name in enumerate(names):
   if j%2!=a.shard:continue
   b4=archive[name].astype(np.float32);means={}
   for seed in [1,2]:
    parent=ROOT.parent/'seed1_maps' if seed==1 else OLD/'fold_maps'
    shard=next(parent/f'shard{i}' for i in [0,1] if (parent/f'shard{i}/{name}_fold1.npy').exists())
    maps={f:np.load(shard/f'{name}_fold{f}.npy',allow_pickle=False).astype(np.float32) for f in [1,2,3,4]}
    for f in maps:frozen[str(shard/f'{name}_fold{f}.npy')]=sha(shard/f'{name}_fold{f}.npy')
    cache=SRC/f'hrnet_seed{seed}_mean_valid/{name}.npz';frozen[str(cache)]=sha(cache)
    with np.load(cache,allow_pickle=False) as z:assert np.array_equal(sum(maps[f]*.25 for f in maps).astype(np.float16),z[name]),name
    subsets=SUBSETS1 if seed==1 else [[1],[4],[1,4],[1,2,3,4]]
    means[seed]={tuple(keep):sum(maps[f]*(1./len(keep)) for f in keep).astype(np.float16).astype(np.float32) for keep in subsets}
   image=next(p for p in (DATA/'data/valid/images').iterdir() if p.stem.removesuffix('_image')==name)
   gray=prepare_gray(np.asarray(Image.open(image).convert('L'),dtype=np.float32)/255.);parent=np.asarray(Image.open(BASE/f'{name}.png'))
   frozen[str(image)]=sha(image);frozen[str(BASE/f'{name}.png')]=sha(BASE/f'{name}.png')
   gray_tensor=torch.from_numpy(gray[None,None]).to('cuda')
   parents={'canonical14':parent}
   for mode,key in [('cache14','r_replay25'),('no347','r_no347'),('no322','r_no322'),('no322_347','r_no322_347')]:
    path=ROOT.parent/f'r_prune_v2/runs/{key}/release/predicted_labels/{name}.png';frozen[str(path)]=sha(path);parents[mode]=np.asarray(Image.open(path))
   for (keep1,keep2,w),group in GROUPS.items():
    p=w[0]*b4+w[1]*means[1][keep1]
    if w[2]:p=p+w[2]*means[2][keep2]
    prob=p[[1,2,0,3]];assert prob.shape==(4,*parent.shape) and np.isfinite(prob).all()
    with torch.no_grad():value,delta=model(torch.from_numpy(np.log(np.maximum(prob,1e-7))[None]).to('cuda'),gray_tensor)
    expert=labels(prob);refined=labels(value.cpu().numpy()[0])
    for cfg in group:
     key=cfg['key'];parent=parents[cfg['R_mode']]
     s1=np.where(np.isin(expert,[0,2]),expert,parent).astype(np.uint8);s2=np.where(refined==2,refined,s1).astype(np.uint8)
     if key=='original27':assert np.array_equal(s2,np.asarray(Image.open(HERE/f'publication805_v1/portable_valid20/predicted_labels/{name}.png')))
     if key=='reference25':
      for stage,pred in [(1,s1),(2,s2)]:assert np.array_equal(pred,np.asarray(Image.open(OLD/f'subset_search/runs/hr2_keep_14/stage{stage}/predicted_labels/{name}.png')))
     if key=='r_replay25':
      for stage,pred in [(1,s1),(2,s2)]:assert np.array_equal(pred,np.asarray(Image.open(ROOT.parent/f'r_prune_v2/runs/r_replay25/stage{stage}/predicted_labels/{name}.png')))
     for stage,pred in [(1,s1),(2,s2)]:
      dest=ROOT/f'runs/{key}/stage{stage}';path=dest/f'predicted_labels/{name}.png';assert not path.exists();Image.fromarray(pred).save(path)
   rows.append({'name':name});print(len(rows),name,round(time.time()-started,1),flush=True)
 assert len(rows)==10 and all(sha(p)==h for p,h in frozen.items())
 (ROOT/f'replay_shard{a.shard}_completion.json').write_text(json.dumps({'names':names,'rows':rows,'configs':CONFIGS,'input_sha256':frozen,'gt_read':False,'test_read':False,'gpu_used':True,'whole_model_raw_inference':False,'reference27_exact':True,'R_cache_matched_reference_exact':True,'candidate_color_masks_generated':False,'refiner_input_groups':len(GROUPS),'reference25_exact':True,'hr_family_parity_exact':True,'elapsed_seconds':time.time()-started},indent=2)+'\n')
if __name__=='__main__':
 import traceback
 try:main()
 except BaseException:(ROOT/f'failure_shard{sys.argv[-1]}.txt').write_text(traceback.format_exc());raise
