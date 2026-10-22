import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import numpy as np,json,hashlib,subprocess,time
R=Path(__file__).resolve().parent;B=R.parent;D=Path('/workspace/komap_model1');P=B/'fusion805_compression_round2_20261010';OLD=B/'fusion805_compression_20261010'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (R/'started.json').exists();(R/'started.json').write_text(json.dumps({'time':time.time(),'pid':os.getpid()}))
frozen={}
def freeze(p):frozen[str(p)]=sha(p);return p
def foldmap(root,name,f):
 p=next(root/f'shard{i}/{name}_fold{f}.npy' for i in [0,1] if (root/f'shard{i}/{name}_fold{f}.npy').exists())
 return np.load(freeze(p)).astype(np.float32)
paths=[freeze(B/f'accuracy81_HR_seed2_valid20_20261010/fold{f}_eval/valid_probabilities.npz') for f in [1,4]]
z=[np.load(p,allow_pickle=False) for p in paths];bp=freeze(D/'outputs/model329_fullstats_width4000_uniform_group_oof/valid_ensemble/probabilities.npz');b=np.load(bp,allow_pickle=False)
names=sorted(z[0].files);assert len(names)==20 and all(set(t.files)==set(names) for t in z+[b])
out=R/'expert';out.mkdir();parity={}
try:
 for name in names:
  h1=sum(foldmap(P/'seed1_maps',name,f)*.5 for f in [2,4]).astype(np.float16).astype(np.float32)
  h2=sum(foldmap(OLD/'fold_maps',name,f)*.5 for f in [1,4]).astype(np.float16).astype(np.float32)
  old=(.5*b[name].astype(np.float32)+.25*h1+.25*h2)[[1,2,0,3]]
  fresh=next(P/f'fresh23_gpu0/expert{i}/{name}.npy' for i in [0,1] if (P/f'fresh23_gpu0/expert{i}/{name}.npy').exists())
  assert np.array_equal(old,np.load(freeze(fresh))),name;parity[name]=True
  vals=[t[name] for t in z];assert all(v.dtype==np.float32 and np.isfinite(v).all() for v in vals)
  newh2=sum(v*.5 for v in vals).astype(np.float16).astype(np.float32)
  new=(.5*b[name].astype(np.float32)+.25*h1+.25*newh2)[[1,2,0,3]]
  np.save(out/f'{name}.npy',new)
finally:
 for t in z+[b]:t.close()
assert all(sha(Path(p))==v for p,v in frozen.items())
(R/'expert_completion.json').write_text(json.dumps({'original_lite23_probability_exact':parity,'inputs_unchanged':True,'input_sha256':frozen,'new_raw_gpu_inference':False},indent=2))
py='/workspace/venv-komap/bin/python';pack=B/'accuracy81_candidate_oracle_20261009/publication805_v1/Model349_fusion805';parent=B/'accuracy81_stage1_fresh_20261009'
ref=freeze(B/'accuracy81_oof_zgray_20261009/training/epoch8.pt')
labels=parent/'runs/release_valid/predicted_labels'
for p in labels.glob('*.png'):freeze(p)
subprocess.run([py,str(pack/'infer_fusion.py'),'--input',str(D/'data/valid/images'),'--release-labels',str(labels),'--expert-probabilities',str(out),'--refiner',str(ref),'--output',str(R/'fusion'),'--device','cpu','--strict-valid'],check=True)
subprocess.run([py,str(parent/'score_masks.py'),'--labels',str(R/'fusion/predicted_labels'),'--gt',str(D/'data/valid/masks'),'--output',str(R/'independent_metrics.json')],check=True)
m=json.loads((R/'independent_metrics.json').read_text());assert m['count']==20
assert all(sha(Path(p))==v for p,v in frozen.items())
value=m['image_miou']*100
(R/'completion.json').write_text(json.dumps({'percent':value,'delta_vs_lite23_pp':value-80.51727047204673,'target81_observed':value>=81,'inputs_unchanged':True,'independent_png_scoring':True,'scope':'full lite23 warm seed2 folds1,4 replacement; saved raw probabilities, CPU F recomputed, previous raw R reused; repeated valid development'},indent=2))
print(value,flush=True)
