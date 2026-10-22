from pathlib import Path
import json,subprocess,hashlib,os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
import numpy as np
from PIL import Image
r=Path('/workspace/model349_compress/fusion805_compression_round2_20261010');f=r/'fresh23_gpu0';base=Path('/workspace/model349_compress');parent=base/'accuracy81_stage1_fresh_20261009';full=base/'accuracy81_candidate_oracle_20261009/publication805_v1/Model349_fusion805';py='/workspace/venv-komap/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert json.loads((f/'process_completion.json').read_text())['exit_codes']==[0,0]
inputs=Path('/workspace/komap_model1/data/valid/images');frozen=json.loads((f/'input_sha256.json').read_text());assert all(sha(inputs/n)==h for n,h in frozen.items())
names=[n.removesuffix('_image.png').removesuffix('.png') for n in frozen]
expected={x['name']:x['sha256'] for x in json.loads((full/'config/input_images.json').read_text())};assert {p.stem.removesuffix('_image'):sha(p) for p in inputs.iterdir() if p.name in frozen}==expected
prior=json.loads((parent/'runs/release_valid/input_manifest.json').read_text());assert {x['name']:x['sha256'] for x in prior['images']}==expected
refiner=base/'accuracy81_oof_zgray_20261009/training/epoch8.pt'
manifest=json.loads((r/'selected_assets.json').read_text())
old_frozen=json.loads((parent/'parent_frozen.json').read_text());assert json.loads((parent/'completion.json').read_text())['parent_weights_unchanged']
for x in manifest['checkpoints']:
 if x['namespace']=='R':assert any(k.endswith('/Model349_exh14/'+x['path']) and v==x['sha256'] for k,v in old_frozen.items())
reused_labels={str(p):sha(p) for p in (parent/'runs/release_valid/predicted_labels').glob('*.png')};assert len(reused_labels)==20
for x in manifest['checkpoints']+manifest['metadata']:
 path=refiner if x['path']=='refiner/epoch8.pt' else (parent/'Model349_exh14' if x['namespace']=='R' else parent/'assets')/x['path'];assert sha(path)==x['sha256']
merged=f/'expert';merged.mkdir();rows=[]
for shard in [0,1]:
 receipt=json.loads((f/f'expert{shard}/completion.json').read_text());assert len(receipt['images'])==10
 for row in receipt['images']:
  src=f/f'expert{shard}'/f"{row['name']}.npy";assert sha(src)==row['sha256'];(merged/src.name).symlink_to(src);rows.append(row)
assert {x['name'] for x in rows}==set(expected)
subprocess.run([py,str(full/'infer_fusion.py'),'--input',str(inputs),'--release-labels',str(parent/'runs/release_valid/predicted_labels'),'--expert-probabilities',str(merged),'--refiner',str(refiner),'--output',str(f/'fusion'),'--device','cpu','--strict-valid'],check=True)
subprocess.run([py,str(parent/'score_masks.py'),'--labels',str(f/'fusion/predicted_labels'),'--gt','/workspace/komap_model1/data/valid/masks','--output',str(f/'independent_metrics.json')],check=True)
metrics=json.loads((f/'independent_metrics.json').read_text());assert metrics['count']==20
assert all(sha(p)==h for p,h in reused_labels.items())
(f/'reused_R_label_sha256.json').write_text(json.dumps(reused_labels,indent=2))
cached=json.loads((r/'dual_seed/result.json').read_text());key='s1_24_s2_14_R_canonical14';stage=1;arr=np.array(cached['class_iou'])[stage];keys=cached['keys'];raw={x['image']:x['image_miou'] for x in metrics['images']};values=np.array([raw[n] for n in cached['names']]);gates={}
for ref in ['original27','reference25','r_replay25',key]:
 delta=(values-arr[keys.index(ref)].mean(1))*100;gates[ref]={'mean_delta_pp':float(delta.mean()),'worst_delta_pp':float(delta.min()),'pass':bool(delta.mean()>=-.01-1e-10 and delta.min()>=-.1-1e-10)}
diff={n:int(np.count_nonzero(np.asarray(Image.open(f/f'fusion/predicted_labels/{n}.png'))!=np.asarray(Image.open(r/f'selected_cpu/stage2/predicted_labels/{n}.png')))) for n in cached['names']}
result={'mean_percent':metrics['image_miou']*100,'gates':gates,'fresh_vs_cached_changed_pixels':diff,'fresh_expert':True,'R_reused_from_previous_raw_inference':True,'whole_pipeline_new_raw_inference':False,'independent_generalization_verified':False,'conditional_loso_pass':False}
(f/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
