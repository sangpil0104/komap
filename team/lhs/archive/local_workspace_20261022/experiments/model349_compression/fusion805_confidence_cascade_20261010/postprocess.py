from pathlib import Path
import json,subprocess,hashlib,os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
import numpy as np
from PIL import Image
r=Path('/workspace/model349_compress/fusion805_confidence_cascade_20261010');cache=Path('/workspace/model349_compress/fusion805_compression_round2_20261010');f=r/'validation';base=Path('/workspace/model349_compress');parent=base/'accuracy81_stage1_fresh_20261009';full=base/'accuracy81_candidate_oracle_20261009/publication805_v1/Model349_fusion805';py='/workspace/venv-komap/bin/python'
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
 path=refiner if x['path']=='refiner/epoch8.pt' else (parent/'Model349_exh14' if x['namespace']=='R' else r/'assets')/x['path'];assert sha(path)==x['sha256']
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
cached=json.loads((cache/'dual_seed/result.json').read_text());key='s1_24_s2_14_R_canonical14';stage=1;arr=np.array(cached['class_iou'])[stage];keys=cached['keys'];raw={x['image']:x['image_miou'] for x in metrics['images']};values=np.array([raw[n] for n in cached['names']]);gates={}
for ref in ['original27','reference25','r_replay25',key]:
 delta=(values-arr[keys.index(ref)].mean(1))*100;gates[ref]={'mean_delta_pp':float(delta.mean()),'worst_delta_pp':float(delta.min()),'pass':bool(delta.mean()>=-.01-1e-10 and delta.min()>=-.1-1e-10)}
diff={n:int(np.count_nonzero(np.asarray(Image.open(f/f'fusion/predicted_labels/{n}.png'))!=np.asarray(Image.open(cache/f'fresh23_gpu0/fusion/predicted_labels/{n}.png')))) for n in cached['names']}
base23=json.loads((cache/'fresh23_gpu0/independent_metrics.json').read_text());ref23={x['image']:x['image_miou'] for x in base23['images']};delta=np.array([(raw[n]-ref23[n])*100 for n in cached['names']]);gates['final_lite23']={'mean_delta_pp':float(delta.mean()),'worst_delta_pp':float(delta.min()),'pass':bool(delta.mean()>=-.01-1e-10 and delta.min()>=-.1-1e-10)}
result={'storage_fp16_inference_fp32':True,'checkpoint_bytes':json.loads((r/'selected_assets.json').read_text())['bytes'],'mean_percent':metrics['image_miou']*100,'gates':gates,'changed_pixels_vs_final_lite23':diff,'fresh_expert':True,'R_reused_from_previous_raw_inference':True,'whole_pipeline_new_raw_inference':False,'independent_generalization_verified':False,'conditional_loso_pass':False}
new_rows=[x for shard in [0,1] for x in json.loads((f/f'expert{shard}/completion.json').read_text())['images']];fixed=base/'fusion805_storage_fp16_20261010/validation';fixed_rows=[x for shard in [0,1] for x in json.loads((fixed/f'expert{shard}/completion.json').read_text())['images']];assert {x['name'] for x in new_rows}=={x['name'] for x in fixed_rows}
result['average_HR_forwards']=float(np.mean([x['HR_forwards'] for x in new_rows]));result['cheap_images']=sum(not x['fallback_full_lite23'] for x in new_rows);result['E_compute_seconds']=sum(x['compute_seconds'] for x in new_rows);result['fixed_E_compute_seconds']=sum(x['compute_seconds'] for x in fixed_rows);result['observed_E_time_ratio']=result['E_compute_seconds']/result['fixed_E_compute_seconds'];result['timing_scope']='sequential valid20 E inference on physical GPU0; R and F excluded; one paired run'
(f/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
