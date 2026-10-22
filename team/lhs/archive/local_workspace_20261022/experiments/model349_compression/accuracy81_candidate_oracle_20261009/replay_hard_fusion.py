"""GT-free CPU replay of Model401/refiner and two-stage mask composition.

Uses preserved parent probabilities, so this is not fresh raw inference.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import sys, json, hashlib
import numpy as np
import torch
from PIL import Image

ROOT = Path('/workspace/model349_compress')
HERE = ROOT / 'accuracy81_candidate_oracle_20261009'
REF = ROOT / 'accuracy81_oof_zgray_20261009'
sys.path.insert(0, str(REF))
from residual_model import Residual
from gray_input import prepare_gray
BASE = ROOT / 'exh14_base_search_20261008/runs/control14_valid20'
SRC = Path('/workspace/komap_model1/outputs/model401_model397_blend_valid_once')
B4 = SRC.parent / 'model329_fullstats_width4000_uniform_group_oof/valid_ensemble/probabilities.npz'
OUT = HERE / 'hard_fusion_cpu_replay'
assert not OUT.exists()
OUT.mkdir()
manifest = json.loads((BASE / 'input_manifest.json').read_text())
palette = np.asarray(manifest['config']['palette'], dtype=np.uint8)
receipt = json.loads((REF / 'training/completion.json').read_text())
checkpoint = Path(receipt['final_checkpoint'])
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()
assert sha(checkpoint) == receipt['final_sha256']
assert sha(B4) == '80ebfddad10abb1bb5f1d7af8472f477fd5b6435b3c7fef0bfa51e21e1314b05'
torch.set_num_threads(2)
model = Residual().eval()
model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict'])
def labels(p):
    return np.array([2, 0, 1, 3], dtype=np.uint8)[p[[2, 0, 1, 3]].argmax(0)]
rows = []
for kind in ['stage1', 'stage2']:
    run = OUT / kind
    (run / 'predicted_labels').mkdir(parents=True)
    (run / 'predicted_masks').mkdir()
    (run / 'input_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
with np.load(B4) as archive:
    for image in manifest['images']:
        name = image['name']
        parts = [archive[name].astype(np.float32)]
        for seed in [1, 2]:
            with np.load(SRC / f'hrnet_seed{seed}_mean_valid/{name}.npz') as z:
                parts.append(z[name].astype(np.float32))
        prob = (.5*parts[0]+.25*parts[1]+.25*parts[2])[[1,2,0,3]]
        gray = prepare_gray(np.asarray(Image.open(image['path']).convert('L'), dtype=np.float32)/255.)
        with torch.no_grad():
            value, delta = model(torch.from_numpy(np.log(np.maximum(prob,1e-7))[None]), torch.from_numpy(gray[None,None]))
        refined = labels(value.numpy()[0])
        saved = np.asarray(Image.open(REF / f'matched_source/refined/predicted_labels/{name}.png'))
        mismatch = int(np.count_nonzero(refined != saved))
        parent = np.asarray(Image.open(BASE / f'predicted_labels/{name}.png'))
        expert = labels(prob)
        stage1 = np.where(np.isin(expert, [0,2]), expert, parent).astype(np.uint8)
        stage2 = np.where(refined == 2, refined, stage1).astype(np.uint8)
        for kind, pred in [('stage1', stage1), ('stage2', stage2)]:
            Image.fromarray(pred).save(OUT / f'{kind}/predicted_labels/{name}.png')
            Image.fromarray(palette[pred]).save(OUT / f'{kind}/predicted_masks/{name}_mask.png')
        rows.append({'name':name, 'refiner_cpu_vs_gpu_changed_pixels':mismatch})
        print(name, mismatch, flush=True)
(OUT / 'completion.json').write_text(json.dumps({'images':rows,'gt_read':False,'parent_probabilities_reused':True,'fresh_raw_inference':False,'refiner_sha256':receipt['final_sha256'],'cpu_only':True},indent=2)+'\n')
