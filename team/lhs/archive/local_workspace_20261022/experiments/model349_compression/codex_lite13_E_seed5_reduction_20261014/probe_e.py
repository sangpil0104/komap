"""Per-fold E probabilities (B4 329 / 329_ws folds with D4 TTA, HRNet392 seed x fold with flip4), same calls as infer_expert.py.
   python probe_e.py PARENT_PACKAGE ASSETS INPUT_IMAGES OUT     -> OUT/<family>/<name>.npy  (team channel order, float16)"""
import sys, json
from pathlib import Path
import numpy as np, torch
PAR, ASSETS, INP, OUT = map(Path, sys.argv[1:5])
sys.path.insert(0, str(PAR / 'model397_reproduction_bundle/model329')); sys.path.insert(0, str(PAR / 'model397_reproduction_bundle/model392'))
from common import load_member, load_gray, predict_probs
from komap_seg.checkpoint import load_model_and_preprocessing, resolve_preprocessing
from komap_seg.dataset import KomapFullDataset, effective_input_scale, scaled_overlap
from komap_seg.inference import predict_probs_tta
torch.set_num_threads(2); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True; dev = 'cuda'
ds = KomapFullDataset(INP, masks_dir=None)
import os
if os.environ.get('PROBE_JOBS'):   # [[tag, 'b4'|'hr', path], ...]
    jobs = [(t, k, Path(p)) for t, k, p in json.loads(Path(os.environ['PROBE_JOBS']).read_text())]
else:
    jobs = [(f'{b}_f{f}', 'b4', ASSETS / f'{b}/fold{f}') for b in ['model329', 'model329_ws'] for f in [1, 2, 3, 4]]
    jobs += [(f'hr392_s{s}_f{f}', 'hr', ASSETS / f'model392/seed{s}/fold{f}/last.pt') for s in [1, 2] for f in [1, 2, 3, 4]]
for tag, kind, path in jobs:
    out = OUT / tag
    if (out / 'complete.json').exists(): continue
    out.mkdir(parents=True, exist_ok=True)
    if kind == 'b4': m = [load_member(path, dev, 'last.pt')]
    else:
        model, saved = load_model_and_preprocessing(path, torch.device(dev)); prep = resolve_preprocessing(saved)
    for i, pair in enumerate(ds.pairs):
        name = pair['stem']
        if kind == 'b4': v = predict_probs(m, load_gray(pair['image']), tta=True, device=dev).cpu().numpy()
        else:
            sample = ds[i]; scale = effective_input_scale(prep['input_scale'], prep['target_width'], sample['width'])
            v = predict_probs_tta(model, sample['image'], patch_size=512, overlap=scaled_overlap(512, 2., 170, scale), batch_size=2, tta=True,
                                  input_scale=scale, device=torch.device(dev), tta_mode='flip4').numpy()
        np.save(out / f'{name}.npy', v.astype(np.float16))
    (out / 'complete.json').write_text(json.dumps({'path': str(path)}) + '\n'); print('done', tag, flush=True)
    m = model = None; torch.cuda.empty_cache()
