"""Read-only checks: no forward pass, optimizer step, or TENT adaptation."""
import gc
import importlib.metadata
import os
import sys

from common import ROOT, SOURCE, VALID, inference, read_json, sha, gpu_inventory, require_komap


def check(load_models=False, require_idle=False):
    require_komap()
    selected = gpu_inventory(require_idle)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(g['uuid'] for g in selected)
    import numpy as np
    import torch
    import cv2
    from PIL import Image
    from tqdm import tqdm
    frozen = inference()
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 6:
        raise RuntimeError('All six requested GPUs must be visible to conda komap.')
    recipe = read_json(SOURCE / 'recipe_manifest.json')
    for name, digest in recipe['files_sha256'].items():
        if sha(SOURCE / name) != digest:
            raise ValueError(f'GitHub recipe hash mismatch: {name}')
    for row in tqdm(read_json(ROOT / 'local_weights.json'), desc='Verify FP32 weights / B4 settings', unit='file'):
        if sha(ROOT / row['path']) != row['sha256']:
            raise ValueError(f"Local asset hash mismatch: {row['path']}")
    weights = read_json(SOURCE / 'checkpoint_manifest.json')
    if weights['count'] != 14 or len(weights['checkpoints']) != 14:
        raise ValueError('Expected the original 14-checkpoint exh14 manifest.')
    for row in weights['checkpoints']:
        if (SOURCE / row['path']).stat().st_size != row['bytes'] or sha(SOURCE / row['path']) != row['sha256']:
            raise ValueError(f"Release checkpoint mismatch: {row['path']}")
    data = read_json(ROOT / 'data_manifest.json')
    for row in data['files']:
        if sha(ROOT / row['path']) != row['sha256']:
            raise ValueError(f"Validation data hash mismatch: {row['path']}")
    config = frozen.load_config(SOURCE / 'configs/model349_lite.yaml')
    images = sorted((VALID / 'images').glob('*.png'))
    if len(images) != 20 or len(list((VALID / 'masks').glob('*.png'))) != 20:
        raise ValueError('Expected exactly 20 validation image/mask pairs.')
    for path in tqdm(images, desc='Validate image/mask pairs', unit='image'):
        name = path.stem.removesuffix('_image')
        labels = frozen.decode_gt(VALID / 'masks' / f'{name}_mask.png', config['palette'])
        with Image.open(path) as im:
            if labels.shape != (im.height, im.width):
                raise ValueError(f'Image/mask shape mismatch: {name}')
    # Import every active dependency, including the HRNet builder, before starting a run.
    import b4_gray, b4_zcdf
    import model335_exhaustive_b4_route_search
    from komap_seg.checkpoint import load_model
    from tent_style_search import load_aspp
    from model253_highres_scale_finetune import Stride1NativeStemASPP
    loaded = []
    if load_models:
        for row in tqdm(weights['checkpoints'], desc='Strict CPU model loading (no forward)', unit='model'):
            path = SOURCE / row['path']
            payload = torch.load(path, map_location='cpu', weights_only=False)
            state = payload.get('state_dict', payload.get('model', payload))
            floating = [t for t in state.values() if torch.is_tensor(t) and torch.is_floating_point(t)]
            if not floating or not all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in floating):
                raise ValueError(f'Expected finite FP32 parameters: {path}')
            if 'fold' in path.parent.name:
                loader = b4_gray.load_member if 'model322' in str(path) else b4_zcdf.load_member
                model = loader(str(path.parent), 'cpu')['model']
            elif 'hrnet' in str(path):
                model = load_model(path, torch.device('cpu'))
            elif 'model269' in str(path):
                model = Stride1NativeStemASPP()
                model.load_state_dict(payload['model'])
            else:
                model = load_aspp(path, torch.device('cpu'))
            loaded.append(row['path'])
            del model, payload, state, floating
            gc.collect()
    return dict(ok=True, python=sys.executable, gpus=selected, validation_count=20,
                original_weight_hashes_match=True, checkpoint_count=14,
                source_recipe_hashes_match=True, cpu_strict_loaded=loaded,
                training_started=False, inference_started=False, forward_calls=0,
                versions={p: importlib.metadata.version(p) for p in
                          ['torch', 'torchvision', 'segmentation-models-pytorch', 'timm',
                           'numpy', 'Pillow', 'albumentations', 'tqdm', 'PyYAML']})
