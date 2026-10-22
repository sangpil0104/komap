"""Technique 2/8 support: build a per-parent hard-example (error) map over the 70 official TRAIN
images, used by evolvelab/dataset.py's `hard_example` crop category (predicted != ground truth
under that parent's own plain inference). Reconstructs each parent as a plain PhaseCNN and loads
only its shared backbone/decoder/head weights (the same base-only transfer used in
evolvelab/models.py::build_model) -- a parent's own hundred_v1-style adapter, if any (only
m016_aspp_w64_staged among the 10 has one), is not replayed here; an acceptable approximation
since these maps are only a soft crop-sampling prior, not a correctness-critical target."""
import numpy as np
import torch
from tqdm import tqdm

from alphase.data.dataset import load_image, load_target
from alphase.data.splits import select_rows
from alphase.inference.predictor import predict_probabilities
from alphase.models.cnn import PhaseCNN
from alphase.utils import sha256, write_json
from evolvelab.workspace import PARENTS, HARD_MAPS, PROJECT


def _load_parent_model(path, device):
    state = torch.load(path, map_location='cpu', weights_only=False)
    model = PhaseCNN(3)
    selected = state['ema_model'] if state['selected_weights'] == 'ema' else state['model']
    base_only = {k: v for k, v in selected.items() if not k.startswith('additions.')}
    missing, unexpected = model.load_state_dict(base_only, strict=False)
    if missing or unexpected:
        raise ValueError(f'Unexpected parent weights while mining {path}: missing={missing} unexpected={unexpected}')
    return model.to(device).eval(), state['resolved_config']


def build_hard_maps():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    per_parent = {}
    for parent, path in tqdm(PARENTS.items(), desc='Parents', unit='parent', dynamic_ncols=True):
        if not path.is_file():
            raise FileNotFoundError(f'Parent checkpoint missing: {path}')
        model, cfg = _load_parent_model(path, device)
        cfg = dict(cfg)
        cfg['inference'] = dict(cfg['inference'], d4=False)
        train_rows, _ = select_rows(cfg, 'official')
        out_dir = HARD_MAPS / parent
        out_dir.mkdir(parents=True, exist_ok=True)
        files = {}
        for row in tqdm(train_rows, desc=parent, unit='image', leave=False, dynamic_ncols=True):
            gray = load_image(row['image_path']).astype(np.float32) / 255
            target = load_target(row['mask_path'])
            pred = predict_probabilities(model, gray, cfg, device).argmax(0)
            hard = pred != target
            out_path = out_dir / f"{row['image_id']}.npz"
            np.savez_compressed(out_path, hard=hard)
            files[row['image_id']] = dict(path=str(out_path.relative_to(PROJECT)), sha256=sha256(out_path),
                                           error_fraction=float(hard.mean()))
        per_parent[parent] = files
        write_json(out_dir / 'manifest.json', dict(parent=parent, images=files))
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    write_json(HARD_MAPS / 'manifest.json', dict(parents=list(PARENTS), per_parent=per_parent))
    return per_parent
