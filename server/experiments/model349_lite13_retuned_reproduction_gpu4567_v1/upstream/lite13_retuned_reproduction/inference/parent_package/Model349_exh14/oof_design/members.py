"""Per-member probability functions shared by OOF prediction and test inference (main class order).

Settings follow the frozen Model349_exh14 inference: single-view B4/HRNet/269, TENT 5 steps (lr 1e-5)
on the high-resolution ASPP candidates with D4 768 tiles on raw + brightness-standardized input.
"""
from __future__ import annotations

import gc
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from common import add_inference_paths

add_inference_paths()
TEAM_TO_MAIN = [1, 2, 0, 3]
B4_STAGES = {'b4_322': 'b4_gray', 'b4_326': 'b4_zcdf', 'b4_329': 'b4_zcdf'}
HR_STAGES = ('hrnet336', 'hrnet347')
HIGH = {'restart51': 'restart51', 'model109': 'model109', 'seed456': 'restart59', 'pairwise': 'restart61'}
HIGH_BANDS = ['model109', 'seed456', 'pairwise', 'ema']       # brightness bands that may keep their own checkpoint
HRNET = dict(patch=512, overlap=170, batch_size=2, input_scale=2.0)
NATIVE = dict(patch=1024, stride=576, scale=1.0)


def _torch():
    import torch
    return torch


def b4(run_dir: Path, image: Path, device: str = 'cuda', tta: bool = False) -> np.ndarray:
    torch = _torch()
    mod = __import__(B4_STAGES[run_dir.name] if run_dir.name in B4_STAGES else 'b4_zcdf')
    mem = mod.load_member(str(run_dir), device)
    with torch.no_grad():
        p = mod.predict_probs([mem], mod.load_gray(str(image)), tta=tta, device=device).cpu().numpy()
    del mem; gc.collect(); torch.cuda.empty_cache()
    return p[TEAM_TO_MAIN].astype(np.float32)


def hrnet(ckpt: Path, image: Path, device: str = 'cuda') -> np.ndarray:
    torch = _torch()
    from komap_seg.checkpoint import load_model
    from komap_seg.dataset import build_eval_transform, load_image_rgb
    from komap_seg.inference import predict_probs_tta
    model = load_model(ckpt, torch.device(device))
    im = build_eval_transform()(image=load_image_rgb(image))['image'].float()
    p = predict_probs_tta(model, im, patch_size=HRNET['patch'], overlap=HRNET['overlap'], batch_size=HRNET['batch_size'],
                          tta=False, input_scale=HRNET['input_scale'], device=torch.device(device), tta_mode='flip4').numpy()
    del model; gc.collect(); torch.cuda.empty_cache()
    return p[TEAM_TO_MAIN].astype(np.float32)


def native269(ckpt: Path, image: Path, device: str = 'cuda') -> np.ndarray:
    torch = _torch()
    from model253_highres_scale_finetune import Stride1NativeStemASPP, predict_scaled
    model = Stride1NativeStemASPP().to(device)
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['model'])
    model.eval()
    p = predict_scaled(model, image, NATIVE['patch'], NATIVE['stride'], NATIVE['scale'], torch.device(device), True, 'none')
    del model; gc.collect(); torch.cuda.empty_cache()
    return np.asarray(p, np.float32)


def image_meta(image: Path, calibration: dict) -> dict:
    """GT-free per-image quantities used by the fusion (z, q, routing params, brightness band)."""
    from hard_source_search import route_params
    from highres1024_search import choose_source
    from prior_representation_search import percentile_image
    raw = np.asarray(Image.open(image).convert('RGB'), dtype=np.uint8)
    gray = raw[..., 0].astype(np.float32)
    mean, std = float(gray.mean()), float(gray.std())
    keys = SimpleNamespace(**{f'source_{k}': k for k in ['restart51'] + HIGH_BANDS})
    return {'z': ((gray - mean) / max(std, 1e-5)).astype(np.float32), 'q': percentile_image(gray).astype(np.float32),
            'params': dict(route_params(mean, std), pair_weight=0.0), 'routed': choose_source(mean, keys),
            'mean': mean, 'prior_z_stats': calibration['priors']['z'], 'prior_q_stats': calibration['priors']['q']}


def high_preaux(ckpt: Path, image: Path, calibration: dict, params: dict, device: str = 'cuda') -> np.ndarray:
    """TENT-adapted (5 steps, lr 1e-5) D4 768-tile probability on raw and standardized input, style-weighted."""
    torch = _torch()
    from style_tta_search import predict_array, transform_image
    from tent_style_search import adapt_model
    raw = np.asarray(Image.open(image).convert('RGB'), dtype=np.uint8)
    style = transform_image(raw, 'standardize', calibration['train_brightness_mean'], calibration['train_brightness_std'])
    model = adapt_model(str(ckpt), raw, torch.device(device), True, 5, 1e-5)
    hi = [predict_array(model, im, 768, 576, torch.device(device), True, 1, 'd4', 'gaussian') for im in (raw, style)]
    pre = (1 - params['style_weight']) * hi[0] + params['style_weight'] * hi[1]
    pre /= np.maximum(pre.sum(0, keepdims=True), 1e-7)
    del model; gc.collect(); torch.cuda.empty_cache()
    return pre.astype(np.float32)


def member_paths(run: Path) -> dict:
    """Checkpoint path of every member inside one runs/foldK/seedS directory."""
    out = {k: run / k for k in B4_STAGES}
    out.update({k: run / k / 'last.pt' for k in HR_STAGES})
    out['n269'] = run / 'model269' / 'best.pt'
    out.update({f'high_{k}': run / v / 'best.pt' for k, v in HIGH.items()})
    return out
