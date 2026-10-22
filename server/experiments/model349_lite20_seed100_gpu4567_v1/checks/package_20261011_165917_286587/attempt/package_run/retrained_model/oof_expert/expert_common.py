"""Expert (E) ensemble of the fusion805 stage-1 rule under the OOF fold x seed design.

E = 0.5 x B4-329 (D4 TTA; the R run's b4_329 checkpoints are reused) + 0.5 x width-normalized HRNet392
(flip4 TTA; trained here per fold x seed). Main class order [Al3Ni, Primary Si, Al, Eutectic Si].
The stage-1 rule replaces R's label by E's label wherever E predicts a class in the chosen subset.
"""
from __future__ import annotations

import gc
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent                                   # Model349_fusion805
OOF_DESIGN = PKG / 'Model349_exh14' / 'oof_design'
MODEL392 = PKG / 'model397_reproduction_bundle' / 'model392'
sys.path.insert(0, str(OOF_DESIGN))
from common import add_inference_paths  # noqa: E402

TEAM_TO_MAIN = [1, 2, 0, 3]
# subsets of classes (main order) that E may overwrite; fixed order: by size, then lexicographic
RULES = [tuple(c) for r in range(0, 5) for c in itertools.combinations(range(4), r)]
ORIGINAL_RULE = (0, 2)                              # fusion805: E's Al3Ni and Al pixels


def e_label(p_main: np.ndarray) -> np.ndarray:
    """argmax with the original team-class priority at exact ties (as in the fusion805 release)."""
    return np.array([2, 0, 1, 3], dtype=np.uint8)[p_main[[2, 0, 1, 3]].argmax(0)]


def apply_rule(r_label: np.ndarray, e_lab: np.ndarray, rule) -> np.ndarray:
    return np.where(np.isin(e_lab, list(rule)), e_lab, r_label).astype(np.uint8) if rule else r_label


def b4_329_d4(run_dir: Path, image: Path, device: str) -> np.ndarray:
    add_inference_paths()
    import members as M
    return M.b4(run_dir, image, device, tta=True)


def hrnet392(ckpt: Path, image: Path, device: str) -> np.ndarray:
    """Width-normalized HRNet392 probability with flip4 TTA (same calls as the fusion805 release)."""
    import torch
    if str(MODEL392) not in sys.path:
        sys.path.insert(0, str(MODEL392))
    for k in [m for m in sys.modules if m == 'komap_seg' or m.startswith('komap_seg.')]:
        del sys.modules[k]                           # exh14 ships a different komap_seg package
    from komap_seg.checkpoint import load_model_and_preprocessing, resolve_preprocessing
    from komap_seg.dataset import KomapFullDataset, effective_input_scale, scaled_overlap
    from komap_seg.inference import predict_probs_tta
    dev = torch.device(device)
    model, saved = load_model_and_preprocessing(ckpt, dev)
    prep = resolve_preprocessing(saved)
    ds = KomapFullDataset(image.parent, masks_dir=None)
    idx = [i for i, p in enumerate(ds.pairs) if Path(p['image']).name == image.name][0]
    sample = ds[idx]
    scale = effective_input_scale(prep['input_scale'], prep['target_width'], sample['width'])
    overlap = scaled_overlap(512, 2.0, 170, scale)
    p = predict_probs_tta(model, sample['image'], patch_size=512, overlap=overlap, batch_size=2, tta=True,
                          input_scale=scale, device=dev, tta_mode='flip4').numpy()
    del model; gc.collect(); torch.cuda.empty_cache()
    return p[TEAM_TO_MAIN].astype(np.float32)


def expert(b4_run: Path, hr_ckpt: Path, image: Path, device: str) -> np.ndarray:
    return 0.5 * b4_329_d4(b4_run, image, device) + 0.5 * hrnet392(hr_ckpt, image, device)
