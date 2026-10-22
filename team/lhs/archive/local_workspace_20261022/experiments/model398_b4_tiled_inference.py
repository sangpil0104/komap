"""Locked B4 preprocessing + overlap-tile inference for Model398 comparisons.

This adapter is intentionally shared by the untouched parent (P0), matched
control (C), and scale treatment (T). It reproduces Model329's whole-resized-
image zcdf3 preprocessing, but replaces full-frame forwarding with fixed
overlap-tile D4 inference so every arm uses the same stitching procedure.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
B4_COMMON_PATH = ROOT / "model326_hybrid_fullstats" / "KoMaP" / "common.py"
TILE_PACKAGE_ROOT = ROOT / "model336_hrnet_widthnorm_candidate"


def _load_b4_common():
    spec = importlib.util.spec_from_file_location("model398_b4_common", B4_COMMON_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Model329-compatible preprocessing from {B4_COMMON_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if str(TILE_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(TILE_PACKAGE_ROOT))

from komap_seg.inference import predict_probs_tta  # noqa: E402

_B4_COMMON = _load_b4_common()

PATCH_SIZE = 1536
OVERLAP = 384
BATCH_SIZE = 1
TTA_MODE = "d4"


def prepare_b4_image(gray: np.ndarray, target_width: int) -> tuple[np.ndarray, torch.Tensor]:
    """Resize gray input and apply Model329's full-resized-image zcdf3 input.

    Returns the resized uint8 image and normalized float tensor (3,H,W).
    The full resized image is passed to `make_input` before any tiles are cut,
    preserving the original image-global statistics and CDF semantics.
    """
    gray = np.asarray(gray)
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("gray must be a 2-D uint8 image")
    if target_width <= 0:
        raise ValueError("target_width must be positive")

    scale = target_width / gray.shape[1]
    resized = _B4_COMMON.resize_gray(gray, scale)
    normalized = _B4_COMMON.make_input(
        resized,
        norm="image_global",
        input_mode="zcdf3",
    )
    tensor = torch.from_numpy(np.ascontiguousarray(normalized)).to(dtype=torch.float32)
    return resized, tensor


@torch.no_grad()
def predict_b4_tiled_probs(
    model: torch.nn.Module,
    gray: np.ndarray,
    *,
    target_width: int,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Return `(4,H,W)` probabilities on the original image grid.

    Model weights must already be loaded. Patch size, overlap, batch size, and
    D4 TTA are constants so P0/C/T cannot silently use different settings.
    """
    gray = np.asarray(gray)
    resized, normalized = prepare_b4_image(gray, target_width)
    resized_h, resized_w = resized.shape
    height, width = gray.shape
    scale = target_width / width

    probs = predict_probs_tta(
        model,
        normalized,
        patch_size=PATCH_SIZE,
        overlap=OVERLAP,
        batch_size=BATCH_SIZE,
        tta=True,
        input_scale=1.0,
        device=torch.device(device),
        tta_mode=TTA_MODE,
    )
    if probs.shape != (4, resized_h, resized_w):
        raise RuntimeError(
            f"unexpected tiled probability shape {tuple(probs.shape)}; "
            f"expected {(4, resized_h, resized_w)}"
        )

    if (resized_h, resized_w) != (height, width):
        if scale > 1.0:
            probs = F.interpolate(
                probs.unsqueeze(0), size=(height, width), mode="area"
            )[0]
        else:
            probs = F.interpolate(
                probs.unsqueeze(0), size=(height, width),
                mode="bilinear", align_corners=False,
            )[0]

    result = probs.float().cpu().numpy()
    if result.shape != (4, height, width) or not np.isfinite(result).all():
        raise RuntimeError("tiled inference returned invalid probabilities")
    if np.any(result < -1e-5) or np.any(result > 1.0 + 1e-5):
        raise RuntimeError("tiled inference probabilities fall outside [0,1]")
    if not np.allclose(result.sum(axis=0), 1.0, atol=2e-3, rtol=0):
        raise RuntimeError("tiled inference probabilities are not normalized")
    return result

