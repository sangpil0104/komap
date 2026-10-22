"""Small, executable numerical contracts; not an implementation of B4 training.

Class order is ALWAYS Al, Al3Ni, PrimarySi, EutecticSi.
Metric values are fractions in [0, 1], not percentages.
"""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np
import torch
import torch.nn.functional as F

CLASS_NAMES = ("Al", "Al3Ni", "PrimarySi", "EutecticSi")
PALETTE = np.array(
    [[153, 127, 76], [25, 76, 153], [76, 178, 76], [204, 204, 204]], dtype=np.uint8
)
IGNORE_INDEX = 255


def rgb_to_ids(rgb: np.ndarray) -> np.ndarray:
    """Strict palette conversion. Never approximate an unknown color."""
    a = np.asarray(rgb)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[-1] not in (3, 4):
        raise ValueError("Expected uint8 RGB/RGBA [H,W,C].")
    if a.shape[-1] == 4:
        if not np.all(a[..., 3] == 255):
            raise ValueError("Non-opaque alpha is not supported by this contract.")
        a = a[..., :3]
    out = np.full(a.shape[:2], IGNORE_INDEX, dtype=np.uint8)
    for c, color in enumerate(PALETTE):
        out[np.all(a == color, axis=-1)] = c
    unknown = out == IGNORE_INDEX
    if np.any(unknown):
        colors = np.unique(a[unknown], axis=0)
        raise ValueError(f"Unknown mask colors: {colors[:10].tolist()}")
    return out


def ids_to_rgb(ids: np.ndarray) -> np.ndarray:
    a = np.asarray(ids)
    if a.ndim != 2 or not np.issubdtype(a.dtype, np.integer):
        raise ValueError("Expected an integer [H,W] label mask.")
    if np.any((a < 0) | (a >= 4)):
        raise ValueError("Submission mask must contain only labels 0..3.")
    return PALETTE[a]


def confusion_matrix(target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Rows=GT, columns=prediction, with ignore only on the target side."""
    t, p = np.asarray(target), np.asarray(pred)
    if t.ndim != 2 or t.shape != p.shape:
        raise ValueError("target and pred must have the same [H,W] shape.")
    if not (np.issubdtype(t.dtype, np.integer) and np.issubdtype(p.dtype, np.integer)):
        raise ValueError("Hard labels must be integer arrays.")
    if np.any(~(((t >= 0) & (t < 4)) | (t == IGNORE_INDEX))):
        raise ValueError("Unknown target label.")
    if np.any((p < 0) | (p >= 4)):
        raise ValueError("Unknown prediction label.")
    valid = t != IGNORE_INDEX
    if not np.any(valid):
        raise ValueError("All-ignore is not a valid evaluation image.")
    linear = 4 * t[valid].astype(np.int64) + p[valid].astype(np.int64)
    return np.bincount(linear, minlength=16).reshape(4, 4)


def metrics_from_cm(cm: np.ndarray) -> dict:
    a = np.asarray(cm)
    if a.shape != (4, 4) or not np.issubdtype(a.dtype, np.integer):
        raise ValueError("Expected an integer 4x4 confusion matrix.")
    if np.any(a < 0) or a.sum() <= 0:
        raise ValueError("Confusion matrix must be nonnegative and nonempty.")
    x = a.astype(np.float64)
    tp = np.diag(x)
    gt = x.sum(axis=1)
    pred = x.sum(axis=0)
    union = gt + pred - tp
    iou = np.ones(4, dtype=np.float64)
    np.divide(tp, union, out=iou, where=union > 0)
    precision = [float(tp[c] / pred[c]) if pred[c] > 0 else None for c in range(4)]
    recall = [float(tp[c] / gt[c]) if gt[c] > 0 else None for c in range(4)]
    errors = int(a.sum() - np.trace(a))
    return {
        "class_names": list(CLASS_NAMES),
        "iou": iou.tolist(),
        "image_miou": float(iou.mean()),
        "precision": precision,
        "recall": recall,
        "gt_fraction": (gt / x.sum()).tolist(),
        "pred_fraction": (pred / x.sum()).tolist(),
        "pixel_error_rate": float(errors / x.sum()),
        "error_pixels": errors,
        "Al3Ni_to_EutecticSi_count": int(a[1, 3]),
        "Al3Ni_to_EutecticSi_fraction_of_all_errors": (
            float(a[1, 3] / errors) if errors else None
        ),
        "present_class_miou_diagnostic": float(iou[gt > 0].mean()),
    }


def summarize_images(cms: Sequence[np.ndarray]) -> dict:
    if not cms:
        raise ValueError("At least one image is required.")
    rows = [metrics_from_cm(cm) for cm in cms]
    total = np.stack(cms).sum(axis=0)
    return {
        "val_image_miou": float(np.mean([r["image_miou"] for r in rows])),
        "val_global_miou": metrics_from_cm(total)["image_miou"],
        "num_images": len(rows),
    }


def _check_d4(x: torch.Tensor, k: int, flip: bool) -> None:
    if x.ndim < 2 or k not in (0, 1, 2, 3) or not isinstance(flip, bool):
        raise ValueError(
            "D4 requires at least two dimensions, k in 0..3 and bool flip."
        )


def d4_forward(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    _check_d4(x, k, flip)
    y = torch.rot90(x, k, dims=(-2, -1))
    return torch.flip(y, dims=(-1,)) if flip else y


def d4_inverse(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    _check_d4(x, k, flip)
    y = torch.flip(x, dims=(-1,)) if flip else x
    return torch.rot90(y, -k, dims=(-2, -1))


def align_context_features(
    features: torch.Tensor,
    detail_rects: torch.Tensor,
    context_rects: torch.Tensor,
    out_hw: tuple[int, int],
) -> torch.Tensor:
    """Sample features using the spec's uniform ROI-grid/align_corners=False contract.

    Rectangles: [B,4] = x0,y0,width,height in native edge coordinates.
    Sampling and coordinates use float32; gradients reach the original features.
    """
    if features.ndim != 4 or not features.is_floating_point():
        raise ValueError("Expected floating features [B,C,H,W].")
    b = features.shape[0]
    h, w = out_hw
    if h <= 0 or w <= 0:
        raise ValueError("Output dimensions must be positive.")
    d = torch.as_tensor(detail_rects, dtype=torch.float32, device=features.device)
    c = torch.as_tensor(context_rects, dtype=torch.float32, device=features.device)
    if d.shape != (b, 4) or c.shape != (b, 4):
        raise ValueError("Each batch item needs detail and context rectangles [B,4].")
    if not (torch.isfinite(d).all() and torch.isfinite(c).all()):
        raise ValueError("Non-finite rectangle.")
    if not ((d[:, 2:] > 0).all() and (c[:, 2:] > 0).all()):
        raise ValueError("Rectangle size must be positive.")
    xx = (torch.arange(w, device=features.device, dtype=torch.float32) + 0.5) / w
    yy = (torch.arange(h, device=features.device, dtype=torch.float32) + 0.5) / h
    xn = d[:, 0, None] + d[:, 2, None] * xx[None, :]
    yn = d[:, 1, None] + d[:, 3, None] * yy[None, :]
    gx = 2 * (xn - c[:, 0, None]) / c[:, 2, None] - 1
    gy = 2 * (yn - c[:, 1, None]) / c[:, 3, None] - 1
    grid = torch.stack(
        (gx[:, None, :].expand(b, h, w), gy[:, :, None].expand(b, h, w)), dim=-1
    )
    return F.grid_sample(
        features.float(),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def restore_native_probs(
    probs: torch.Tensor,
    native_hw: tuple[int, int],
    k: int = 0,
    flip: bool = False,
) -> torch.Tensor:
    """Inverse shared D4, then restore to an integer native ROI with gradients."""
    if probs.ndim != 4 or probs.shape[1] != 4 or not probs.is_floating_point():
        raise ValueError("Expected probabilities [B,4,H,W].")
    if min(native_hw) <= 0 or not torch.isfinite(probs).all():
        raise ValueError("Invalid probability tensor or output size.")
    if (probs < -1e-6).any() or (probs > 1 + 1e-6).any():
        raise ValueError("Input is not a probability map.")
    if not torch.allclose(
        probs.float().sum(1), torch.ones_like(probs[:, 0]).float(), atol=1e-4, rtol=1e-4
    ):
        raise ValueError("Class probabilities must sum to one.")
    canonical = d4_inverse(probs.float(), k, flip)
    result = F.interpolate(
        canonical, size=native_hw, mode="bilinear", align_corners=False, antialias=False
    )
    result = result.clamp_min(0)
    return result / result.sum(dim=1, keepdim=True).clamp_min(1e-12)


def relative_features(
    grayscale: torch.Tensor,
    sigmas: tuple[float, ...] = (4.0, 16.0),
    eps_std: float = 0.01,
    clip_z: float = 5.0,
    truncate: float = 3.0,
) -> torch.Tensor:
    """Native raw + Gaussian-relative channels. No mask/GT is accepted."""
    if (
        grayscale.ndim != 4
        or grayscale.shape[1] != 1
        or not grayscale.is_floating_point()
    ):
        raise ValueError("Expected grayscale [B,1,H,W].")
    if (
        not torch.isfinite(grayscale).all()
        or (grayscale < 0).any()
        or (grayscale > 1).any()
    ):
        raise ValueError("Expected finite grayscale in [0,1].")
    if (
        eps_std <= 0
        or clip_z <= 0
        or truncate <= 0
        or not sigmas
        or any(s <= 0 for s in sigmas)
    ):
        raise ValueError("Invalid relative feature parameters.")
    x = grayscale.float()
    channels = [x]
    for sigma in sigmas:
        radius = int(np.ceil(truncate * sigma))
        coords = torch.arange(-radius, radius + 1, device=x.device, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel = kernel / kernel.sum()

        def gaussian(v: torch.Tensor) -> torch.Tensor:
            mode_x = "reflect" if v.shape[-1] > radius else "replicate"
            mode_y = "reflect" if v.shape[-2] > radius else "replicate"
            v = F.pad(v, (radius, radius, 0, 0), mode=mode_x)
            v = F.conv2d(v, kernel.view(1, 1, 1, -1))
            v = F.pad(v, (0, 0, radius, radius), mode=mode_y)
            return F.conv2d(v, kernel.view(1, 1, -1, 1))

        mu = gaussian(x)
        var = (gaussian(x * x) - mu * mu).clamp_min(0)
        z = (x - mu) / torch.sqrt(var + eps_std**2)
        channels.append(z.clamp(-clip_z, clip_z) / clip_z)
    return torch.cat(channels, dim=1)
