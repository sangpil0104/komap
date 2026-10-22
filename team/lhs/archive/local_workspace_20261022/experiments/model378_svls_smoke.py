#!/usr/bin/env python3
"""CPU smoke/audit for the fixed Model378 2D SVLS CE target.

This file neither trains a model nor reads valid/test data. Pass --train-zip
to repeat the label-only hard-argmax preservation audit on train masks.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


PALETTE = np.asarray(
    ((153, 127, 76), (25, 76, 153), (76, 178, 76), (204, 204, 204)),
    dtype=np.uint8,
)
NUM_CLASSES = 4


def svls_kernel_2d(sigma: float = 1.0) -> torch.Tensor:
    """Official SVLS-style 3x3 Gaussian with center reset to neighbor mass."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    coords = torch.arange(-1, 2, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    neighbors_sum = 1.0 - kernel[1, 1]
    kernel[1, 1] = neighbors_sum
    kernel /= kernel.sum()
    if not torch.isclose(kernel[1, 1], torch.tensor(0.5), atol=1e-7):
        raise AssertionError(f"expected center weight 0.5, got {kernel[1, 1].item()}")
    return kernel


@torch.no_grad()
def svls_targets(labels: torch.Tensor, sigma: float = 1.0,
                 num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Return replicate-padded spatially smoothed one-hot targets (B,C,H,W)."""
    if labels.ndim != 3 or labels.dtype not in (torch.int64, torch.int32):
        raise ValueError("labels must be an integer tensor with shape (B,H,W)")
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= num_classes):
        raise ValueError("labels contain an id outside [0, num_classes)")
    onehot = F.one_hot(labels.long(), num_classes).permute(0, 3, 1, 2).float()
    kernel = svls_kernel_2d(sigma).to(device=labels.device, dtype=onehot.dtype)
    weights = kernel[None, None].expand(num_classes, 1, 3, 3).contiguous()
    return F.conv2d(F.pad(onehot, (1, 1, 1, 1), mode="replicate"),
                    weights, groups=num_classes)


def svls_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor,
                       pixel_weights: torch.Tensor | None = None) -> torch.Tensor:
    """Soft-target CE; optional weights have shape (B,H,W), as in parent CE."""
    if logits.shape != soft_targets.shape or logits.ndim != 4:
        raise ValueError("logits and soft targets must share shape (B,C,H,W)")
    per_pixel = -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1)
    if pixel_weights is not None:
        if pixel_weights.shape != per_pixel.shape:
            raise ValueError("pixel_weights must have shape (B,H,W)")
        per_pixel = per_pixel * pixel_weights
    return per_pixel.mean()


def smoke_test() -> dict:
    labels = torch.zeros((1, 7, 7), dtype=torch.long)
    labels[:, :, 3:] = 3
    labels[:, 2:5, 0:2] = 2
    targets = svls_targets(labels)
    if not torch.allclose(targets.sum(dim=1), torch.ones_like(labels, dtype=torch.float32),
                          atol=1e-6):
        raise AssertionError("soft targets do not sum to one")
    homogeneous = torch.full((1, 5, 5), 2, dtype=torch.long)
    homogeneous_targets = svls_targets(homogeneous)
    if not torch.allclose(homogeneous_targets[:, 2, 2, 2], torch.ones(1), atol=1e-6):
        raise AssertionError("homogeneous interior target is not one-hot")
    logits = torch.randn((1, NUM_CLASSES, 7, 7), requires_grad=True)
    pixel_weights = torch.ones((1, 7, 7))
    loss = svls_cross_entropy(logits, targets, pixel_weights)
    loss.backward()
    if not torch.isfinite(loss) or logits.grad is None or not torch.isfinite(logits.grad).all():
        raise AssertionError("SVLS CE produced a nonfinite loss or gradient")
    return {
        "kernel": svls_kernel_2d().tolist(),
        "center_weight": float(svls_kernel_2d()[1, 1]),
        "kernel_sum": float(svls_kernel_2d().sum()),
        "simplex_max_abs_error": float((targets.sum(1) - 1.0).abs().max()),
        "smoke_loss": float(loss.detach()),
        "gradient_finite": True,
    }


def decode_mask(rgb: np.ndarray, path: str) -> np.ndarray:
    ids = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        ids[np.all(rgb == color, axis=2)] = class_id
    if np.any(ids == 255):
        raise ValueError(f"unknown mask color in {path}")
    return ids


def audit_train_zip(path: Path) -> dict:
    masks = []
    with zipfile.ZipFile(path) as archive:
        names = sorted(name for name in archive.namelist()
                       if name.startswith("train/masks/") and name.endswith("_mask.png"))
        for name in names:
            rgb = np.asarray(Image.open(io.BytesIO(archive.read(name))).convert("RGB"))
            masks.append((name, decode_mask(rgb, name)))
    if len(masks) != 70:
        raise ValueError(f"expected 70 train masks, found {len(masks)}")

    kernel = svls_kernel_2d().numpy()
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    flips = np.zeros(NUM_CLASSES, dtype=np.int64)
    ties = np.zeros(NUM_CLASSES, dtype=np.int64)
    simplex_error = 0.0
    for name, ids in masks:
        onehot = np.eye(NUM_CLASSES, dtype=np.float32)[ids].transpose(2, 0, 1)
        padded = np.pad(onehot, ((0, 0), (1, 1), (1, 1)), mode="edge")
        soft = np.zeros_like(onehot)
        height, width = ids.shape
        for y in range(3):
            for x in range(3):
                soft += padded[:, y:y + height, x:x + width] * kernel[y, x]
        simplex_error = max(simplex_error, float(np.max(np.abs(soft.sum(0) - 1.0))))
        maximum = soft.max(axis=0)
        own = np.take_along_axis(soft, ids[None], axis=0)[0]
        strict_flip = own < maximum - 1e-7
        tie = ((np.abs(own - maximum) <= 1e-7)
               & (np.sum(np.abs(soft - maximum[None]) <= 1e-7, axis=0) > 1))
        for class_id in range(NUM_CLASSES):
            region = ids == class_id
            counts[class_id] += int(region.sum())
            flips[class_id] += int(np.count_nonzero(region & strict_flip))
            ties[class_id] += int(np.count_nonzero(region & tie))

    return {
        "scope": "train.zip masks only; label-target preservation diagnostic, not model evaluation",
        "images": len(masks),
        "simplex_max_abs_error": simplex_error,
        "strict_argmax_flip_count_by_class": flips.tolist(),
        "strict_argmax_flip_pct_by_class": (100.0 * flips / counts).tolist(),
        "tie_count_by_class": ties.tolist(),
        "tie_pct_by_class": (100.0 * ties / counts).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-zip", type=Path)
    args = parser.parse_args()
    result = {"smoke_test": smoke_test()}
    if args.train_zip:
        result["train_mask_audit"] = audit_train_zip(args.train_zip)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
