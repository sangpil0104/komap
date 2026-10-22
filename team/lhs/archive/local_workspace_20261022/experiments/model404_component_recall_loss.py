#!/usr/bin/env python3
"""Component-normalized recall loss for tiny Eutectic-Si regions.

Component targets are built from the already-augmented training masks only.
Components that touch a patch edge are ignored to avoid treating crop-truncated
fragments as complete objects. No validation/test data is needed by this code.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F


TARGET_CLASS = 3  # class order: Al, Al3Ni, Primary Si, Eutectic Si
# At nominal 2x linear training scale these bins approximately target native
# 1-64 px components; the 64-256 px native range is deliberately excluded.
AREA_BINS = ((4, 16), (16, 64), (64, 256))


@dataclass(frozen=True)
class ComponentTargets:
    """Compact IDs for eligible GT components and their crop/bin metadata."""

    component_ids: torch.Tensor  # (B,H,W), 0 is ignored/background
    batch_indices: torch.Tensor  # one entry per compact component ID
    bin_indices: torch.Tensor  # one entry per compact component ID
    areas: torch.Tensor  # augmented-grid pixels, one entry per component
    num_bins: int

    @property
    def count(self) -> int:
        return int(self.areas.numel())


def build_component_targets(
    masks: torch.Tensor | np.ndarray,
    *,
    target_class: int = TARGET_CLASS,
    area_bins: tuple[tuple[int, int], ...] = AREA_BINS,
    exclude_crop_border: bool = True,
) -> ComponentTargets:
    """Label 8-connected target-class regions after augmentation.

    The area intervals are half-open, measured on the 512x512 training mask.
    Under the nominal 2x linear training scale, the selected 4..256 pixel
    range approximately covers 1..64 native pixels. This is an approximation
    because per-image width normalization and geometric augmentation change
    the exact mapping.
    """
    if isinstance(masks, torch.Tensor):
        if masks.device.type != "cpu":
            raise ValueError("component labeling must run on the CPU before device transfer")
        mask_array = masks.detach().numpy()
    else:
        mask_array = np.asarray(masks)
    if mask_array.ndim != 3:
        raise ValueError(f"expected BxHxW masks, got shape {mask_array.shape}")
    if not area_bins or any(lo < 1 or hi <= lo for lo, hi in area_bins):
        raise ValueError("area bins must be nonempty positive half-open intervals")
    for previous, current in zip(area_bins, area_bins[1:]):
        if previous[1] != current[0]:
            raise ValueError("area bins must be contiguous and ordered")

    batch_size, height, width = mask_array.shape
    compact = np.zeros((batch_size, height, width), dtype=np.int32)
    batch_indices: list[int] = []
    bin_indices: list[int] = []
    areas: list[int] = []
    next_id = 1

    for batch_index in range(batch_size):
        binary = np.ascontiguousarray(mask_array[batch_index] == target_class, dtype=np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8, ltype=cv2.CV_32S
        )
        for local_id in range(1, count):
            x, y, component_width, component_height, area = map(int, stats[local_id])
            if exclude_crop_border and (
                x == 0 or y == 0 or x + component_width >= width or y + component_height >= height
            ):
                continue
            bin_index = next(
                (idx for idx, (lower, upper) in enumerate(area_bins) if lower <= area < upper),
                None,
            )
            if bin_index is None:
                continue
            compact[batch_index][labels == local_id] = next_id
            batch_indices.append(batch_index)
            bin_indices.append(bin_index)
            areas.append(area)
            next_id += 1

    return ComponentTargets(
        component_ids=torch.from_numpy(compact.astype(np.int64, copy=False)),
        batch_indices=torch.tensor(batch_indices, dtype=torch.long),
        bin_indices=torch.tensor(bin_indices, dtype=torch.long),
        areas=torch.tensor(areas, dtype=torch.float32),
        num_bins=len(area_bins),
    )


def component_recall_loss(
    logits: torch.Tensor,
    targets: ComponentTargets,
    *,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Equalize small-component recall pressure within each crop/size bin.

    For each eligible component, compute ``-log(mean p(target_class))``.
    Components are averaged within each (crop, area-bin) cell; nonempty cells
    then receive equal weight. Thus a bin with many components does not drown
    out another bin or crop. The existing CE+Dice objective still supplies
    pixel/precision supervision and controls false positives.
    """
    if logits.ndim != 4:
        raise ValueError(f"expected BxCxHxW logits, got shape {tuple(logits.shape)}")
    batch_size, num_classes, height, width = logits.shape
    if not 0 <= TARGET_CLASS < num_classes:
        raise ValueError(f"target class {TARGET_CLASS} absent from {num_classes} logits")
    if tuple(targets.component_ids.shape) != (batch_size, height, width):
        raise ValueError("component target grid must match the full-resolution logits")
    if targets.count == 0:
        return logits.float().sum() * 0.0

    device = logits.device
    component_ids = targets.component_ids.to(device=device, non_blocking=True).reshape(-1)
    class_probability = F.softmax(logits.float(), dim=1)[:, TARGET_CLASS].reshape(-1)
    component_probability_sum = torch.zeros(
        targets.count + 1, device=device, dtype=class_probability.dtype
    ).scatter_add_(0, component_ids, class_probability * component_ids.gt(0))
    component_mean_probability = component_probability_sum[1:] / targets.areas.to(
        device=device, dtype=class_probability.dtype
    ).clamp_min(1.0)
    per_component_loss = -torch.log(component_mean_probability.clamp(min=eps, max=1.0))

    batch_indices = targets.batch_indices.to(device=device)
    bin_indices = targets.bin_indices.to(device=device)
    num_bins = targets.num_bins
    if torch.any(bin_indices >= num_bins):
        raise ValueError("component target bins differ from the fixed Model404 bins")
    cell_ids = batch_indices * num_bins + bin_indices
    cell_count = batch_size * num_bins
    cell_sums = torch.zeros(cell_count, device=device, dtype=per_component_loss.dtype)
    cell_sums.scatter_add_(0, cell_ids, per_component_loss)
    cell_sizes = torch.zeros_like(cell_sums)
    cell_sizes.scatter_add_(0, cell_ids, torch.ones_like(per_component_loss))
    present = cell_sizes > 0
    return (cell_sums[present] / cell_sizes[present]).mean()
