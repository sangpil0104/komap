#!/usr/bin/env python3
"""Balanced recall loss for tiny Al3Ni and Eutectic-Si GT components."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F


TARGET_CLASSES = (1, 3)  # Al3Ni and Eutectic Si in [Al, Al3Ni, Primary Si, Eutectic Si].
CLASS_NAMES = {1: "Al3Ni", 3: "Eutectic Si"}
# With nominal 2x linear input scaling these augmented-grid bins approximate
# native-grid areas [1,64); width normalization/augmentation make this only an
# approximate correspondence. The native 64-256 px range is guard-only.
AREA_BINS = ((4, 16), (16, 64), (64, 256))


@dataclass(frozen=True)
class ComponentTargets:
    component_ids: torch.Tensor  # (B,H,W), zero means ignored
    batch_indices: torch.Tensor  # one entry per compact component ID
    class_ids: torch.Tensor  # semantic class ID, one per component
    class_slots: torch.Tensor  # balanced target-class slot, one per component
    bin_indices: torch.Tensor  # area-bin slot, one per component
    areas: torch.Tensor  # augmented-grid pixels, one per component
    num_bins: int
    num_target_classes: int

    @property
    def count(self) -> int:
        return int(self.areas.numel())


def build_component_targets(
    masks: torch.Tensor | np.ndarray,
    *,
    target_classes: tuple[int, ...] = TARGET_CLASSES,
    area_bins: tuple[tuple[int, int], ...] = AREA_BINS,
    exclude_crop_border: bool = True,
) -> ComponentTargets:
    """Label selected 8-connected classes after augmentation on the CPU."""
    if isinstance(masks, torch.Tensor):
        if masks.device.type != "cpu":
            raise ValueError("component labeling must run on the CPU before device transfer")
        mask_array = masks.detach().numpy()
    else:
        mask_array = np.asarray(masks)
    if mask_array.ndim != 3:
        raise ValueError(f"expected BxHxW masks, got shape {mask_array.shape}")
    if not target_classes or len(set(target_classes)) != len(target_classes):
        raise ValueError("target classes must be a nonempty unique sequence")
    if any(class_id not in CLASS_NAMES for class_id in target_classes):
        raise ValueError(f"only locked tiny-component classes {tuple(CLASS_NAMES)} are supported")
    if not area_bins or any(lo < 1 or hi <= lo for lo, hi in area_bins):
        raise ValueError("area bins must be nonempty positive half-open intervals")
    for previous, current in zip(area_bins, area_bins[1:]):
        if previous[1] != current[0]:
            raise ValueError("area bins must be contiguous and ordered")

    batch_size, height, width = mask_array.shape
    compact = np.zeros((batch_size, height, width), dtype=np.int32)
    batch_indices: list[int] = []
    class_ids: list[int] = []
    class_slots: list[int] = []
    bin_indices: list[int] = []
    areas: list[int] = []
    next_id = 1

    for batch_index in range(batch_size):
        for class_slot, class_id in enumerate(target_classes):
            binary = np.ascontiguousarray(mask_array[batch_index] == class_id, dtype=np.uint8)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary, connectivity=8, ltype=cv2.CV_32S
            )
            for local_id in range(1, count):
                x, y, component_width, component_height, area = map(int, stats[local_id])
                if exclude_crop_border and (
                    x == 0 or y == 0 or x + component_width >= width
                    or y + component_height >= height
                ):
                    continue
                bin_index = next(
                    (idx for idx, (lower, upper) in enumerate(area_bins)
                     if lower <= area < upper),
                    None,
                )
                if bin_index is None:
                    continue
                compact[batch_index][labels == local_id] = next_id
                batch_indices.append(batch_index)
                class_ids.append(class_id)
                class_slots.append(class_slot)
                bin_indices.append(bin_index)
                areas.append(area)
                next_id += 1

    return ComponentTargets(
        component_ids=torch.from_numpy(compact.astype(np.int64, copy=False)),
        batch_indices=torch.tensor(batch_indices, dtype=torch.long),
        class_ids=torch.tensor(class_ids, dtype=torch.long),
        class_slots=torch.tensor(class_slots, dtype=torch.long),
        bin_indices=torch.tensor(bin_indices, dtype=torch.long),
        areas=torch.tensor(areas, dtype=torch.float32),
        num_bins=len(area_bins),
        num_target_classes=len(target_classes),
    )


def component_recall_loss(
    logits: torch.Tensor,
    targets: ComponentTargets,
    *,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Equal-weight each crop × target-class × size-bin cell.

    Each GT component contributes ``-log(mean probability of its true class)``.
    Components are averaged within a crop/class/size cell, and nonempty cells
    are averaged equally. Existing CE+Dice and boundary losses remain active,
    retaining pixel-level precision and false-positive supervision.
    """
    if logits.ndim != 4:
        raise ValueError(f"expected BxCxHxW logits, got shape {tuple(logits.shape)}")
    batch_size, num_classes, height, width = logits.shape
    if any(class_id >= num_classes for class_id in TARGET_CLASSES):
        raise ValueError(f"target classes {TARGET_CLASSES} absent from {num_classes} logits")
    if tuple(targets.component_ids.shape) != (batch_size, height, width):
        raise ValueError("component target grid must match the full-resolution logits")
    if targets.count == 0:
        return logits.float().sum() * 0.0

    device = logits.device
    component_ids = targets.component_ids.to(device=device, non_blocking=True).reshape(-1)
    probabilities = F.softmax(logits.float(), dim=1).permute(0, 2, 3, 1).reshape(-1, num_classes)
    class_ids = targets.class_ids.to(device=device)
    pixel_class_ids = class_ids[(component_ids - 1).clamp_min(0)]
    pixel_probabilities = probabilities.gather(1, pixel_class_ids[:, None]).squeeze(1)
    component_probability_sum = torch.zeros(
        targets.count + 1, device=device, dtype=pixel_probabilities.dtype
    ).scatter_add_(0, component_ids, pixel_probabilities * component_ids.gt(0))
    component_mean_probability = component_probability_sum[1:] / targets.areas.to(
        device=device, dtype=pixel_probabilities.dtype
    ).clamp_min(1.0)
    per_component_loss = -torch.log(component_mean_probability.clamp(min=eps, max=1.0))

    batch_indices = targets.batch_indices.to(device=device)
    class_slots = targets.class_slots.to(device=device)
    bin_indices = targets.bin_indices.to(device=device)
    if torch.any(class_slots >= targets.num_target_classes) or torch.any(bin_indices >= targets.num_bins):
        raise ValueError("component target class/size slots exceed declared dimensions")
    cell_ids = (
        (batch_indices * targets.num_target_classes + class_slots) * targets.num_bins
        + bin_indices
    )
    cell_count = batch_size * targets.num_target_classes * targets.num_bins
    cell_sums = torch.zeros(cell_count, device=device, dtype=per_component_loss.dtype)
    cell_sums.scatter_add_(0, cell_ids, per_component_loss)
    cell_sizes = torch.zeros_like(cell_sums)
    cell_sizes.scatter_add_(0, cell_ids, torch.ones_like(per_component_loss))
    present = cell_sizes > 0
    return (cell_sums[present] / cell_sizes[present]).mean()
