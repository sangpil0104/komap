#!/usr/bin/env python3
"""GT-component recall plus prediction-connected-component false-positive loss."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import model404_component_recall_loss_v4 as recall_v4


TARGET_CLASSES = recall_v4.TARGET_CLASSES
PROTECTED_RADIUS = 1
CONNECTIVITY = 8


@dataclass(frozen=True)
class JointTargets:
    components: recall_v4.ComponentTargets
    masks: torch.Tensor

    @property
    def count(self) -> int:
        return self.components.count


@dataclass(frozen=True)
class FalseComponentTargets:
    component_ids: torch.Tensor  # (B,H,W), zero means not penalized
    batch_indices: torch.Tensor
    class_ids: torch.Tensor
    class_slots: torch.Tensor
    areas: torch.Tensor
    num_target_classes: int

    @property
    def count(self) -> int:
        return int(self.areas.numel())


def build_component_targets(masks: torch.Tensor) -> JointTargets:
    if masks.device.type != "cpu":
        raise ValueError("Model407 targets must be built from CPU masks")
    return JointTargets(
        components=recall_v4.build_component_targets(masks),
        masks=masks.detach().clone(),
    )


def build_false_component_targets(
    logits: torch.Tensor,
    masks: torch.Tensor,
    *,
    protected_radius: int = PROTECTED_RADIUS,
    exclude_crop_border: bool = True,
) -> FalseComponentTargets:
    """Partition detached argmax islands and select unmatched interior components.

    A predicted target-class component is protected if it touches the same-class
    one-pixel GT halo or the crop border. The argmax partition is discrete and
    detached; gradients later flow through the selected component probabilities.
    No probability threshold or component-area cutoff is used.
    """
    if logits.ndim != 4 or masks.ndim != 3:
        raise ValueError("expected logits BxCxHxW and CPU masks BxHxW")
    if masks.device.type != "cpu":
        raise ValueError("predicted-component targets require CPU masks")
    batch_size, num_classes, height, width = logits.shape
    if tuple(masks.shape) != (batch_size, height, width):
        raise ValueError("mask grid must match logits")
    if any(class_id >= num_classes for class_id in TARGET_CLASSES):
        raise ValueError("a target class is missing from logits")
    if protected_radius < 0 or protected_radius > 2:
        raise ValueError("protected GT halo radius must be between zero and two")

    predictions = np.ascontiguousarray(
        logits.detach().argmax(dim=1).to(device="cpu").numpy()
    )
    mask_array = np.ascontiguousarray(masks.detach().numpy())
    compact = np.zeros((batch_size, height, width), dtype=np.int32)
    batch_indices: list[int] = []
    class_ids: list[int] = []
    class_slots: list[int] = []
    areas: list[int] = []
    next_id = 1
    kernel_size = 2 * protected_radius + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    for batch_index in range(batch_size):
        for class_slot, class_id in enumerate(TARGET_CLASSES):
            predicted = np.ascontiguousarray(
                predictions[batch_index] == class_id, dtype=np.uint8
            )
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                predicted, connectivity=CONNECTIVITY, ltype=cv2.CV_32S
            )
            gt = np.ascontiguousarray(mask_array[batch_index] == class_id, dtype=np.uint8)
            halo = cv2.dilate(gt, kernel) if protected_radius else gt
            overlap_counts = np.bincount(labels[halo != 0], minlength=count)
            remap = np.zeros(count, dtype=np.int32)

            for local_id in range(1, count):
                x, y, component_width, component_height, area = map(
                    int, stats[local_id]
                )
                if area <= 0:
                    raise ValueError("OpenCV returned a zero-area predicted component")
                if overlap_counts[local_id] > 0:
                    continue
                if exclude_crop_border and (
                    x == 0 or y == 0 or x + component_width >= width
                    or y + component_height >= height
                ):
                    continue
                remap[local_id] = next_id
                batch_indices.append(batch_index)
                class_ids.append(class_id)
                class_slots.append(class_slot)
                areas.append(area)
                next_id += 1

            mapped = remap[labels]
            selected = mapped > 0
            compact[batch_index, selected] = mapped[selected]

    return FalseComponentTargets(
        component_ids=torch.from_numpy(compact.astype(np.int64, copy=False)),
        batch_indices=torch.tensor(batch_indices, dtype=torch.long),
        class_ids=torch.tensor(class_ids, dtype=torch.long),
        class_slots=torch.tensor(class_slots, dtype=torch.long),
        areas=torch.tensor(areas, dtype=torch.float32),
        num_target_classes=len(TARGET_CLASSES),
    )


def predicted_component_fp_loss(
    logits: torch.Tensor,
    targets: FalseComponentTargets,
    *,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Penalize every unmatched predicted component, with count-sensitive cells.

    Each selected component contributes ``-log(1 - mean p_target)``. Component
    penalties are summed within each crop-by-target-class cell, then averaged
    over all cells (including empty cells). Thus crops containing more false
    islands receive proportionally more suppression pressure.
    """
    if logits.ndim != 4:
        raise ValueError("expected BxCxHxW logits")
    batch_size, num_classes, height, width = logits.shape
    if tuple(targets.component_ids.shape) != (batch_size, height, width):
        raise ValueError("false-component target grid must match logits")
    if any(class_id >= num_classes for class_id in TARGET_CLASSES):
        raise ValueError("a target class is missing from logits")
    if targets.count == 0:
        return logits.float().sum() * 0.0

    device = logits.device
    component_ids = targets.component_ids.to(
        device=device, non_blocking=True
    ).reshape(-1)
    probabilities = (
        F.softmax(logits.float(), dim=1)
        .permute(0, 2, 3, 1)
        .reshape(-1, num_classes)
    )
    component_class_ids = targets.class_ids.to(device=device)
    pixel_class_ids = component_class_ids[(component_ids - 1).clamp_min(0)]
    pixel_probabilities = probabilities.gather(
        1, pixel_class_ids[:, None]
    ).squeeze(1)
    probability_sums = torch.zeros(
        targets.count + 1, device=device, dtype=pixel_probabilities.dtype
    ).scatter_add_(
        0, component_ids, pixel_probabilities * component_ids.gt(0)
    )
    component_means = probability_sums[1:] / targets.areas.to(
        device=device, dtype=pixel_probabilities.dtype
    ).clamp_min(1.0)
    penalties = -torch.log1p(-component_means.clamp(max=1.0 - eps))

    batch_indices = targets.batch_indices.to(device=device)
    class_slots = targets.class_slots.to(device=device)
    if torch.any(batch_indices >= batch_size):
        raise ValueError("predicted-component batch index exceeds logits batch")
    if torch.any(class_slots >= targets.num_target_classes):
        raise ValueError("predicted-component class slot exceeds declared dimensions")
    cell_ids = batch_indices * targets.num_target_classes + class_slots
    cell_count = batch_size * targets.num_target_classes
    cell_sums = torch.zeros(cell_count, device=device, dtype=penalties.dtype)
    cell_sums.scatter_add_(0, cell_ids, penalties)
    return cell_sums.mean()


def component_recall_loss(logits: torch.Tensor, targets: JointTargets) -> torch.Tensor:
    """Keep balanced GT recall and replace pixel hard-negatives with CC FP loss."""
    if not isinstance(targets, JointTargets):
        raise TypeError("Model407 requires JointTargets with full CPU GT masks")
    positive = recall_v4.component_recall_loss(logits, targets.components)
    false_targets = build_false_component_targets(logits, targets.masks)
    negative = predicted_component_fp_loss(logits, false_targets)
    return positive + negative


def self_test() -> None:
    masks = torch.zeros((1, 24, 24), dtype=torch.long)
    masks[0, 9:11, 9:11] = 1
    logits = torch.zeros((1, 4, 24, 24), dtype=torch.float32)
    logits[0, 1, 2, 2] = 8.0   # isolated unmatched interior component
    logits[0, 1, 9, 11] = 8.0  # same-class GT halo: protected
    logits[0, 1, 0, 17] = 8.0  # crop-border component: protected
    logits.requires_grad_()
    targets = build_component_targets(masks)
    fp_targets = build_false_component_targets(logits, masks)
    assert fp_targets.count == 1
    assert fp_targets.areas.tolist() == [1.0]
    one_component_penalty = predicted_component_fp_loss(logits.detach(), fp_targets)
    two_islands = logits.detach().clone()
    two_islands[0, 1, 5, 5] = 8.0
    two_targets = build_false_component_targets(two_islands, masks)
    two_component_penalty = predicted_component_fp_loss(two_islands, two_targets)
    assert two_targets.count == 2
    assert two_component_penalty > one_component_penalty * 1.8
    loss = component_recall_loss(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1, 2, 2] > 0
    assert torch.allclose(logits.grad[0, :, 9, 11], torch.zeros(4), atol=1e-8)
    assert torch.allclose(logits.grad[0, :, 0, 17], torch.zeros(4), atol=1e-8)

    no_islands = torch.zeros((1, 4, 24, 24), dtype=torch.float32, requires_grad=True)
    empty_targets = build_false_component_targets(no_islands, masks)
    empty_loss = predicted_component_fp_loss(no_islands, empty_targets)
    assert empty_targets.count == 0 and empty_loss.item() == 0.0
    empty_loss.backward()
    assert torch.isfinite(no_islands.grad).all()
    print("Model407 smoke passed: GT-halo/border protection and CC FP gradients.")


if __name__ == "__main__":
    self_test()
