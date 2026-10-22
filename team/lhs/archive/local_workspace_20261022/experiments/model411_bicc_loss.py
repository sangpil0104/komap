#!/usr/bin/env python3
"""Small-component bidirectional loss with per-component, not count, reduction.

The auxiliary follows the GT/prediction-component split of BiCC, adapted to
the two error-prone KoMaP phases and to the native-small proxy bins already
used by Models 404-410. Connected-component assignments are detached; all
loss gradients flow through the original semantic probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import model404_component_recall_loss_v4 as recall_v4


TARGET_CLASSES = recall_v4.TARGET_CLASSES
AREA_BINS = recall_v4.AREA_BINS
CONNECTIVITY = 8
PROTECTED_RADIUS = 1
GT_BRANCH_WEIGHT = 0.75
PRED_BRANCH_WEIGHT = 0.25


@dataclass(frozen=True)
class SmallBiccTargets:
    gt_components: recall_v4.ComponentTargets
    masks: torch.Tensor


@dataclass(frozen=True)
class PredictedComponents:
    component_ids: torch.Tensor
    class_ids: torch.Tensor
    batch_indices: torch.Tensor
    class_slots: torch.Tensor
    bin_indices: torch.Tensor
    areas: torch.Tensor
    num_target_classes: int
    num_bins: int

    @property
    def count(self) -> int:
        return int(self.areas.numel())


def build_component_targets(masks: torch.Tensor) -> SmallBiccTargets:
    """Build small GT components and preserve CPU masks for detached matching."""
    if masks.device.type != "cpu" or masks.ndim != 3:
        raise ValueError("expected CPU training masks with shape BxHxW")
    return SmallBiccTargets(
        gt_components=recall_v4.build_component_targets(masks),
        masks=masks.detach().clone(),
    )


def build_unmatched_predicted_components(
    logits: torch.Tensor,
    masks: torch.Tensor,
    *,
    protected_radius: int = PROTECTED_RADIUS,
) -> PredictedComponents:
    """Label small argmax components with no same-class GT within a fixed halo.

    Crop-border components are ignored because their full extent is not visible.
    No probability threshold is searched: the partition uses the same argmax
    rule as the official evaluator.
    """
    if logits.ndim != 4 or masks.ndim != 3 or masks.device.type != "cpu":
        raise ValueError("expected logits BxCxHxW and CPU masks BxHxW")
    batch_size, num_classes, height, width = logits.shape
    if num_classes != 4 or tuple(masks.shape) != (batch_size, height, width):
        raise ValueError("Model411 expects four classes and matching mask/logit grids")
    if protected_radius not in (0, 1):
        raise ValueError("protected radius is fixed to zero or one pixel")

    predictions = np.ascontiguousarray(
        logits.detach().argmax(dim=1).to(device="cpu").numpy()
    )
    mask_array = np.ascontiguousarray(masks.detach().numpy())
    compact = np.zeros((batch_size, height, width), dtype=np.int32)
    batch_indices: list[int] = []
    class_ids: list[int] = []
    class_slots: list[int] = []
    bin_indices: list[int] = []
    areas: list[int] = []
    kernel_size = 2 * protected_radius + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    next_id = 1

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
                if not 4 <= area < 256 or overlap_counts[local_id] > 0:
                    continue
                if (x == 0 or y == 0 or x + component_width >= width
                        or y + component_height >= height):
                    continue
                bin_index = next(
                    (idx for idx, (lower, upper) in enumerate(AREA_BINS)
                     if lower <= area < upper),
                    None,
                )
                if bin_index is None:
                    continue
                remap[local_id] = next_id
                batch_indices.append(batch_index)
                class_ids.append(class_id)
                class_slots.append(class_slot)
                bin_indices.append(bin_index)
                areas.append(area)
                next_id += 1
            mapped = remap[labels]
            selected = mapped > 0
            compact[batch_index, selected] = mapped[selected]

    return PredictedComponents(
        component_ids=torch.from_numpy(compact.astype(np.int64, copy=False)),
        class_ids=torch.tensor(class_ids, dtype=torch.long),
        batch_indices=torch.tensor(batch_indices, dtype=torch.long),
        class_slots=torch.tensor(class_slots, dtype=torch.long),
        bin_indices=torch.tensor(bin_indices, dtype=torch.long),
        areas=torch.tensor(areas, dtype=torch.float32),
        num_target_classes=len(TARGET_CLASSES),
        num_bins=len(AREA_BINS),
    )


def _component_cell_means(
    per_component: torch.Tensor,
    batch_indices: torch.Tensor,
    class_slots: torch.Tensor,
    bin_indices: torch.Tensor,
    *,
    batch_size: int,
    num_target_classes: int,
    num_bins: int,
) -> torch.Tensor:
    """Average within crop×class×size cells, then equally across active cells."""
    if per_component.numel() == 0:
        return per_component.sum()
    device = per_component.device
    batch_indices = batch_indices.to(device=device)
    class_slots = class_slots.to(device=device)
    bin_indices = bin_indices.to(device=device)
    cell_ids = (batch_indices * num_target_classes + class_slots) * num_bins + bin_indices
    cell_count = batch_size * num_target_classes * num_bins
    sums = torch.zeros(cell_count, dtype=per_component.dtype, device=device)
    sums.scatter_add_(0, cell_ids, per_component)
    counts = torch.zeros_like(sums)
    counts.scatter_add_(0, cell_ids, torch.ones_like(per_component))
    active = counts > 0
    return (sums[active] / counts[active]).mean()


def _gt_recall_term(logits: torch.Tensor, components: recall_v4.ComponentTargets) -> torch.Tensor:
    return recall_v4.component_recall_loss(logits, components)


def _pred_false_component_term(logits: torch.Tensor, targets: PredictedComponents) -> torch.Tensor:
    if logits.ndim != 4:
        raise ValueError("expected logits BxCxHxW")
    batch_size, num_classes, height, width = logits.shape
    if tuple(targets.component_ids.shape) != (batch_size, height, width):
        raise ValueError("predicted component target grid must match logits")
    if targets.count == 0:
        return logits.float().sum() * 0.0

    component_ids = targets.component_ids.to(
        device=logits.device, non_blocking=True
    ).reshape(-1)
    probabilities = F.softmax(logits.float(), dim=1).permute(0, 2, 3, 1).reshape(
        -1, num_classes
    )
    class_ids = targets.class_ids.to(device=logits.device)
    pixel_classes = class_ids[(component_ids - 1).clamp_min(0)]
    pixel_probability = probabilities.gather(1, pixel_classes[:, None]).squeeze(1)
    sums = torch.zeros(targets.count + 1, device=logits.device, dtype=pixel_probability.dtype)
    sums.scatter_add_(0, component_ids, pixel_probability * component_ids.gt(0))
    means = sums[1:] / targets.areas.to(
        device=logits.device, dtype=pixel_probability.dtype
    ).clamp_min(1.0)
    per_component = -torch.log1p(-means.clamp(max=1.0 - 1e-5))
    return _component_cell_means(
        per_component,
        targets.batch_indices,
        targets.class_slots,
        targets.bin_indices,
        batch_size=batch_size,
        num_target_classes=targets.num_target_classes,
        num_bins=targets.num_bins,
    )


def component_recall_loss(logits: torch.Tensor, targets: SmallBiccTargets) -> torch.Tensor:
    """Fixed 0.75 GT-recall / 0.25 prediction-FP auxiliary loss."""
    if not isinstance(targets, SmallBiccTargets):
        raise TypeError("Model411 requires CPU masks with GT component metadata")
    gt_term = _gt_recall_term(logits, targets.gt_components)
    pred_targets = build_unmatched_predicted_components(logits, targets.masks)
    fp_term = _pred_false_component_term(logits, pred_targets)
    return GT_BRANCH_WEIGHT * gt_term + PRED_BRANCH_WEIGHT * fp_term


def self_test() -> None:
    masks = torch.zeros((1, 32, 32), dtype=torch.long)
    masks[0, 12:14, 12:14] = 1  # one four-pixel GT component
    targets = build_component_targets(masks)
    assert targets.gt_components.count == 1

    logits = torch.zeros((1, 4, 32, 32), dtype=torch.float32)
    logits[0, 1, 12:14, 12:14] = 4.0  # true positive, protected by GT halo
    logits[0, 3, 3:5, 3:5] = 3.0      # unmatched four-pixel Eutectic island
    logits[0, 1, 0:2, 20:22] = 3.0    # crop-border island, ignored
    logits.requires_grad_()

    pred = build_unmatched_predicted_components(logits, masks)
    assert pred.count == 1, pred.count
    assert pred.class_ids.tolist() == [3]
    assert pred.areas.tolist() == [4.0]
    loss = component_recall_loss(logits, targets)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 3, 3, 3] > 0  # suppress the unmatched target phase
    assert torch.allclose(logits.grad[0, :, 0, 20], torch.zeros(4), atol=1e-8)

    # Duplicating an equally confident island must not double the mean loss
    # contribution of its crop×class×size-bin cell.
    pred_logits = logits.detach().clone()
    pred_logits[0, 3, 20:22, 20:22] = 3.0
    duplicated = build_unmatched_predicted_components(pred_logits, masks)
    assert duplicated.count == 2
    one_term = _pred_false_component_term(logits.detach(), pred)
    two_term = _pred_false_component_term(pred_logits, duplicated)
    assert torch.allclose(one_term, two_term, rtol=0.0, atol=1e-6), (one_term, two_term)

    dense_masks = torch.zeros((1, 512, 512), dtype=torch.long)
    dense_logits = torch.zeros((1, 4, 512, 512), dtype=torch.float32)
    for row in range(14):
        for col in range(28):
            y, x = 8 + row * 12, 8 + col * 12
            dense_logits[0, 3, y:y + 2, x:x + 2] = 8.0
    dense_targets = build_unmatched_predicted_components(dense_logits, dense_masks)
    assert dense_targets.count == 392, dense_targets.count
    print(
        "Model411 BiCC loss smoke passed: matching, border guard, gradients, "
        "mean reduction, 392-component stress."
    )


if __name__ == "__main__":
    self_test()
