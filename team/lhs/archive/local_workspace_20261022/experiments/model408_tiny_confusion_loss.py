#!/usr/bin/env python3
"""Hard confusion hinge on tiny GT components; no prediction-side FP penalty."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import model404_component_recall_loss_v4 as recall_v4


TARGET_CLASSES = recall_v4.TARGET_CLASSES
CLASS_NAMES = recall_v4.CLASS_NAMES
# Fixed from the earlier train-OOF confusion audit, before this experiment:
# Al3Ni is chiefly confused with Al; Eutectic Si with Al3Ni and Al.
CONFUSERS = {1: (0,), 3: (0, 1)}


def build_component_targets(masks: torch.Tensor):
    """Use the existing post-augmentation 1–64 native-pixel proxy bins."""
    return recall_v4.build_component_targets(masks)


def tiny_confusion_hinge_loss(logits: torch.Tensor, targets) -> torch.Tensor:
    """Penalize only GT tiny-component pixels currently beaten by a confuser.

    For each selected GT pixel, compare its true-class logit with the largest
    fixed confuser logit and apply ``relu(confuser - true)`` (zero margin).
    Pixel losses are averaged within each GT component; components are then
    averaged within crop × target-class × size-bin cells. Correctly classified
    pixels and all non-component pixels receive no auxiliary gradient.
    """
    if logits.ndim != 4:
        raise ValueError("expected logits BxCxHxW")
    batch_size, num_classes, height, width = logits.shape
    if num_classes != 4 or tuple(targets.component_ids.shape) != (batch_size, height, width):
        raise ValueError("Model408 requires four-class logits and matching component targets")
    if targets.count == 0:
        return logits.float().sum() * 0.0

    device = logits.device
    component_ids = targets.component_ids.to(device=device, non_blocking=True).reshape(-1)
    selected = component_ids > 0
    flat_logits = logits.float().permute(0, 2, 3, 1).reshape(-1, num_classes)
    component_class_ids = targets.class_ids.to(device=device)
    pixel_class_ids = component_class_ids[(component_ids - 1).clamp_min(0)]
    pixel_true_logits = flat_logits.gather(1, pixel_class_ids[:, None]).squeeze(1)

    # The confuser set is class-specific and fixed; no prediction threshold,
    # validation tuning, component relabeling, or FP-side loss is used.
    pixel_confuser_logits = torch.where(
        pixel_class_ids == 1,
        flat_logits[:, 0],
        torch.maximum(flat_logits[:, 0], flat_logits[:, 1]),
    )
    pixel_hinge = F.relu(pixel_confuser_logits - pixel_true_logits) * selected

    component_sums = torch.zeros(
        targets.count + 1, device=device, dtype=pixel_hinge.dtype
    ).scatter_add_(0, component_ids, pixel_hinge)
    component_means = component_sums[1:] / targets.areas.to(
        device=device, dtype=pixel_hinge.dtype
    ).clamp_min(1.0)

    batch_indices = targets.batch_indices.to(device=device)
    class_slots = targets.class_slots.to(device=device)
    bin_indices = targets.bin_indices.to(device=device)
    cell_ids = (
        (batch_indices * targets.num_target_classes + class_slots) * targets.num_bins
        + bin_indices
    )
    cell_count = batch_size * targets.num_target_classes * targets.num_bins
    cell_sums = torch.zeros(cell_count, device=device, dtype=component_means.dtype)
    cell_sums.scatter_add_(0, cell_ids, component_means)
    cell_sizes = torch.zeros_like(cell_sums)
    cell_sizes.scatter_add_(0, cell_ids, torch.ones_like(component_means))
    present = cell_sizes > 0
    return (cell_sums[present] / cell_sizes[present]).mean()


def component_recall_loss(logits: torch.Tensor, targets) -> torch.Tensor:
    """Trainer-compatible entry point; treatment is the confusion hinge only."""
    return tiny_confusion_hinge_loss(logits, targets)


def self_test() -> None:
    masks = torch.zeros((1, 32, 32), dtype=torch.long)
    masks[0, 4:6, 4:6] = 1  # Al3Ni, 4 augmented pixels
    masks[0, 16:18, 16:19] = 3  # Eutectic Si, 6 augmented pixels
    targets = build_component_targets(masks)
    assert targets.count == 2

    logits = torch.zeros((1, 4, 32, 32), dtype=torch.float32)
    # One Al3Ni pixel is confused as Al; remaining pixels are already correct.
    logits[0, 0, 4, 4] = 2.0
    logits[0, 1, 4:6, 5:6] = 2.0
    logits[0, 1, 5, 4] = 2.0
    # Eutectic pixels are confused as Al3Ni, its strongest fixed confuser.
    logits[0, 1, 16:18, 16:19] = 1.5
    logits.requires_grad_()
    loss = tiny_confusion_hinge_loss(logits, targets)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1, 4, 4] < 0  # raise true Al3Ni logit
    assert logits.grad[0, 0, 4, 4] > 0  # lower confuser Al logit
    assert logits.grad[0, 3, 16, 16] < 0  # raise true Eutectic logit
    assert logits.grad[0, 1, 16, 16] > 0  # lower confuser Al3Ni logit
    assert torch.allclose(logits.grad[0, :, 10, 10], torch.zeros(4), atol=1e-8)
    assert torch.allclose(logits.grad[0, :, 4, 5], torch.zeros(4), atol=1e-8)


if __name__ == "__main__":
    self_test()
    print("Model408 tiny-confusion loss self-test passed")
