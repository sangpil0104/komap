#!/usr/bin/env python3
"""Joint tiny-GT-component recall and hard-negative island suppression loss."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

import model404_component_recall_loss_v4 as recall_v4


TARGET_CLASSES = recall_v4.TARGET_CLASSES
HARD_NEGATIVE_TOP_FRACTION = 0.001
PROTECTED_RADIUS = 1


@dataclass(frozen=True)
class JointTargets:
    components: recall_v4.ComponentTargets
    masks: torch.Tensor

    @property
    def count(self) -> int:
        return self.components.count


def build_component_targets(masks: torch.Tensor) -> JointTargets:
    if masks.device.type != "cpu":
        raise ValueError("joint targets must be built from CPU masks")
    return JointTargets(
        components=recall_v4.build_component_targets(masks),
        masks=masks.detach().clone(),
    )


def hard_negative_island_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    *,
    top_fraction: float = HARD_NEGATIVE_TOP_FRACTION,
    protected_radius: int = PROTECTED_RADIUS,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Penalize the highest target-class probabilities outside a one-pixel GT halo.

    For each crop and target class, the top 0.1% negative probabilities are
    selected without a tuned probability threshold. A one-pixel max-pool halo
    around that class's GT protects thin boundaries and immediately adjacent
    annotation pixels. The penalty is averaged equally over nonempty
    crop-by-class cells.
    """
    if logits.ndim != 4 or masks.ndim != 3:
        raise ValueError("expected logits BxCxHxW and masks BxHxW")
    batch_size, num_classes, height, width = logits.shape
    if tuple(masks.shape) != (batch_size, height, width):
        raise ValueError("hard-negative mask grid must match logits")
    if any(class_id >= num_classes for class_id in TARGET_CLASSES):
        raise ValueError("a target class is missing from logits")
    if not 0.0 < top_fraction <= 0.01:
        raise ValueError("locked hard-negative fraction must be in (0, 0.01]")
    if protected_radius < 0 or protected_radius > 2:
        raise ValueError("protected GT halo radius must be between zero and two")
    masks = masks.to(device=logits.device, non_blocking=True)
    probabilities = F.softmax(logits.float(), dim=1)
    kernel = 2 * protected_radius + 1
    penalties: list[torch.Tensor] = []
    for batch_index in range(batch_size):
        for class_id in TARGET_CLASSES:
            gt = masks[batch_index:batch_index + 1].eq(class_id).float().unsqueeze(1)
            protected = F.max_pool2d(gt, kernel_size=kernel, stride=1,
                                     padding=protected_radius).squeeze(1).bool()
            negative = ~protected[0]
            values = probabilities[batch_index, class_id][negative]
            if values.numel() == 0:
                continue
            count = max(1, int(math.ceil(top_fraction * values.numel())))
            hardest = torch.topk(values, k=count, largest=True, sorted=False).values
            penalties.append(-torch.log1p(-hardest.clamp(max=1.0 - eps)).mean())
    if not penalties:
        return logits.float().sum() * 0.0
    return torch.stack(penalties).mean()


def component_recall_loss(logits: torch.Tensor, targets: JointTargets) -> torch.Tensor:
    """Return an equally scaled sum; the outer-train calibrator sets its weight."""
    if not isinstance(targets, JointTargets):
        raise TypeError("Model406 requires JointTargets with full GT masks")
    positive = recall_v4.component_recall_loss(logits, targets.components)
    negative = hard_negative_island_loss(logits, targets.masks)
    return positive + negative


def self_test() -> None:
    masks = torch.zeros((1, 24, 24), dtype=torch.long)
    masks[0, 9:11, 9:11] = 1
    targets = build_component_targets(masks)
    logits = torch.zeros((1, 4, 24, 24), dtype=torch.float32)
    logits[0, 1, 2, 2] = 8.0  # confident isolated false-positive island
    logits[0, 1, 9, 11] = 8.0  # immediately adjacent to GT: protected halo
    logits.requires_grad_()
    joint = component_recall_loss(logits, targets)
    assert torch.isfinite(joint)
    joint.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1, 2, 2] > 0
    assert torch.allclose(logits.grad[0, :, 9, 11], torch.zeros(4), atol=1e-8)
    assert recall_v4.TARGET_CLASSES == TARGET_CLASSES
    print("Model406 smoke passed: component recall, hard-negative suppression, GT halo.")


if __name__ == "__main__":
    self_test()
