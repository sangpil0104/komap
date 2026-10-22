"""Segmentation loss (CE + Dice, no class weights) and the boundary-head loss.

No inverse-frequency class weights on purpose: they shift the decision
boundary toward the rare phases, which showed up as Eutectic Si bleeding
~1px past every phase boundary (predicted area 1.35x ground truth). Dice
averaged over classes already keeps the rare phases in play.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from komap_seg.palette import NUM_CLASSES


class DiceLoss(nn.Module):
    """Multi-class soft Dice, averaged over classes."""

    def __init__(self, num_classes: int = NUM_CLASSES, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        onehot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * onehot).sum(dims)
        cardinality = (probs + onehot).sum(dims)
        dice = (2 * intersection + self.eps) / (cardinality + self.eps)
        return 1 - dice.mean()


class ComboLoss(nn.Module):
    """ce_weight * CrossEntropy + dice_weight * soft Dice."""

    def __init__(self, num_classes: int = NUM_CLASSES, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.dice = DiceLoss(num_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                ce_pixel_weights: torch.Tensor | None = None) -> torch.Tensor:
        if ce_pixel_weights is None:
            ce = F.cross_entropy(logits, targets)
        else:
            ce_pixels = F.cross_entropy(logits, targets, reduction="none")
            ce = (ce_pixels * ce_pixel_weights).mean()
        return self.ce_weight * ce + self.dice_weight * self.dice(logits, targets)


def boundary_ce_pixel_weights(targets: torch.Tensor, weight: float = 2.0,
                              radius: int = 3, sigma: float = 1.5) -> torch.Tensor:
    """Normalized per-image CE weights based on distance from a GT class edge.

    ``targets`` is the already augmented ``(B,H,W)`` patch mask. Pixels within
    ``radius`` of a 4-neighbor class transition receive a Gaussian boost. Each
    image's weights are normalized to mean 1, preserving the CE scale.
    """
    if weight < 0 or radius < 0 or sigma <= 0:
        raise ValueError("boundary CE weight/radius must be nonnegative and sigma positive")
    mask = targets.long()
    edge = torch.zeros_like(mask, dtype=torch.bool)
    edge[:, 1:, :] |= mask[:, 1:, :] != mask[:, :-1, :]
    edge[:, :-1, :] |= mask[:, :-1, :] != mask[:, 1:, :]
    edge[:, :, 1:] |= mask[:, :, 1:] != mask[:, :, :-1]
    edge[:, :, :-1] |= mask[:, :, :-1] != mask[:, :, 1:]

    distance = torch.full_like(mask, radius + 1, dtype=torch.float32)
    distance[edge] = 0.0
    frontier = edge[:, None].float()
    for step in range(1, radius + 1):
        expanded = F.max_pool2d(frontier, kernel_size=3, stride=1, padding=1) > 0
        newly_reached = expanded[:, 0] & (distance > radius)
        distance[newly_reached] = float(step)
        frontier = newly_reached[:, None].float()
    near_edge = distance <= radius
    weights = 1.0 + weight * torch.exp(-distance.square() / (2.0 * sigma * sigma)) * near_edge
    weights = weights / weights.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return weights


def directional_unlikelihood_loss(logits: torch.Tensor, targets: torch.Tensor,
                                  target_class: int = 3,
                                  suppressed_class: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Suppress one competing class on one side of a directed GT interface.

    Support is restricted to target-class pixels with an axial 4-neighbor of
    ``suppressed_class``. The stable term is
    logsumexp(all logits) - logsumexp(logits excluding suppressed_class),
    equivalent to ``-log(1 - p_suppressed_class)``. It is averaged over the
    entire patch, not only support pixels. Returns the loss and support-pixel
    count; callers can log the latter without affecting the objective.
    """
    if logits.ndim != 4 or targets.ndim != 3:
        raise ValueError("expected logits (B,C,H,W) and targets (B,H,W)")
    if logits.shape[0] != targets.shape[0] or logits.shape[-2:] != targets.shape[-2:]:
        raise ValueError("logit and target batch/spatial dimensions differ")
    num_classes = logits.shape[1]
    if not (0 <= target_class < num_classes and 0 <= suppressed_class < num_classes):
        raise ValueError("class index outside logit channels")
    if target_class == suppressed_class or num_classes < 2:
        raise ValueError("target and suppressed classes must be distinct")

    labels = targets.long()
    neighboring_competitor = torch.zeros_like(labels, dtype=torch.bool)
    neighboring_competitor[:, :-1, :] |= labels[:, 1:, :] == suppressed_class
    neighboring_competitor[:, 1:, :] |= labels[:, :-1, :] == suppressed_class
    neighboring_competitor[:, :, :-1] |= labels[:, :, 1:] == suppressed_class
    neighboring_competitor[:, :, 1:] |= labels[:, :, :-1] == suppressed_class
    support = (labels == target_class) & neighboring_competitor

    stable_logits = logits.float()
    all_log_partition = torch.logsumexp(stable_logits, dim=1)
    keep_channels = [index for index in range(num_classes) if index != suppressed_class]
    other_log_partition = torch.logsumexp(stable_logits[:, keep_channels], dim=1)
    per_pixel = all_log_partition - other_log_partition
    loss = (per_pixel * support.to(per_pixel.dtype)).mean()
    return loss, support.sum()


def gt_boundary_band(mask: torch.Tensor, num_classes: int = NUM_CLASSES, width: int = 1) -> torch.Tensor:
    """(B, H, W) class mask -> (B, H, W) float {0, 1}: pixels whose
    (2*width+1)^2 window holds more than one class. A step edge yields a
    band 2*width px wide (width px on each side)."""
    onehot = F.one_hot(mask, num_classes).permute(0, 3, 1, 2).float()
    present = F.max_pool2d(onehot, 2 * width + 1, stride=1, padding=width)
    return (present.sum(1) > 1).float()


def _soft_dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    inter = (pred * target).sum()
    return 1 - (2 * inter + eps) / (pred.sum() + target.sum() + eps)


class BoundaryHeadLoss(nn.Module):
    """BCE (positives reweighted by the batch's neg/pos ratio) + Dice between
    the 1-channel boundary logits and the GT boundary band. Class-agnostic,
    so it can't push decision boundaries toward any particular phase."""

    def __init__(self, num_classes: int = NUM_CLASSES, width: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.width = width

    def forward(self, boundary_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logit = boundary_logits.float().squeeze(1)
        gt = gt_boundary_band(targets, self.num_classes, self.width)
        pos = gt.sum().clamp(min=1.0)
        pos_weight = (gt.numel() - pos) / pos
        bce = F.binary_cross_entropy_with_logits(logit, gt, pos_weight=pos_weight)
        return bce + _soft_dice(torch.sigmoid(logit), gt)
