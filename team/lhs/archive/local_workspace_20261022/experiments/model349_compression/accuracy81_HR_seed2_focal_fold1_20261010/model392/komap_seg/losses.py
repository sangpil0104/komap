"""Focal gamma2 replaces CE; original Dice and boundary loss retained."""
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


def focal_ce(logits, targets, class_weights=None):
    ce = F.cross_entropy(logits, targets, weight=class_weights, reduction='none')
    pt = F.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
    return ((1-pt).square() * ce).mean()


class ComboLoss(nn.Module):
    """ce_weight * CrossEntropy + dice_weight * soft Dice."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        if class_weights is not None and tuple(class_weights.shape) != (num_classes,):
            raise ValueError(f"expected {num_classes} CE class weights, got {tuple(class_weights.shape)}")
        self.dice = DiceLoss(num_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.ce_weight * focal_ce(logits, targets, self.class_weights)
            + self.dice_weight * self.dice(logits, targets)
        )


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
