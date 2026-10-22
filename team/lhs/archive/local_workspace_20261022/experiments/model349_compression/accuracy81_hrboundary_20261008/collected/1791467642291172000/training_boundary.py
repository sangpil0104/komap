"""Original HR336 boundary objective, isolated from immutable release source."""
import torch
import torch.nn as nn
import torch.nn.functional as F
NUM_CLASSES=4

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

def forward_with_boundary(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One encoder+decoder pass -> (seg_logits (B,C,H,W), boundary_logits (B,1,H,W))."""
    features = model.encoder(x)
    decoder_output = model.decoder(features)
    return model.segmentation_head(decoder_output), model.boundary_head(decoder_output)
