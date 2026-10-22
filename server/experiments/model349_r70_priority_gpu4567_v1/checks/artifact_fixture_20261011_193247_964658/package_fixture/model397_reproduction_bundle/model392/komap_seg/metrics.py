"""Confusion-matrix based per-class IoU / mIoU."""

import torch

from komap_seg.palette import NUM_CLASSES


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """pred, target: any shape of class indices (e.g. (H, W) or (B, H, W))."""
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    idx = target * num_classes + pred
    conf = torch.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)
    return conf.float()


def iou_per_class(conf: torch.Tensor) -> torch.Tensor:
    """conf[i, j] = count of pixels with true class i predicted as class j."""
    intersection = torch.diag(conf)
    union = conf.sum(0) + conf.sum(1) - intersection
    return intersection / union.clamp(min=1)


def mean_iou(conf: torch.Tensor) -> float:
    return iou_per_class(conf).mean().item()
