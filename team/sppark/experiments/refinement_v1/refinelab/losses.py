import torch
from torch.nn import functional as F

from alphase.losses.segmentation import segmentation_loss as original_loss


def segmentation_loss(outputs, target, weights):
    class_weights = weights.get("class_weights")
    if class_weights is None:
        return original_loss(outputs, target, weights)
    w = outputs["logits"].new_tensor(class_weights, dtype=torch.float32)
    if w.shape != (4,) or not torch.isfinite(w).all() or (w <= 0).any():
        raise ValueError("Four finite positive CE class weights are required")
    loss, pieces = original_loss(outputs, target, {**weights, "ce": 0.})
    ce = []
    for z, t in zip(outputs["logits"].float(), target):
        valid = t != 255
        if not valid.any():
            raise ValueError("No valid target pixels")
        ce.append(F.cross_entropy(z[:, valid].T, t[valid], weight=w))
    weighted = torch.stack(ce).mean()
    return loss + weights["ce"] * weighted, {**pieces, "ce": weighted.detach()}
