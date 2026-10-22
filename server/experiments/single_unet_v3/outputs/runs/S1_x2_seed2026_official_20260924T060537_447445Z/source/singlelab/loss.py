import torch
from torch.nn import functional as F


def segmentation_loss(logits, target, edge, cfg):
    """Equal image weighting; valid central pixels only; S3 changes CE weights only."""
    logits = logits.float()
    losses, ces, dices = [], [], []
    for z, truth, band in zip(logits, target, edge):
        valid = truth != 255
        if not valid.any():
            raise ValueError("Empty supervised crop")
        z, truth = z[:, valid], truth[valid]
        weights = 1 + cfg["boundary_weight"] * band[valid]
        ce = (F.cross_entropy(z.T, truth, reduction="none") * weights).sum() / weights.sum()
        probs = z.softmax(0)
        terms = []
        for c in truth.unique():
            mask = (truth == c).float()
            terms.append(1 - (2 * (probs[c] * mask).sum() + 1e-6) /
                         (probs[c].sum() + mask.sum() + 1e-6))
        dice = torch.stack(terms).mean()
        losses.append(cfg["ce"] * ce + cfg["dice"] * dice)
        ces.append(ce.detach())
        dices.append(dice.detach())
    return torch.stack(losses).mean(), {"ce": torch.stack(ces).mean(), "dice": torch.stack(dices).mean()}
