import torch
from torch.nn import functional as F


def reconstruction_loss(logits, target, hidden_valid):
    prediction = logits.float().sigmoid()
    count = hidden_valid.sum()
    if count <= 0:
        raise ValueError("No real masked pixels")
    return ((prediction - target).abs() * hidden_valid).sum() / count


def segmentation_loss(logits, target):
    terms = []
    for prediction, truth in zip(logits.float(), target):
        valid = truth != 255
        if not valid.any():
            raise ValueError("Empty supervised crop")
        prediction, truth = prediction[:, valid], truth[valid]
        ce = F.cross_entropy(prediction.T, truth)
        probs = prediction.softmax(0)
        dice = []
        for c in truth.unique():
            mask = (truth == c).float()
            dice.append(1 - (2 * (probs[c] * mask).sum() + 1e-6) / (probs[c].sum() + mask.sum() + 1e-6))
        terms.append(ce + torch.stack(dice).mean())
    return torch.stack(terms).mean()
