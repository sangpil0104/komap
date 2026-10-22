import torch
import torch.nn.functional as F


def lovasz_present(probabilities, labels):
    losses = []
    for c in labels.unique():
        foreground = (labels == c).float()
        errors = (foreground - probabilities[c]).abs()
        errors, order = torch.sort(errors, descending=True)
        sorted_fg = foreground[order]
        total = sorted_fg.sum()
        intersection = total - sorted_fg.cumsum(0)
        union = total + (1-sorted_fg).cumsum(0)
        gradient = 1-intersection/union
        gradient = torch.cat([gradient[:1], gradient[1:]-gradient[:-1]])
        losses.append(torch.dot(errors, gradient))
    return torch.stack(losses).mean()


def boundary_target(target):
    valid = target != 255
    boundary = torch.zeros_like(valid)
    horizontal = valid[:, :, 1:] & valid[:, :, :-1] & (target[:, :, 1:] != target[:, :, :-1])
    vertical = valid[:, 1:] & valid[:, :-1] & (target[:, 1:] != target[:, :-1])
    boundary[:, :, 1:] |= horizontal
    boundary[:, :, :-1] |= horizontal
    boundary[:, 1:] |= vertical
    boundary[:, :-1] |= vertical
    return boundary.float(), valid


def segmentation_loss(outputs, target, weights):
    logits = outputs["logits"].float()
    ce, dice, lovasz = [], [], []
    for z, t in zip(logits, target):
        valid = t != 255
        if not valid.any():
            raise ValueError("No valid target pixels")
        t, z = t[valid], z[:, valid]
        ce.append(F.cross_entropy(z.T, t))
        p = z.softmax(0)
        present = t.unique()
        dice.append(torch.stack([1-(2*p[c, t == c].sum()+1e-6)/(p[c].sum()+(t == c).sum()+1e-6) for c in present]).mean())
        lovasz.append(lovasz_present(p, t) if weights["lovasz"] else z.sum()*0)
    pieces = dict(ce=torch.stack(ce).mean(), dice=torch.stack(dice).mean(), lovasz=torch.stack(lovasz).mean())
    loss = sum(weights[name] * value for name, value in pieces.items())
    if weights["auxiliary"]:
        valid = (target != 255).float().unsqueeze(1)
        one_hot = F.one_hot(target.clamp_max(3), 4).permute(0, 3, 1, 2).float() * valid
        shape = outputs["aux_logits"].shape[-2:]
        fraction = F.adaptive_avg_pool2d(valid, shape)
        soft = F.adaptive_avg_pool2d(one_hot, shape) / fraction.clamp_min(1e-8)
        ce_map = -(soft * outputs["aux_logits"].float().log_softmax(1)).sum(1, keepdim=True)
        aux = ((ce_map * fraction).sum((1, 2, 3)) / fraction.sum((1, 2, 3)).clamp_min(1)).mean()
        loss = loss + weights["auxiliary"] * aux
    if weights["boundary"]:
        boundary, valid = boundary_target(target)
        z = outputs["boundary_logits"][:, 0].float()
        terms = []
        for zi, bi, vi in zip(z, boundary, valid):
            prob, truth = zi[vi].sigmoid(), bi[vi]
            bce = F.binary_cross_entropy_with_logits(zi[vi], truth)
            dice_b = 1-(2*(prob*truth).sum()+1e-6)/(prob.sum()+truth.sum()+1e-6)
            terms.append((bce+dice_b)/2)
        loss = loss + weights["boundary"] * torch.stack(terms).mean()
    return loss, {name: value.detach() for name, value in pieces.items()}
