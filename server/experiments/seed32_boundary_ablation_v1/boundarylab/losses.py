"""Global-batch CE/Dice with differentiable distributed reductions."""
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as distributed_nn
import torch.nn.functional as F


def global_sum(value):
    return distributed_nn.all_reduce(value, op=dist.ReduceOp.SUM) if dist.is_initialized() else value


def ce_dice(logits, target, band=None, extra=0.0):
    valid = target != 255
    safe = target.masked_fill(~valid, 0)
    p = logits.float().softmax(1) * valid[:, None]
    oh = F.one_hot(safe, 4).permute(0,3,1,2).float() * valid[:, None]
    weights = valid.float() if band is None else valid.float() * (1 + extra * band)
    ce = F.cross_entropy(logits.float(), target, ignore_index=255, reduction='none')
    numerator = (ce * weights).sum()
    denominator = weights.sum()
    inter = (p * oh).sum((0,2,3))
    denom = p.sum((0,2,3)) + oh.sum((0,2,3))
    combined = global_sum(torch.cat([numerator[None], denominator[None], inter, denom]))
    ce = combined[0] / combined[1].clamp_min(1)
    dice = 1 - ((2*combined[2:6]+1)/(combined[6:10]+1)).mean()
    return ce + dice


def native_loss(probability, target):
    valid = target != 255
    oh = F.one_hot(target.masked_fill(~valid, 0), 4).permute(0,3,1,2).float() * valid[:, None]
    p = probability * valid[:, None]
    nll = F.nll_loss(probability.clamp_min(1e-7).log(), target, ignore_index=255)
    inter = (p*oh).sum((0,2,3))
    denom = p.sum((0,2,3)) + oh.sum((0,2,3))
    return nll + 1 - ((2*inter+1)/(denom+1)).mean()


def loss_for(output, sample, cfg):
    base = ce_dice(output['logits'], sample['target'], sample['boundary'], cfg['boundary_ce_extra'])
    native = base.new_zeros(())
    if cfg['native_weight']:
        if output['native'] is None or sample['native_target'] is None:
            raise ValueError('Missing aligned native target')
        native = native_loss(output['native'], sample['native_target'])
    return base + cfg['native_weight']*native, dict(detail=base.detach(), native=native.detach())
