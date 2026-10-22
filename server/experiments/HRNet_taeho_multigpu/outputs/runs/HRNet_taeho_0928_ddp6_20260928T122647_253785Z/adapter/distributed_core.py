"""Uneven batch shards with global-batch loss and synchronized BatchNorm."""

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_reduce
import torch.nn.functional as F

# Import runtime first to expose this experiment's frozen source package.
import runtime
from komap_seg.dataset import build_train_transform
from komap_seg.losses import gt_boundary_band
from komap_seg.model import forward_with_boundary


def batch_sizes(batch, world):
    if not 1 <= world <= batch:
        raise ValueError("GPU count must be between 1 and global batch size")
    return [batch // world + (rank < batch % world) for rank in range(world)]


def batch_slice(batch, world, rank):
    counts = batch_sizes(batch, world)
    offset = sum(counts[:rank])
    return slice(offset, offset + counts[rank])


def augmentation(variant):
    transform = build_train_transform()
    if variant == "no_gaussian":
        transform.transforms = [
            t
            for t in transform.transforms
            if type(t).__name__ not in {"GaussNoise", "GaussianBlur"}
        ]
    elif variant != "baseline":
        raise ValueError(variant)
    return transform


class TrainForward(torch.nn.Module):
    def __init__(self, network, boundary_head=True):
        super().__init__()
        self.network = network
        self.boundary_head = boundary_head

    def forward(self, x):
        if self.boundary_head:
            return forward_with_boundary(self.network, x)
        return self.network(x), None


def global_loss(logits, boundary, target, boundary_weight=1.0):
    """Same CE, batch Dice, boundary weighted BCE/Dice as a single batch.

    Every rank evaluates the same globally reduced scalar. Autograd all_reduce
    sums backward contributions; DDP's gradient average cancels world_size.
    This also preserves correct weighting for unequal rank batch sizes.
    """
    probs = logits.softmax(1)
    onehot = F.one_hot(target, 4).permute(0, 3, 1, 2).float()
    ce = F.cross_entropy(logits, target, reduction="sum")
    intersection = (probs * onehot).sum((0, 2, 3))
    cardinality = (probs + onehot).sum((0, 2, 3))
    stats = [ce[None], intersection, cardinality, logits.new_tensor([target.numel()])]
    if boundary is not None:
        b = boundary.float().squeeze(1)
        gt = gt_boundary_band(target)
        p = b.sigmoid()
        stats.append(
            torch.stack(
                [
                    (F.softplus(-b) * gt).sum(),
                    (F.softplus(b) * (1 - gt)).sum(),
                    (p * gt).sum(),
                    p.sum(),
                    gt.sum(),
                ]
            )
        )
    totals = torch.cat(stats)
    if dist.is_initialized() and dist.get_world_size() > 1:
        totals = all_reduce(totals)
    ce_sum, inter, card, count = totals[0], totals[1:5], totals[5:9], totals[9]
    loss = 0.5 * ce_sum / count + 0.5 * (
        1 - ((2 * inter + 1e-6) / (card + 1e-6)).mean()
    )
    if boundary is not None:
        positive, negative, b_inter, p_sum, gt_sum = totals[10:]
        pos = gt_sum.clamp(min=1.0)
        pos_weight = (count - pos) / pos
        bce = (pos_weight * positive + negative) / count
        dice = 1 - (2 * b_inter + 1) / (p_sum + gt_sum + 1)
        loss = loss + boundary_weight * (bce + dice)
    return loss


def broadcast_batch(iterator, cfg, rank, world, device):
    """Rank 0 uses the original global DataLoader, then each rank takes a slice.

    Broadcasting a batch retains the original sampler/augmentation stream
    without padding samples or changing the global optimizer batch size.
    """
    batch, size = cfg["batch_size"], cfg["patch_size"]
    if rank == 0:
        x, y = next(iterator)
        x = x.to(device, non_blocking=True).contiguous()
        y = y.to(device, non_blocking=True).contiguous()
    else:
        x = torch.empty(batch, 3, size, size, device=device)
        y = torch.empty(batch, size, size, device=device, dtype=torch.int64)
    dist.broadcast(x, src=0)
    dist.broadcast(y, src=0)
    selected = batch_slice(batch, world, rank)
    return x[selected].contiguous(), y[selected].contiguous()
