"""Historical CE+Dice, plus differentiable native NLL/Lovasz ablation.
Lovasz: installed SMP0.5.0 adaptation of Maxim Berman's MIT implementation.
https://github.com/bermanmaxim/LovaszSoftmax (source attribution preserved in SMP).
"""

import torch
import torch.nn.functional as F
from segmentation_models_pytorch.losses.lovasz import _lovasz_softmax
from .contracts import restore_native_probs


def legacy_loss(logits, target):
    valid = target != 255
    if not valid.any():
        raise ValueError("All-ignore training batch")
    p = logits.float().softmax(1)
    oh = F.one_hot(target.masked_fill(~valid, 0), 4).permute(0, 3, 1, 2).float()
    p = p * valid[:, None]
    oh = oh * valid[:, None]
    inter = (p * oh).sum((0, 2, 3))
    denom = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return F.cross_entropy(logits.float(), target, ignore_index=255) + (
        1 - ((2 * inter + 1) / (denom + 1)).mean()
    )


def losses(output, batch, cfg):
    logits = output["logits_detail"]
    base = legacy_loss(logits, batch["target_detail"])
    result = {"base": base, "native_nll": None, "native_lovasz": None, "total": base}
    if not cfg["loss"]["native_enabled"]:
        return result
    nll = []
    lovasz = []
    for i, (target, geometry) in enumerate(
        zip(batch["target_native"], batch["geometry"])
    ):
        prob = restore_native_probs(
            logits[i : i + 1].float().softmax(1),
            tuple(target.shape),
            geometry["d4_k"],
            geometry["d4_flip"],
        )
        if not (target != 255).any():
            raise ValueError("All-ignore native ROI")
        nll.append(
            F.nll_loss(
                prob.clamp_min(cfg["loss"]["probability_eps"]).log(),
                target[None],
                ignore_index=255,
            )
        )
        lovasz.append(
            _lovasz_softmax(
                prob, target[None], classes="present", per_image=True, ignore=255
            )
        )
    result["native_nll"] = torch.stack(nll).mean()
    result["native_lovasz"] = torch.stack(lovasz).mean()
    result["total"] = base + cfg["loss"]["native_weight"] * (
        result["native_nll"]
        + cfg["loss"]["native_lovasz_weight"] * result["native_lovasz"]
    )
    return result
