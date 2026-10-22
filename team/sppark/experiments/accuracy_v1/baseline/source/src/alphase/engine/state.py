from copy import deepcopy
import math
import random

import numpy as np
import torch


class EMA:
    def __init__(self, model, decay):
        self.module = deepcopy(model).eval().requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        for ema, raw in zip(self.module.parameters(), model.parameters()):
            ema.lerp_(raw.detach(), 1-self.decay)
        for ema, raw in zip(self.module.buffers(), model.buffers()):
            ema.copy_(raw)


def optimizer_for(model, cfg, stage):
    groups = {}
    depth = len(model.backbone.blocks) if hasattr(model, "backbone") else 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        backbone = name.startswith("backbone.")
        lr = stage.get("backbone_lr", 0) if backbone else stage["new_lr"]
        if backbone and stage.get("layer_lr_decay"):
            layer = int(name.split(".")[2])+1 if name.startswith("backbone.blocks.") else (depth+1 if "norm" in name else 0)
            lr *= stage["layer_lr_decay"] ** (depth+1-layer)
        wd = cfg["train"]["backbone_weight_decay" if backbone else "new_weight_decay"]
        if parameter.ndim <= 1 or name.endswith("bias"):
            wd = 0
        key = (lr, wd)
        if key not in groups:
            groups[key] = dict(params=[], lr=lr, weight_decay=wd, names=[])
        groups[key]["params"].append(parameter)
        groups[key]["names"].append(name)
    optimizer = torch.optim.AdamW(list(groups.values()))
    warmup, total = stage["warmup_updates"], stage["max_updates"]
    def factor(step):
        if warmup and step < warmup:
            return (step+1)/warmup
        progress = min(1., max(0., (step-warmup)/max(total-warmup, 1)))
        return .01 + .99*(1+math.cos(math.pi*progress))/2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    return optimizer, scheduler


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
