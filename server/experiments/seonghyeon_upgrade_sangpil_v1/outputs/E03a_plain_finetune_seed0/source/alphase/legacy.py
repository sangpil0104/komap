"""Independent implementation of the frozen, audited legacy inference contract."""

import types
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from .io import local_path


def original_module(cfg):
    path = local_path(cfg["paths"]["legacy_source"]) / "vendor/common.py"
    # Execute read-only source without writing __pycache__ into the old run.
    module = types.ModuleType("frozen_legacy_common")
    module.__file__ = str(path)
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module


def d4(x, k):
    if k >= 4:
        x = x.flip(-1)
    return torch.rot90(x, k % 4, (-2, -1))


def inverse(x, k):
    x = torch.rot90(x, -(k % 4), (-2, -1))
    return x.flip(-1) if k >= 4 else x


@torch.inference_mode()
def native_probs(model, gray, cfg, device):
    h0, w0 = gray.shape
    scale = cfg["geometry"]["target_width"] / w0
    g = (
        cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        if scale != 1
        else gray
    )
    h, w = g.shape
    a = (g.astype(np.float32) / 255.0 - 0.449) / 0.226
    x = torch.from_numpy(np.stack([a, a, a]))[None].to(device)
    x = F.pad(x, (0, (-w) % 32, 0, (-h) % 32), mode="reflect")
    total = 0
    ks = range(8) if cfg["inference"]["d4"] else [0]
    for k in ks:
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"
        ):
            logits = model.forward_logits(d4(x, k))
        total = total + inverse(logits.float().softmax(1), k)
    total = total[:, :, :h, :w] / len(ks)
    if scale != 1:
        total = F.interpolate(
            total, size=(h0, w0), mode="area" if scale > 1 else "bilinear"
        )
    return total[0]
