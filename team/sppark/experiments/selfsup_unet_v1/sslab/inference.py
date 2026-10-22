from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from .data import crop


def autocast(device, precision):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def tile_starts(length, core, stride, offset=0):
    if not 0 < stride <= core:
        raise ValueError("Invalid tile stride")
    last = max(0, length - core)
    starts = {0, last}
    starts.update(range(offset % stride, last + 1, stride))
    return sorted(starts)


def d4(array, index):
    out = np.rot90(array, index % 4, axes=(-2, -1))
    return np.flip(out, axis=-1) if index >= 4 else out


def undo_d4(array, index):
    if index >= 4:
        array = np.flip(array, axis=-1)
    return np.rot90(array, -(index % 4), axes=(-2, -1)).copy()


def forward_probability(model, g, scale, device, precision):
    # Interpolate image intensities, never discrete predictions.
    image = torch.from_numpy(np.ascontiguousarray(g))[None, None]
    h, w = g.shape
    image = F.interpolate(image, size=(h * scale, w * scale), mode="bilinear", align_corners=False)
    ph, pw = -image.shape[-2] % 32, -image.shape[-1] % 32
    if ph or pw:
        # np.pad also works for very small images where torch reflect is restricted.
        arr = np.pad(image.numpy(), ((0, 0), (0, 0), (0, ph), (0, pw)), mode="reflect")
        image = torch.from_numpy(arr)
    image = image.expand(-1, 3, -1, -1).contiguous().to(device)
    with autocast(device, precision):
        logits = model(image)
    probability = logits[0, :, :h * scale, :w * scale].float().softmax(0).cpu()
    return probability


@torch.inference_mode()
def predict_view(model, g, cfg, device, show_tiles=False):
    settings, scale = cfg["inference"], cfg["scale"]
    precision = cfg["train"]["precision"]
    h, w = g.shape
    if settings["mode"] == "full":
        pad = settings["pad"]
        extended = crop(g, -pad, -pad, w + 2 * pad, h + 2 * pad)
        p = forward_probability(model, extended, scale, device, precision)
        p = p[:, pad * scale:(pad + h) * scale, pad * scale:(pad + w) * scale]
        return F.interpolate(p[None], size=(h, w), mode="bilinear", align_corners=False)[0].numpy()
    if settings["mode"] != "tiled":
        raise ValueError("Inference mode must be full or tiled")
    core, halo = settings["core"], settings["halo"]
    stride = round(core * (1 - settings["overlap"]))
    xs = tile_starts(w, core, stride, settings["offset"])
    ys = tile_starts(h, core, stride, settings["offset"])
    axis = np.linspace(-1, 1, core, dtype=np.float32)
    weight = np.exp(-2 * (axis[:, None] ** 2 + axis[None, :] ** 2)).clip(.001)
    result, count = np.zeros((4, h, w), np.float32), np.zeros((h, w), np.float32)
    positions = [(x, y) for y in ys for x in xs]
    for x, y in tqdm(positions, desc="Tiles", disable=not show_tiles, leave=False, unit="tile"):
        patch = crop(g, x - halo, y - halo, core + 2 * halo, core + 2 * halo)
        p = forward_probability(model, patch, scale, device, precision)
        p = p[:, halo * scale:(halo + core) * scale, halo * scale:(halo + core) * scale]
        p = F.interpolate(p[None], size=(core, core), mode="bilinear", align_corners=False)[0].numpy()
        hh, ww = min(core, h - y), min(core, w - x)
        result[:, y:y + hh, x:x + ww] += p[:, :hh, :ww] * weight[:hh, :ww]
        count[y:y + hh, x:x + ww] += weight[:hh, :ww]
    if not (count > 0).all():
        raise ValueError("Uncovered output pixels")
    return result / count[None]


@torch.inference_mode()
def predict(model, g, cfg, device, show_tiles=False):
    model.eval()
    total = np.zeros((4, *g.shape), np.float32)
    views = 8 if cfg["inference"]["d4"] else 1
    for k in range(views):
        p = predict_view(model, d4(g, k), cfg, device, show_tiles)
        total += undo_d4(p, k)
    total /= views
    if not np.isfinite(total).all():
        raise ValueError("Non-finite probabilities")
    return total / total.sum(0, keepdims=True).clip(1e-8)
