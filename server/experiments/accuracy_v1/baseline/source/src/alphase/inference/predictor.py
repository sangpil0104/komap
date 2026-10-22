from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from alphase.data.dataset import build_inputs
from alphase.data.geometry import crop_array, d4, tile_starts, undo_d4
from alphase.utils import amp_context


def normalize(probability):
    if not torch.isfinite(probability).all():
        raise ValueError("Non-finite prediction probabilities")
    return probability / probability.sum(0, keepdim=True).clamp_min(1e-12)


@torch.inference_mode()
def predict_view(model, gray, cfg, device):
    hybrid = cfg["model"]["name"] == "alphase_hybrid_v1"
    detail, dino = build_inputs(gray, cfg["data"]["image_channels"] == ["gray"], hybrid)
    h, w = gray.shape
    settings = cfg["inference"]
    precision = cfg["train"]["precision"]
    dino = dino.unsqueeze(0).to(device) if hybrid else None
    with amp_context(device, precision):
        if settings["mode"] == "full_frame":
            logits = model(detail.unsqueeze(0).to(device), dino)["logits"][0, :, :h, :w]
            return logits.float().softmax(0).cpu()
        if settings["mode"] != "tiled":
            raise ValueError("inference.mode must be full_frame or tiled")
        core, halo = settings["local_core_size"], settings["halo"]
        if core % 32 or (core+2*halo) % 32:
            raise ValueError("Tile core and halo input must be multiples of 32")
        stride = round(core * (1-settings["overlap_fraction"]))
        context = model.encode_context(dino) if hybrid else None
        result, count = torch.zeros(4, h, w), torch.zeros(h, w)
        axis = torch.linspace(-1, 1, core)
        weight = torch.exp(-2*(axis[:, None]**2+axis[None, :]**2)).clamp_min(.001)
        for y in tile_starts(h, core, stride):
            for x in tile_starts(w, core, stride):
                box = (x-halo, y-halo, x+core+halo, y+core+halo)
                patch = torch.from_numpy(crop_array(detail.numpy()[..., :h, :w], box)).unsqueeze(0).to(device)
                if hybrid:
                    output = model.decode_local(patch, context, torch.tensor([box], dtype=torch.float32, device=device), dino.shape[-2:])
                else:
                    output = model(patch)
                prob = output["logits"][0, :, halo:halo+core, halo:halo+core].float().softmax(0).cpu()
                hh, ww = min(core, h-y), min(core, w-x)
                result[:, y:y+hh, x:x+ww] += prob[:, :hh, :ww] * weight[:hh, :ww]
                count[y:y+hh, x:x+ww] += weight[:hh, :ww]
        if not (count > 0).all():
            raise ValueError("Uncovered image pixels")
        return normalize(result / count)


@torch.inference_mode()
def predict_probabilities(model, gray, cfg, device):
    model.eval()
    h, w = gray.shape
    original = torch.from_numpy(np.asarray(gray, dtype=np.float32))[None, None]
    result = torch.zeros(4, h, w)
    n = 0
    for scale in cfg["inference"]["scales"]:
        if scale <= 0:
            raise ValueError("Inference scale must be positive")
        resized = F.interpolate(original, size=(max(1, round(h*scale)), max(1, round(w*scale))), mode="bilinear", align_corners=False, antialias=True)
        for index in range(8 if cfg["inference"]["d4"] else 1):
            transformed = d4(resized, index)[0, 0].numpy()
            probability = predict_view(model, transformed, cfg, device)
            probability = undo_d4(probability, index)
            probability = F.interpolate(probability[None], size=(h, w), mode="bilinear", align_corners=False)[0]
            result += normalize(probability)
            n += 1
    result = normalize(result/n)
    bias = torch.tensor(cfg["inference"].get("class_logit_bias", [0]*4))[:, None, None]
    if bias.any():
        result = (result.clamp_min(1e-7).log()+bias).softmax(0)
    return result.numpy()
