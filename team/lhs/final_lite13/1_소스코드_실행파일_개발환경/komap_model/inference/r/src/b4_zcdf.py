# R ensemble inference dependency.

# Unused training/search entrypoints are intentionally omitted.

from __future__ import annotations


import os

import numpy as np

import torch

import torch.nn.functional as F

from PIL import Image

PALETTE = np.array([
    [153, 127, 76],   # 0 brown : bright matrix
    [25, 76, 153],    # 1 blue  : mid-gray elongated phase
    [76, 178, 76],    # 2 green : dark polygonal particles
    [204, 204, 204],  # 3 gray  : fine dark needles
], dtype=np.uint8)

NUM_CLASSES = len(PALETTE)

IMAGENET_MEAN, IMAGENET_STD = 0.449, 0.226  # gray equivalent

def load_gray(path):
    return np.array(Image.open(path).convert("L"))

def normalize(gray, mode, image_stats=None):
    """uint8 HxW -> float32 HxW."""
    x = gray.astype(np.float32) / 255.0
    if mode == "imagenet":
        return (x - IMAGENET_MEAN) / IMAGENET_STD
    if mode == "image":  # per-image standardization (absorbs exposure differences)
        return (x - x.mean()) / (x.std() + 1e-6)
    if mode == "image_global":  # train crop uses statistics of its full resized source image
        mean, std = image_stats if image_stats is not None else (x.mean(), x.std())
        return (x - mean) / (std + 1e-6)
    raise ValueError(mode)

def morph_features(gray):
    """Hand-crafted maps that encode how each phase's crystallography shows up in OM.

    ridge   : thin dark line-like structures (eutectic Si needles), max over scales of the
              scale-normalized largest Hessian eigenvalue.
    highpass: local contrast relative to a smooth background (faceted primary Si / Al3Ni edges).
    Scales are in pixels of the image the network sees.
    """
    import cv2
    g0 = gray.astype(np.float32) / 255.0
    ridge = np.zeros_like(g0)
    for s in (1.5, 3.0):
        g = cv2.GaussianBlur(g0, (0, 0), s)
        dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
        dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
        dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
        lmax = 0.5 * (dxx + dyy + np.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2))
        ridge = np.maximum(ridge, s * s * np.clip(lmax, 0, None))
    highpass = g0 - cv2.GaussianBlur(g0, (0, 0), 8)
    return ridge, highpass

def make_input(gray, norm, input_mode="gray3", stats=None, image_stats=None, cdf_lut=None):
    """uint8 HxW -> float32 3xHxW network input."""
    if input_mode == "zcdf3":
        gray01 = gray.astype(np.float32) / 255.0
        mean, std = image_stats if image_stats is not None else (
            float(gray01.mean()), float(gray01.std()))
        if cdf_lut is None:
            hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
            cdf_lut = (np.cumsum(hist) / max(float(hist.sum()), 1.0)).astype(np.float32)
        zscore = np.clip((gray01 - mean) / (std + 1e-6), -4.0, 4.0) / 3.0
        cdf = (cdf_lut[gray] - 0.5) / 0.288675
        return np.stack([(gray01 - 0.5) / 0.25, zscore, cdf]).astype(np.float32)

    x = normalize(gray, norm, image_stats)
    if input_mode == "gray3":
        return np.stack([x, x, x])
    if input_mode == "feat3":
        ridge, hp = morph_features(gray)
        return np.stack([x, (ridge - stats[0]) / stats[1], (hp - stats[2]) / stats[3]]).astype(np.float32)
    raise ValueError(input_mode)

def build_model(arch, encoder, pretrained=True):
    import segmentation_models_pytorch as smp
    cls = {"unet": smp.Unet, "unetpp": smp.UnetPlusPlus, "fpn": smp.FPN,
           "deeplabv3p": smp.DeepLabV3Plus, "upernet": smp.UPerNet, "segformer": smp.Segformer}[arch]
    return cls(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
               in_channels=3, classes=NUM_CLASSES)

def _d4(x, k):  # x: B,C,H,W ; k in 0..7
    if k >= 4:
        x = x.flip(-1)
    return torch.rot90(x, k % 4, dims=(-2, -1))

def _d4_inv(x, k):
    x = torch.rot90(x, -(k % 4), dims=(-2, -1))
    if k >= 4:
        x = x.flip(-1)
    return x

def resize_gray(gray, scale):
    if scale == 1:
        return gray
    import cv2
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

@torch.no_grad()
def predict_probs(members, gray, tta=True, device="cuda"):
    """Full-image softmax probs (C,H,W) at original resolution, averaged over members and D4 TTA.

    members: list of dicts from load_member(). Each sees the image resized by its own factor:
    a fixed `scale`, or `target_width / W` when the member normalizes magnification per image.
    """
    H, W = gray.shape
    ks = range(8) if tta else [0]
    acc = 0
    for mem in members:
        m, norm, input_mode, stats = mem["model"], mem["norm"], mem["input"], mem["stats"]
        scale = member_scale(mem, W)
        g = resize_gray(gray, scale)
        h, w = g.shape
        x = torch.from_numpy(make_input(g, norm, input_mode, stats))[None].to(device)
        x = F.pad(x, (0, (-w) % 32, 0, (-h) % 32), mode="reflect")
        part = 0
        for k in ks:
            # FP16 autocast: BF16 is not available on older GPUs; FP16 is broadly supported.
            with torch.autocast("cuda", dtype=torch.float16, enabled=torch.device(device).type == "cuda"):
                logits = m(_d4(x, k))
            part = part + _d4_inv(logits.float().softmax(1), k)
        part = part[:, :, :h, :w] / len(ks)
        if scale != 1:
            part = F.interpolate(part, size=(H, W), mode="area" if scale > 1 else "bilinear")
        acc = acc + part
    return (acc / len(members))[0]

def member_scale(mem, width):
    """Resize factor for an image of this width: per-image magnification normalization or fixed."""
    tw = mem.get("target_width") or 0
    return (tw / width if tw else mem["scale"]) * mem.get("scale_mult", 1.0)

def make_member(model, norm, scale=1.0, input_mode="gray3", stats=None, target_width=0, scale_mult=1.0):
    return {"model": model, "norm": norm, "scale": scale, "input": input_mode, "stats": stats,
            "target_width": target_width, "scale_mult": scale_mult}

def load_member(run_dir, device="cuda", weights="last.pt"):
    import json
    a = json.load(open(os.path.join(run_dir, "args.json")))
    m = build_model(a["arch"], a["encoder"], pretrained=False)
    m.load_state_dict(torch.load(os.path.join(run_dir, weights), map_location="cpu"))
    return make_member(m.to(device).eval(), a["norm"], a.get("scale", 1.0), a.get("input", "gray3"),
                       a.get("feat_stats"), a.get("target_width", 0))
