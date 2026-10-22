"""Whole-image inference, preserving the archived B4 preprocessing and D4 order."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .data import normalize
from .models import restore_probs


def d4(x, k):
    if k >= 4:
        x = x.flip(-1)
    return torch.rot90(x, k % 4, (-2, -1))


def inverse_d4(x, k):
    x = torch.rot90(x, -(k % 4), (-2, -1))
    return x.flip(-1) if k >= 4 else x


@torch.inference_mode()
def predict(model, gray, cfg, device, callback=None):
    h0, w0 = gray.shape
    scale = cfg['target_width'] / w0
    g = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) if scale != 1 else gray
    h, w = g.shape
    a = normalize(g)
    x = torch.from_numpy(np.stack([a,a,a]))[None].to(device)
    x = F.pad(x, (0, (-w)%32, 0, (-h)%32), mode='reflect')
    orientations = list(range(8)) if cfg['tta'] else [0]
    total = torch.zeros((1,4,*x.shape[-2:]), device=device, dtype=torch.float32)
    for k in orientations:
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.device(device).type == 'cuda'):
            logits = model.detail_logits(d4(x, k))
        total.add_(inverse_d4(logits.float().softmax(1), k))
        if callback:
            callback(k+1, len(orientations))
    total = total[:, :, :h, :w] / len(orientations)
    total = restore_probs(total, (h0,w0), cfg['restore'])
    if model.native_refiner is not None:
        native_gray = torch.from_numpy(normalize(gray))[None,None].to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.device(device).type == 'cuda'):
            total = model.refine(native_gray, total)
    return total[0].float()
