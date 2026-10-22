# R ensemble inference dependency.

# Unused training/search entrypoints are intentionally omitted.

from __future__ import annotations


from pathlib import Path

import numpy as np

import torch

from PIL import Image

from torch.cuda.amp import autocast

from advanced_inference import blend_window

from advanced_inference import inverse_view

from advanced_inference import transform_view

from advanced_inference import views

from improve_five_causes import NUM_CLASSES

from improve_five_causes import decode_mask

from improve_five_causes import encode_mask

from raw_zcdf_aspp import representation

from native_stem_finetune import Stride1NativeStemASPP

def resize_pair(image: np.ndarray, mask: np.ndarray, scale: float,
                minimum_side: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    factor = max(float(scale), float(minimum_side) / min(height, width))
    if abs(factor - 1.0) < 1e-6:
        return image, mask
    size = (int(round(width * factor)), int(round(height * factor)))
    image = np.asarray(Image.fromarray(image).resize(size, Image.Resampling.BICUBIC),
                       dtype=np.uint8)
    mask_rgb = encode_mask(mask).resize(size, Image.Resampling.NEAREST)
    return image, decode_mask(mask_rgb)

@torch.no_grad()
def predict_scaled(model, image_path: Path, patch_size: int, stride: int,
                   scale: float, device: torch.device, use_amp: bool,
                   tta: str) -> np.ndarray:
    source = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    scaled, _ = resize_pair(source, np.zeros(source.shape[:2], dtype=np.int64),
                            scale, patch_size)
    height, width = scaled.shape[:2]
    features = representation(scaled)
    ys = [0] if height <= patch_size else list(range(0, height - patch_size + 1, stride))
    xs = [0] if width <= patch_size else list(range(0, width - patch_size + 1, stride))
    last_y, last_x = max(height - patch_size, 0), max(width - patch_size, 0)
    if ys[-1] != last_y:
        ys.append(last_y)
    if xs[-1] != last_x:
        xs.append(last_x)
    padded_height, padded_width = max(height, ys[-1] + patch_size), max(width, xs[-1] + patch_size)
    padded = np.pad(features, ((0, 0), (0, padded_height - height),
                               (0, padded_width - width)), mode="reflect")
    probability_sum = np.zeros((NUM_CLASSES, padded_height, padded_width), dtype=np.float32)
    weight_sum = np.zeros((padded_height, padded_width), dtype=np.float32)
    window = blend_window(patch_size, "gaussian")
    locations = [(top, left) for top in ys for left in xs]
    transforms = [(0, False)] if tta == "none" else views(tta)
    for top, left in locations:
        inputs = torch.from_numpy(padded[:, top:top + patch_size, left:left + patch_size])
        inputs = inputs.unsqueeze(0).to(device, non_blocking=True)
        probabilities = None
        for rotation, flip in transforms:
            transformed = transform_view(inputs, rotation, flip)
            with autocast(enabled=use_amp):
                value = torch.softmax(model(transformed), dim=1)
            value = inverse_view(value, rotation, flip)
            probabilities = value if probabilities is None else probabilities + value
        value = (probabilities / len(transforms)).float().cpu().numpy()[0]
        probability_sum[:, top:top + patch_size, left:left + patch_size] += value * window[None]
        weight_sum[top:top + patch_size, left:left + patch_size] += window
    probability_sum = probability_sum[:, :height, :width] / np.maximum(
        weight_sum[None, :height, :width], 1e-7)
    original_size = (source.shape[1], source.shape[0])
    restored = np.stack([
        np.asarray(Image.fromarray(probability_sum[c]).resize(
            original_size, Image.Resampling.BILINEAR), dtype=np.float32)
        for c in range(NUM_CLASSES)
    ])
    return restored / np.maximum(restored.sum(0, keepdims=True), 1e-7)
