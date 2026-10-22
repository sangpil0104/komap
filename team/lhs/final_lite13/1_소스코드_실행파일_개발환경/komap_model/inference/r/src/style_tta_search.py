# R ensemble inference dependency.

# Unused training/search entrypoints are intentionally omitted.

from __future__ import annotations


import numpy as np

import torch

from PIL import Image

from advanced_inference import blend_window

from advanced_inference import inverse_view

from advanced_inference import transform_view

from advanced_inference import views

from improve_five_causes import normalize_image

from improve_five_causes import scaled_pair

from improve_five_causes import tile_starts

def transform_image(image, kind, train_mean, train_std):
    gray = image[..., 0].astype(np.float32)
    if kind == "standardize":
        value = (gray - gray.mean()) / max(gray.std(), 1e-5) * train_std + train_mean
    elif kind == "percentile":
        low, high = np.percentile(gray, (2.0, 98.0))
        value = (gray - low) / max(high - low, 1e-5) * 255.0
    else:
        raise ValueError(kind)
    value = np.clip(value, 0.0, 255.0).astype(np.uint8)
    return np.repeat(value[..., None], 3, axis=2)

@torch.no_grad()
def predict_array(model, image, patch_size, stride, device, use_amp, tile_batch_size, tta, blend):
    scaled, _ = scaled_pair(image, np.zeros(image.shape[:2], dtype=np.int64), patch_size)
    height, width = scaled.shape[:2]
    ys, xs = tile_starts(height, patch_size, stride), tile_starts(width, patch_size, stride)
    padded_height, padded_width = max(height, ys[-1] + patch_size), max(width, xs[-1] + patch_size)
    padded = np.pad(scaled, ((0, padded_height - height), (0, padded_width - width), (0, 0)), mode="reflect")
    probability_sum = np.zeros((4, padded_height, padded_width), dtype=np.float32)
    weight_sum = np.zeros((padded_height, padded_width), dtype=np.float32)
    locations = [(top, left) for top in ys for left in xs]
    transforms = views(tta)
    window = blend_window(patch_size, blend)
    for start in range(0, len(locations), tile_batch_size):
        batch_locations = locations[start:start + tile_batch_size]
        inputs = torch.stack([normalize_image(padded[top:top + patch_size, left:left + patch_size])
                              for top, left in batch_locations]).to(device, non_blocking=True)
        probability_batch = None
        for rotation, flip in transforms:
            view = transform_view(inputs, rotation, flip)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                value = torch.softmax(model(view), dim=1)
            value = inverse_view(value, rotation, flip)
            probability_batch = value if probability_batch is None else probability_batch + value
        probability_batch = (probability_batch / len(transforms)).float().cpu().numpy()
        for index, (top, left) in enumerate(batch_locations):
            probability_sum[:, top:top + patch_size, left:left + patch_size] += probability_batch[index] * window
            weight_sum[top:top + patch_size, left:left + patch_size] += window
    value = probability_sum / np.maximum(weight_sum[None], 1e-7)
    value = value[:, :height, :width]
    original_size = image.shape[1], image.shape[0]
    restored = np.stack([
        np.asarray(Image.fromarray(value[c]).resize(original_size, Image.Resampling.BILINEAR), dtype=np.float32)
        for c in range(4)
    ])
    return restored / np.maximum(restored.sum(axis=0, keepdims=True), 1e-7)
