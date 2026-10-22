# Model349 inference dependency; extracted from the recorded source snapshot.

# Unused training/search entrypoints are intentionally omitted.

from __future__ import annotations


import numpy as np

import torch

from PIL import Image

from improve_five_causes import NUM_CLASSES

from improve_five_causes import normalize_image

from improve_five_causes import scaled_pair

from improve_five_causes import tile_starts

def transform_view(x: torch.Tensor, rotation: int, flip: bool) -> torch.Tensor:
    if flip:
        x = torch.flip(x, dims=(-1,))
    return torch.rot90(x, rotation, dims=(-2, -1))

def inverse_view(x: torch.Tensor, rotation: int, flip: bool) -> torch.Tensor:
    x = torch.rot90(x, -rotation, dims=(-2, -1))
    if flip:
        x = torch.flip(x, dims=(-1,))
    return x

def views(mode: str):
    if mode == "legacy":
        return [(0, False), (0, True), (2, True), (1, False)]
    if mode == "d4":
        return [(rotation, flip) for flip in (False, True) for rotation in range(4)]
    raise ValueError(mode)

def blend_window(patch_size: int, mode: str) -> np.ndarray:
    if mode == "uniform":
        return np.ones((patch_size, patch_size), dtype=np.float32)
    axis = np.linspace(-1.0, 1.0, patch_size, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    window = np.exp(-(xx * xx + yy * yy) / (2.0 * 0.55 * 0.55)).astype(np.float32)
    return 0.15 + 0.85 * (window / window.max())

@torch.no_grad()
def predict_probabilities(model, image_path, patch_size, stride, device, use_amp, tile_batch_size,
                          tta_mode, blend_mode, scale=1.0) -> np.ndarray:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    original_size = image.shape[1], image.shape[0]
    if abs(float(scale) - 1.0) > 1e-6:
        if float(scale) <= 0:
            raise ValueError("scale must be positive")
        new_size = (int(round(image.shape[1] * float(scale))),
                    int(round(image.shape[0] * float(scale))))
        image = np.asarray(Image.fromarray(image).resize(
            new_size, Image.Resampling.BICUBIC), dtype=np.uint8)
    scaled, _ = scaled_pair(image, np.zeros(image.shape[:2], dtype=np.int64), patch_size)
    height, width = scaled.shape[:2]
    ys, xs = tile_starts(height, patch_size, stride), tile_starts(width, patch_size, stride)
    padded_height, padded_width = max(height, ys[-1] + patch_size), max(width, xs[-1] + patch_size)
    padded = np.pad(scaled, ((0, padded_height - height), (0, padded_width - width), (0, 0)), mode="reflect")
    probability_sum = np.zeros((NUM_CLASSES, padded_height, padded_width), dtype=np.float32)
    weight_sum = np.zeros((padded_height, padded_width), dtype=np.float32)
    window = blend_window(patch_size, blend_mode)
    locations = [(top, left) for top in ys for left in xs]
    transforms = views(tta_mode)
    for start in range(0, len(locations), tile_batch_size):
        batch_locations = locations[start:start + tile_batch_size]
        inputs = torch.stack([
            normalize_image(padded[top:top + patch_size, left:left + patch_size])
            for top, left in batch_locations
        ]).to(device, non_blocking=True)
        probability_batch = None
        for rotation, flip in transforms:
            view = transform_view(inputs, rotation, flip)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                probabilities = torch.softmax(model(view), dim=1)
            probabilities = inverse_view(probabilities, rotation, flip)
            probability_batch = probabilities if probability_batch is None else probability_batch + probabilities
        probability_batch = (probability_batch / float(len(transforms))).float().cpu().numpy()
        for index, (top, left) in enumerate(batch_locations):
            probability_sum[:, top:top + patch_size, left:left + patch_size] += probability_batch[index] * window[None]
            weight_sum[top:top + patch_size, left:left + patch_size] += window
    probabilities = probability_sum / np.maximum(weight_sum[None], 1e-6)
    probabilities = probabilities[:, :height, :width]
    restored = np.stack([
        np.asarray(Image.fromarray(probabilities[class_id]).resize(original_size, Image.Resampling.BILINEAR), dtype=np.float32)
        for class_id in range(NUM_CLASSES)
    ])
    return restored / np.maximum(restored.sum(axis=0, keepdims=True), 1e-7)
