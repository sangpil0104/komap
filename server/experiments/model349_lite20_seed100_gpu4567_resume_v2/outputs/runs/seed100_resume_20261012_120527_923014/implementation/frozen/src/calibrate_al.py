# Model349 inference dependency; extracted from the recorded source snapshot.

# Unused training/search entrypoints are intentionally omitted.

from __future__ import annotations


from pathlib import Path

import numpy as np

import torch

from PIL import Image

from torch.cuda.amp import autocast

from improve_five_causes import NUM_CLASSES

from improve_five_causes import normalize_image

from improve_five_causes import scaled_pair

from improve_five_causes import tile_starts

from train_models_2_to_6 import UNetPlusPlusResNet34

def load_model(checkpoint: str, device: torch.device) -> torch.nn.Module:
    model = UNetPlusPlusResNet34(pretrained=False).to(device)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"])
    model.eval()
    return model

@torch.no_grad()
def predict_probabilities(
    model: torch.nn.Module,
    image_path: Path,
    patch_size: int,
    stride: int,
    device: torch.device,
    use_amp: bool,
    tile_batch_size: int,
    tta: bool,
) -> np.ndarray:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    scaled, _ = scaled_pair(image, np.zeros(image.shape[:2], dtype=np.int64), patch_size)
    height, width = scaled.shape[:2]
    ys, xs = tile_starts(height, patch_size, stride), tile_starts(width, patch_size, stride)
    padded_height, padded_width = max(height, ys[-1] + patch_size), max(width, xs[-1] + patch_size)
    padded = np.pad(scaled, ((0, padded_height - height), (0, padded_width - width), (0, 0)), mode="reflect")
    probability_sum = np.zeros((NUM_CLASSES, padded_height, padded_width), dtype=np.float32)
    weight_sum = np.zeros((padded_height, padded_width), dtype=np.float32)
    locations = [(top, left) for top in ys for left in xs]
    for start in range(0, len(locations), tile_batch_size):
        batch_locations = locations[start:start + tile_batch_size]
        inputs = torch.stack([
            normalize_image(padded[top:top + patch_size, left:left + patch_size])
            for top, left in batch_locations
        ]).to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            probabilities = torch.softmax(model(inputs), dim=1)
            if tta:
                views = [torch.flip(inputs, dims=(-1,)), torch.flip(inputs, dims=(-2,)),
                         torch.rot90(inputs, 1, dims=(-2, -1))]
                undo = [lambda x: torch.flip(x, dims=(-1,)), lambda x: torch.flip(x, dims=(-2,)),
                        lambda x: torch.rot90(x, -1, dims=(-2, -1))]
                for view, undo_fn in zip(views, undo):
                    probabilities = probabilities + undo_fn(torch.softmax(model(view), dim=1))
                probabilities = probabilities / 4.0
        probabilities = probabilities.float().cpu().numpy()
        for index, (top, left) in enumerate(batch_locations):
            probability_sum[:, top:top + patch_size, left:left + patch_size] += probabilities[index]
            weight_sum[top:top + patch_size, left:left + patch_size] += 1.0
    probabilities = probability_sum / np.maximum(weight_sum[None], 1e-6)
    probabilities = probabilities[:, :height, :width]
    # Resize each probability map back to the original image size before the
    # class decision. This avoids a nearest-neighbor label resize changing the
    # effect of the Al calibration at the original boundary.
    original_size = image.shape[1], image.shape[0]
    restored = np.stack([
        np.asarray(Image.fromarray(probabilities[class_id]).resize(original_size, Image.Resampling.BILINEAR),
                   dtype=np.float32)
        for class_id in range(NUM_CLASSES)
    ])
    return restored / np.maximum(restored.sum(axis=0, keepdims=True), 1e-7)
