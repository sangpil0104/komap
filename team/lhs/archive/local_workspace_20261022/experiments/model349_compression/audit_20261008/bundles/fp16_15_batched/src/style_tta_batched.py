"""Frozen high-resolution D4 prediction: four views per forward, original sum order."""
import numpy as np
import torch
from PIL import Image
from advanced_inference import blend_window, inverse_view, transform_view, views
from improve_five_causes import normalize_image, scaled_pair, tile_starts

@torch.no_grad()
def predict_array(model, image, patch_size, stride, device, use_amp, tile_batch_size, tta, blend):
    scaled, _ = scaled_pair(image, np.zeros(image.shape[:2], dtype=np.int64), patch_size)
    height, width = scaled.shape[:2]
    ys, xs = tile_starts(height, patch_size, stride), tile_starts(width, patch_size, stride)
    padded_height, padded_width = max(height, ys[-1] + patch_size), max(width, xs[-1] + patch_size)
    padded = np.pad(scaled, ((0, padded_height - height), (0, padded_width - width), (0, 0)), mode='reflect')
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
        n = len(batch_locations)
        for offset in range(0, len(transforms), 4):
            group = transforms[offset:offset + 4]
            joined = torch.cat([transform_view(inputs, rotation, flip) for rotation, flip in group], dim=0)
            with torch.autocast(device_type='cuda', enabled=use_amp):
                values = torch.softmax(model(joined), dim=1)
            for j, (rotation, flip) in enumerate(group):
                value = inverse_view(values[j*n:(j+1)*n], rotation, flip)
                probability_batch = value if probability_batch is None else probability_batch + value
            del joined, values
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
