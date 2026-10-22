# Model349 inference dependency; extracted from the recorded source snapshot.

# Unused training/search entrypoints are intentionally omitted.

from __future__ import annotations


from pathlib import Path

import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

from PIL import Image

from torch.cuda.amp import autocast

from advanced_inference import blend_window

from advanced_inference import inverse_view

from advanced_inference import transform_view

from advanced_inference import views

from encoder_structure_search import ASPPUNetPP

from improve_five_causes import NUM_CLASSES

from improve_five_causes import scaled_pair

class TeamResidualASPP(nn.Module):
    """Residual ASPP variant described in the team report: d=1,2,4,8."""

    def __init__(self, channels: int = 512, branch_channels: int = 128) -> None:
        super().__init__()
        self.branches = nn.ModuleList([
            self._branch(channels, branch_channels, 1, 1),
            self._branch(channels, branch_channels, 3, 2),
            self._branch(channels, branch_channels, 3, 4),
            self._branch(channels, branch_channels, 3, 8),
        ])
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(branch_channels * 5, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _branch(channels: int, branch_channels: int, kernel: int, dilation: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(channels, branch_channels, kernel,
                      padding=dilation * (kernel // 2), dilation=dilation, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        values = [branch(x) for branch in self.branches]
        pooled = self.pool(x)
        pooled = F.interpolate(pooled, size=(height, width), mode="bilinear", align_corners=False)
        values.append(pooled)
        return x + self.scale * self.project(torch.cat(values, dim=1))

class TeamRepresentationASPP(ASPPUNetPP):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__(pretrained=pretrained)
        self.context = TeamResidualASPP()

def representation(image: np.ndarray) -> np.ndarray:
    """Return CxHxW approximately standardized raw/z/CDF channels."""
    gray = image[..., 0].astype(np.float32) / 255.0
    z = (gray - float(gray.mean())) / max(float(gray.std()), 1e-5)
    z = np.clip(z, -4.0, 4.0) / 3.0
    flat = gray.reshape(-1)
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(order.size, dtype=np.float32)
    ranks[order] = np.arange(order.size, dtype=np.float32)
    cdf = (ranks / max(order.size - 1, 1)).reshape(gray.shape)
    values = np.stack([
        (gray - 0.5) / 0.25,
        z,
        (cdf - 0.5) / 0.288675,
    ], axis=0)
    return values.astype(np.float32)

@torch.no_grad()
def predict_probabilities(model: nn.Module, image_path: Path, patch_size: int, stride: int,
                          device: torch.device, use_amp: bool, tile_batch_size: int) -> np.ndarray:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    scaled, _ = scaled_pair(image, np.zeros(image.shape[:2], dtype=np.int64), patch_size)
    height, width = scaled.shape[:2]
    features = representation(scaled)
    ys = [0] if height <= patch_size else list(range(0, height - patch_size + 1, stride))
    xs = [0] if width <= patch_size else list(range(0, width - patch_size + 1, stride))
    last_y, last_x = max(height - patch_size, 0), max(width - patch_size, 0)
    if ys[-1] != last_y:
        ys.append(last_y)
    if xs[-1] != last_x:
        xs.append(last_x)
    padded_height = max(height, ys[-1] + patch_size)
    padded_width = max(width, xs[-1] + patch_size)
    padded = np.pad(features, ((0, 0), (0, padded_height - height),
                               (0, padded_width - width)), mode="reflect")
    probability_sum = np.zeros((NUM_CLASSES, padded_height, padded_width), dtype=np.float32)
    weight_sum = np.zeros((padded_height, padded_width), dtype=np.float32)
    window = blend_window(patch_size, "gaussian")
    locations = [(top, left) for top in ys for left in xs]
    for start in range(0, len(locations), tile_batch_size):
        batch_locations = locations[start:start + tile_batch_size]
        inputs = torch.stack([
            torch.from_numpy(padded[:, top:top + patch_size, left:left + patch_size])
            for top, left in batch_locations
        ]).to(device, non_blocking=True)
        probabilities = None
        for rotation, flip in views("d4"):
            value = transform_view(inputs, rotation, flip)
            with autocast(enabled=use_amp):
                value = torch.softmax(model(value), dim=1)
            value = inverse_view(value, rotation, flip)
            probabilities = value if probabilities is None else probabilities + value
        probabilities = (probabilities / 8.0).float().cpu().numpy()
        for index, (top, left) in enumerate(batch_locations):
            probability_sum[:, top:top + patch_size, left:left + patch_size] += (
                probabilities[index] * window[None]
            )
            weight_sum[top:top + patch_size, left:left + patch_size] += window
    value = probability_sum / np.maximum(weight_sum[None], 1e-7)
    value = value[:, :height, :width]
    original_size = image.shape[1], image.shape[0]
    restored = np.stack([
        np.asarray(Image.fromarray(value[class_id]).resize(original_size, Image.Resampling.BILINEAR),
                   dtype=np.float32)
        for class_id in range(NUM_CLASSES)
    ])
    return restored / np.maximum(restored.sum(axis=0, keepdims=True), 1e-7)
