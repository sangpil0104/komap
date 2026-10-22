# Model349 inference dependency; extracted from the recorded source snapshot.

# Unused training/search entrypoints are intentionally omitted.

from __future__ import annotations


from typing import Tuple

import numpy as np

import torch

import torch.nn as nn

from PIL import Image

NUM_CLASSES = 4

CLASS_COLORS: Tuple[Tuple[int, int, int], ...] = (
    # blue  -> official Al3Ni
    (25, 76, 153),
    # green -> official Primary Si
    (76, 178, 76),
    # brown -> official Al
    (153, 127, 76),
    # gray  -> official Eutectic Si
    (204, 204, 204),
)

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)

IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

def decode_mask(mask: Image.Image) -> np.ndarray:
    rgb = np.asarray(mask.convert("RGB"), dtype=np.uint8)
    result = np.full(rgb.shape[:2], -1, dtype=np.int64)
    for class_id, color in enumerate(CLASS_COLORS):
        result[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)] = class_id
    if (result < 0).any():
        unknown = np.unique(rgb[result < 0].reshape(-1, 3), axis=0)
        raise ValueError(f"Unknown mask colors: {unknown[:10].tolist()}")
    return result

def encode_mask(mask: np.ndarray) -> Image.Image:
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(CLASS_COLORS):
        output[mask == class_id] = color
    return Image.fromarray(output, mode="RGB")

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
