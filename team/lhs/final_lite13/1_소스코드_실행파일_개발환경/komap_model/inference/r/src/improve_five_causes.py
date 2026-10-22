# R ensemble inference dependency.

# Unused training/search entrypoints are intentionally omitted.

from __future__ import annotations


from typing import List

from typing import Tuple

import numpy as np

import torch

from PIL import Image

from unet_resnet34 import IMAGENET_MEAN

from unet_resnet34 import IMAGENET_STD

from unet_resnet34 import NUM_CLASSES

from unet_resnet34 import decode_mask

from unet_resnet34 import encode_mask

def normalize_image(image_np: np.ndarray) -> torch.Tensor:
    image_float = image_np.astype(np.float32) / 255.0
    image_float = (image_float - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(image_float.transpose(2, 0, 1)).float()

def scaled_pair(image: np.ndarray, mask: np.ndarray, patch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Scale only when the short side is smaller than a patch; never distort ratio."""
    height, width = image.shape[:2]
    scale = max(1.0, float(patch_size) / float(min(height, width)))
    if abs(scale - 1.0) < 1e-6:
        return image, mask
    new_size = (max(patch_size, int(round(width * scale))), max(patch_size, int(round(height * scale))))
    image_out = np.asarray(Image.fromarray(image).resize(new_size, Image.Resampling.BICUBIC), dtype=np.uint8)
    mask_out = decode_mask(encode_mask(mask).resize(new_size, Image.Resampling.NEAREST))
    return image_out, mask_out

def tile_starts(length: int, patch_size: int, stride: int) -> List[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts
