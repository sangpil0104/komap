# Model349 inference dependency; extracted from the recorded source snapshot.

# Unused training/search entrypoints are intentionally omitted.

from __future__ import annotations


from pathlib import Path

import albumentations as A

import numpy as np

from albumentations.pytorch import ToTensorV2

from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)

IMAGENET_STD = (0.229, 0.224, 0.225)

def load_image_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))

def build_eval_transform() -> A.Compose:
    return A.Compose([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])
