from __future__ import annotations

import numpy as np
import pytest
import torch
from model_lab.dataset import (
    PALETTE,
    compute_3channels,
    pad_to_multiple,
    KomapTrainDataset,
    KomapEvalDataset,
)
from model_lab.workspace import get_dataset_samples, load_config


def test_compute_3channels_and_padding():
    gray = np.linspace(0.0, 1.0, 10000, dtype=np.float32).reshape(100, 100)
    ch3 = compute_3channels(gray)
    assert ch3.shape == (3, 100, 100)
    # Check bounds
    assert ch3[0].min() >= 0.0 and ch3[0].max() <= 1.0
    assert ch3[2].min() >= 0.0 and ch3[2].max() <= 1.0

    padded = pad_to_multiple(ch3, multiple=32)
    assert padded.shape == (3, 128, 128)


def test_dataset_splits_count():
    train_samples = get_dataset_samples("train")
    valid_samples = get_dataset_samples("valid")
    test_samples = get_dataset_samples("test")

    assert len(train_samples) == 70, f"Expected 70 train images, got {len(train_samples)}"
    assert len(valid_samples) == 20, f"Expected 20 valid images, got {len(valid_samples)}"
    assert len(test_samples) == 10, f"Expected 10 test images, got {len(test_samples)}"


def test_train_dataset_sampling_and_shapes():
    train_samples = get_dataset_samples("train")
    cfg = load_config()
    stage = {"name": "test_s0", "mode": "patch", "core_size": 256, "max_updates": 5}
    ds = KomapTrainDataset(train_samples, cfg, stage)

    item = ds[0]
    assert "image" in item and "mask" in item
    # 256 + 2*64 = 384
    assert item["image"].shape == (3, 384, 384)
    assert item["mask"].shape == (384, 384)
    assert item["image"].dtype == torch.float32
    assert item["mask"].dtype == torch.int64


def test_eval_dataset_shapes():
    valid_samples = get_dataset_samples("valid")
    ds = KomapEvalDataset(valid_samples, has_masks=True)
    item = ds[0]

    assert "image" in item and "mask" in item
    assert "original_h" in item and "original_w" in item
    assert item["image"].ndim == 3
    assert item["image"].shape[0] == 3
    assert item["image"].shape[1] % 32 == 0
    assert item["image"].shape[2] % 32 == 0

