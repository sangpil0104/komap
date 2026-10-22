#!/usr/bin/env python3
"""Synthetic tests for Model388 sampler-dose and component oracle metrics."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from model388_component_sampler_audit import (
    PALETTE,
    component_bin_oracle_gains,
    exposure_audit,
)


def make_target() -> np.ndarray:
    target = np.zeros((100, 100), dtype=np.uint8)
    target[0:10, 0:10] = 1       # 100 px component
    target[20:40, 20:40] = 1     # 400 px component
    target[50:60, 0:10] = 2
    target[50:90, 50:90] = 3     # 1600 px component
    return target


def write_mask(path: Path, ids: np.ndarray) -> None:
    rgb = PALETTE[ids]
    Image.fromarray(rgb).save(path)


def main() -> None:
    target = make_target()
    prediction = target.copy()
    prediction[0:5, 0:10] = 0
    diagnostic = component_bin_oracle_gains(target, prediction)
    small = diagnostic["cells"]["1:64_to_lt256"]
    assert small["component_count"] == 1
    assert small["fn_pixels"] == 50
    assert small["oracle_miou_gain"] > 0

    with tempfile.TemporaryDirectory(prefix="model388-smoke-") as temp:
        root = Path(temp)
        for fold in range(1, 5):
            mask_dir = root / f"oof_fold{fold}" / "masks"
            mask_dir.mkdir(parents=True)
            write_mask(mask_dir / f"A{fold}_step1_sample1_mask.png", target)
        result = exposure_audit(root, (3, 4))
    fold3 = result["per_outer_fold"]["3"]
    class1 = fold3["classes"]["Al3Ni"]
    assert abs(class1["class_total_anchor_delta"]) < 1e-9
    assert abs(class1["component_size_bins"]["64_to_lt256"]["candidate_to_current_dose_ratio"] - 2.5) < 1e-6
    assert fold3["training_images"] == 3  # held-out fold is excluded
    assert result["oof_predictions_read"] is False
    print("Model388 synthetic smoke passed: held-out fold excluded, class anchor totals preserved, size-bin dose shifts, component oracle gain computed.")


if __name__ == "__main__":
    main()
