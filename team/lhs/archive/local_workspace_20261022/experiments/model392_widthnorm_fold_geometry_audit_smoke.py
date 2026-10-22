#!/usr/bin/env python3
"""Synthetic-only test for the Model392 fold geometry metadata audit."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from model392_widthnorm_fold_geometry_audit import audit_fold


def save_image(path: Path, width: int) -> None:
    Image.fromarray(np.zeros((16, width, 3), dtype=np.uint8)).save(path)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="model392-geometry-smoke-") as temp:
        root = Path(temp)
        train = root / "train_fold1/images"
        heldout = root / "oof_fold1/images"
        train.mkdir(parents=True)
        heldout.mkdir(parents=True)
        for name, width in (("A1_step1_sample1", 40), ("A2_step1_sample1", 80),
                            ("A3_step2_sample1", 80)):
            save_image(train / f"{name}_image.png", width)
        save_image(heldout / "A4_step4_sample1_image.png", 80)

        result = audit_fold(root, 1)
        assert result["target_width_from_training_mode"] == 80
        assert result["heldout_target_width_control_images"] == 1
        assert result["group_overlap"] == []
        assert result["geometry_by_native_width"]["40"]["native_crop_and_tile_px"] == 128
        assert result["geometry_by_native_width"]["80"]["native_crop_and_tile_px"] == 256

        save_image(heldout / "A3_step4_sample1_image.png", 40)
        try:
            audit_fold(root, 1)
        except ValueError as error:
            assert "specimen leakage" in str(error)
        else:
            raise AssertionError("group overlap between train and held-out images was accepted")

    print("PASS: train-only modal width, geometry by native width, held-out control count, and group-disjointness guard.")
    print("Scope: temporary synthetic images only; no masks, predictions, OOF scores, valid, or test used.")


if __name__ == "__main__":
    main()
