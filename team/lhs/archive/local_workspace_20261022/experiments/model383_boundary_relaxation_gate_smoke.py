#!/usr/bin/env python3
"""Synthetic test for the Model383 CPU-only label-relaxation feasibility gate."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from model374_train_oof_error_diagnostic import PALETTE
from model383_boundary_relaxation_gate import analyze, relaxed_band


def main() -> None:
    truth = np.ones((32, 32), dtype=np.uint8)
    truth[:, 16:] = 3
    band = relaxed_band(truth)
    assert band[:, 15:17].all()
    assert not band[:, :14].any()
    assert not band[:, 17:].any()

    with tempfile.TemporaryDirectory(prefix="model383-blr-smoke-") as temporary:
        root = Path(temporary)
        cv_root = root / "cv"
        model_root = root / "model374"
        mask_dir = cv_root / "oof_fold1" / "masks"
        oof_dir = model_root / "fold1_oof"
        mask_dir.mkdir(parents=True)
        oof_dir.mkdir(parents=True)
        archives = {}
        for index in range(18):
            image_id = f"A{index + 1}_step1_sample1"
            Image.fromarray(PALETTE[truth]).save(mask_dir / f"{image_id}_mask.png")
            prediction = truth.copy()
            prediction[12, 15] = 3
            probabilities = np.full((4, *truth.shape), 0.001, dtype=np.float32)
            for class_id in range(4):
                probabilities[class_id, prediction == class_id] = 0.997
            archives[image_id] = probabilities
        np.savez_compressed(oof_dir / "valid_probabilities.npz", **archives)

        result = analyze(cv_root, model_root, folds=(1,))
        assert result["images"] == 18
        assert result["scope"] == "partial train OOF; exploratory only"
        assert result["preregistered_cpu_gate"] is None
        assert result["go"] is None
        assert result["boundary_oracle_gain_pp"] > 0
        print({
            "relaxed_band": "passed",
            "partial_fold_oracle_gain_pp": result["boundary_oracle_gain_pp"],
            "partial_data_cannot_return_go": result["go"] is None,
        })


if __name__ == "__main__":
    main()
