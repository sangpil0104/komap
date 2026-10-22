#!/usr/bin/env python3
"""Synthetic smoke test for the Model383 untouched-fold no-training gate."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from model374_train_oof_error_diagnostic import PALETTE
from model383_no_training_gate import analyze


def make_probabilities(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    result = np.full((4, *truth.shape), 0.001, dtype=np.float32)
    for class_id in range(4):
        result[class_id, prediction == class_id] = 0.997
    bad = prediction != truth
    result[:, bad] = 0.005
    result[truth[bad], np.where(bad)[0], np.where(bad)[1]] = 0.45
    result[prediction[bad], np.where(bad)[0], np.where(bad)[1]] = 0.54
    return result


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="model383-stop-gate-smoke-") as temporary:
        root = Path(temporary)
        cv_root = root / "cv"
        model374_root = root / "m374"
        model376_root = root / "m376"
        for fold in (3, 4):
            masks = cv_root / f"oof_fold{fold}" / "masks"
            masks.mkdir(parents=True)
            out374 = model374_root / f"fold{fold}_oof"
            out376 = model376_root / f"fold{fold}_oof"
            out374.mkdir(parents=True)
            out376.mkdir(parents=True)
            archive374, archive376 = {}, {}
            start = 1 if fold == 3 else 18
            for group_number in range(start, start + 17):
                image_id = f"A{group_number}_step1_sample1"
                truth = np.ones((32, 32), dtype=np.uint8)
                truth[:, 16:] = 3
                pred374 = truth.copy()
                pred376 = truth.copy()
                pred374[12, 15] = 3
                rgb = PALETTE[truth]
                Image.fromarray(rgb).save(masks / f"{image_id}_mask.png")
                archive374[image_id] = make_probabilities(truth, pred374)
                archive376[image_id] = make_probabilities(truth, pred376)
            np.savez_compressed(out374 / "valid_probabilities.npz", **archive374)
            np.savez_compressed(out376 / "valid_probabilities.npz", **archive376)

        result = analyze(cv_root, model374_root, model376_root)
        assert result["images"] == 34 and result["groups"] == 34
        assert result["folds"] == [3, 4]
        assert set(result["no_training_gates"]) == {
            "fold34_band_oracle_gain_at_least_5pp",
            "fold34_errors_more_than_3px_from_band_below_15pct",
            "fold34_band_errors_top2_margin_lt_0_20_at_least_30pct",
            "model376_does_not_show_low_band_headroom_recovery",
        }
        assert 0 <= result["band_error_fraction_gt_top2_and_margin_lt_0_20"] <= 1
        print({
            "images": result["images"],
            "groups": result["groups"],
            "scope": result["purpose"],
            "go_to_source_review": result["go_to_model383_source_review"],
        })


if __name__ == "__main__":
    main()
