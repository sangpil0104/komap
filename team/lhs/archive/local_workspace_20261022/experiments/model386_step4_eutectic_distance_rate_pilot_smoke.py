#!/usr/bin/env python3
"""Synthetic accounting and argmax-map input checks for Model386."""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from model386_step4_eutectic_distance_rate_pilot import analyze, eutectic_distance_bin_counts


def main() -> None:
    target = np.full((30, 30), 3, dtype=np.uint8)
    target[:, :3] = 0
    prediction = target.copy()
    for x in (3, 5, 7, 11):
        prediction[:, x] = 0

    result = eutectic_distance_bin_counts(target, prediction)
    bins = result["bins"]
    assert result["gt_eutectic_pixels"] == 810
    assert result["eutectic_false_negative_pixels"] == 120
    expected = {
        "0_to_lt2px": (60, 30),
        "2_to_lt4px": (60, 30),
        "4_to_lt8px": (120, 30),
        "8px_or_more": (570, 30),
    }
    for name, (support, misses) in expected.items():
        assert bins[name]["gt_eutectic_pixels"] == support, (name, bins[name])
        assert bins[name]["eutectic_false_negative_pixels"] == misses, (name, bins[name])
        assert np.isclose(bins[name]["false_negative_rate"], misses / support)

    with tempfile.TemporaryDirectory(prefix="model386-smoke-") as temp:
        root = Path(temp)
        cv_root, oof_root = root / "cv", root / "oof"
        palette = np.asarray([(153, 127, 76), (25, 76, 153), (76, 178, 76), (204, 204, 204)], dtype=np.uint8)
        for fold, step_id in ((1, "A1_step4_sample1"), (2, "A2_step4_sample1")):
            mask_dir = cv_root / f"oof_fold{fold}" / "masks"
            out_dir = oof_root / f"fold{fold}_oof"
            mask_dir.mkdir(parents=True)
            out_dir.mkdir(parents=True)
            keys = [step_id, f"A{fold + 10}_step1_sample1"]
            predictions = {}
            for image_id in keys:
                rgb = np.zeros((30, 30, 3), dtype=np.uint8)
                rgb[:] = palette[3]
                rgb[:, :3] = palette[0]
                Image.fromarray(rgb).save(mask_dir / f"{image_id}_mask.png")
                target_ids = np.full((30, 30), 3, dtype=np.uint8)
                target_ids[:, :3] = 0
                pred = target_ids.copy()
                for x in (3, 5, 7, 11):
                    pred[:, x] = 0
                predictions[image_id] = pred
            np.savez_compressed(out_dir / "valid_predictions.npz", **predictions)
        integrated = analyze(cv_root, oof_root, (1, 2), draws=200)
        assert integrated["oof_images_total"] == 4
        assert integrated["step4_images"] == 2
        assert integrated["prediction_input_type_by_fold"]["1"].startswith("uint8 argmax")
        assert integrated["distance_bins"]["0_to_lt2px"]["false_negative_rate"] == 0.5

    print({"status": "pass", "counts": result})


if __name__ == "__main__":
    main()
