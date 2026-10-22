#!/usr/bin/env python3
"""Synthetic test for Model383 v2 Shapley attribution and bootstrap."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from model374_train_oof_error_diagnostic import PALETTE
from model383_no_training_gate_v2 import analyze


def make_probabilities(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    result = np.full((4, *truth.shape), 0.001, dtype=np.float32)
    for class_id in range(4):
        result[class_id, prediction == class_id] = 0.997
    bad = prediction != truth
    result[:, bad] = 0.005
    ys, xs = np.where(bad)
    result[truth[bad], ys, xs] = 0.45
    result[prediction[bad], ys, xs] = 0.54
    return result


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="model383-v2-smoke-") as temporary:
        root = Path(temporary)
        cv_root, model374_root, model376_root = root / "cv", root / "m374", root / "m376"
        for fold in (3, 4):
            masks = cv_root / f"oof_fold{fold}" / "masks"
            masks.mkdir(parents=True)
            out374, out376 = model374_root / f"fold{fold}_oof", model376_root / f"fold{fold}_oof"
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
                Image.fromarray(PALETTE[truth]).save(masks / f"{image_id}_mask.png")
                archive374[image_id] = make_probabilities(truth, pred374)
                archive376[image_id] = make_probabilities(truth, pred376)
            np.savez_compressed(out374 / "valid_probabilities.npz", **archive374)
            np.savez_compressed(out376 / "valid_probabilities.npz", **archive376)

        result = analyze(cv_root, model374_root, model376_root, draws=200)
        assert result["images"] == 34 and result["groups"] == 34
        assert result["bootstrap"]["draws"] == 200
        assert np.isclose(
            result["model376_shapley_band_contribution_pp"] + result["model376_shapley_outside_contribution_pp"],
            result["model376_full_delta_vs374_pp"], atol=1e-10,
        )
        assert result["otherwise"] == "inconclusive; this result alone does not authorize Model383 training"
        print({
            "images": result["images"],
            "groups": result["groups"],
            "shapley_sums_to_full_delta": True,
            "cancellation": result["cancel_model383_for_proxy_evidence"],
            "no_automatic_go": True,
        })


if __name__ == "__main__":
    main()
