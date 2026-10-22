#!/usr/bin/env python3
"""Synthetic end-to-end smoke test for the CPU-only Model381 diagnostic."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from model381_boundary_error_diagnostic import MODELS, PALETTE, run


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="model381-smoke-") as temporary:
        root = Path(temporary)
        cv_root = root / "cv"
        model_roots = {name: root / name for name in MODELS}
        fold_items: dict[int, list[tuple[str, str, np.ndarray]]] = {fold: [] for fold in range(1, 5)}
        for image_index in range(70):
            group_number = image_index + 1 if image_index < 59 else image_index - 58
            sample_number = 1 if image_index < 59 else 2
            group = f"A{group_number}"
            image_id = f"{group}_step1_sample{sample_number}"
            fold = (group_number - 1) % 4 + 1
            truth = np.empty((24, 24), dtype=np.uint8)
            truth[:, :12] = 1
            truth[:, 12:] = 3
            truth[:4, :] = 0
            truth[20:, :] = 2
            fold_items[fold].append((image_id, group, truth))

        for fold, items in fold_items.items():
            image_dir = cv_root / f"oof_fold{fold}/images"
            mask_dir = cv_root / f"oof_fold{fold}/masks"
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            archives = {name: {} for name in MODELS}
            for image_id, _group, truth in items:
                rgb = PALETTE[truth]
                Image.fromarray(rgb).save(mask_dir / f"{image_id}_mask.png")
                Image.fromarray(np.zeros_like(rgb)).save(image_dir / f"{image_id}_image.png")
                for model_index, model in enumerate(MODELS):
                    prediction = truth.copy()
                    prediction[8 + model_index, 11] = 3
                    probability = np.full((4, *truth.shape), 0.0025, dtype=np.float32)
                    for class_id in range(4):
                        probability[class_id, prediction == class_id] = 0.9925
                    archives[model][image_id] = probability
            for model, archive in archives.items():
                output = model_roots[model] / f"fold{fold}_oof/valid_probabilities.npz"
                output.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(output, **archive)

        result = run(cv_root, model_roots, draws=200)
        assert result["models"]["Model374"]["images"] == 70
        assert result["three_model_error_location_jaccard"]["groups"] > 0
        assert set(result["model378_diagnostic_go_gate"]) == {
            "at_least_60pct_pair_swap_errors_within_2px",
            "signed_offset_median_abs_le_0_5px_and_ci_contains_zero",
            "three_model_group_equal_error_jaccard_at_least_0_50",
            "374_restricted_oracle_gain_at_least_0_50pp",
        }
        print({
            "images": 70,
            "groups": 59,
            "model381_pipeline": "passed",
            "diagnostic_go": result["model378_diagnostic_go"],
        })


if __name__ == "__main__":
    main()
