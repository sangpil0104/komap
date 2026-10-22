#!/usr/bin/env python3
"""Synthetic CPU checks for Model377 OOF scoring and paired gates."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


MODULE_PATH = Path(__file__).with_name("model377_oof_evaluation.py")
SPEC = importlib.util.spec_from_file_location("model377_oof_evaluation", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main() -> None:
    truth = np.asarray([[0, 3, 1], [0, 3, 1]], dtype=np.uint8)
    support = MODULE.adjacent_support(truth, side=3, neighbor_class=1)
    assert support.sum() == 2

    probs = np.zeros((4, *truth.shape), dtype=np.float32)
    for y in range(truth.shape[0]):
        for x in range(truth.shape[1]):
            probs[truth[y, x], y, x] = 1
    assert np.allclose(MODULE.class_ious(probs, truth), np.asarray([1.0, 1.0, 0.0, 1.0]))

    delta, ci = MODULE.group_bootstrap(np.asarray([1.0, 2.0, 3.0]), ["A1", "A2", "A3"], 500, 1)
    assert delta == 2.0 and len(ci) == 2
    paired = MODULE.paired_group_delta({"A1": 0.2, "A2": 0.3},
                                       {"A1": 0.1, "A2": 0.2}, 500, 2)
    assert abs(paired["delta_pp"] + 10.0) < 1e-8

    with tempfile.TemporaryDirectory(prefix="model377-eval-smoke-") as temp:
        root = Path(temp)
        cv_root = root / "cv"
        model_roots = {name: root / name for name in ("model374", "model376", "model377")}
        image_index = 0
        fold_sizes = (18, 18, 17, 17)
        patterns = np.asarray((
            (0, 0, 1, 1, 3, 3, 2, 2),
            (0, 0, 3, 3, 1, 1, 2, 2),
            (0, 0, 2, 2, 3, 3, 1, 1),
        ), dtype=np.uint8)
        truth = np.repeat(patterns, repeats=3, axis=0)
        truth_rgb = MODULE.PALETTE[truth]
        probs = np.full((4, *truth.shape), 0.01, dtype=np.float32)
        for class_id in range(4):
            probs[class_id, truth == class_id] = 0.97

        for fold, count in enumerate(fold_sizes, 1):
            mask_dir = cv_root / f"oof_fold{fold}" / "masks"
            image_dir = cv_root / f"oof_fold{fold}" / "images"
            mask_dir.mkdir(parents=True)
            image_dir.mkdir(parents=True)
            predictions = {name: {} for name in model_roots}
            for _ in range(count):
                group_id = 100 + (image_index % 59)
                image_id = f"A{group_id}_step1_sample{image_index}"
                Image.fromarray(truth_rgb).save(mask_dir / f"{image_id}_mask.png")
                Image.fromarray(truth_rgb).save(image_dir / f"{image_id}_image.png")
                for model in model_roots:
                    predictions[model][image_id] = probs
                image_index += 1
            for model, model_root in model_roots.items():
                output = model_root / f"fold{fold}_oof"
                output.mkdir(parents=True)
                np.savez_compressed(output / "valid_probabilities.npz", **predictions[model])

        result = MODULE.run(cv_root, model_roots["model374"], model_roots["model376"],
                            model_roots["model377"], draws=200)
        assert result["images"] == 70 and result["specimen_groups"] == 59
        assert abs(result["adoption_comparison_model377_vs_model374"]["delta_pp"]) < 1e-8
        assert result["adoption_comparison_model377_vs_model374"]["resolution_stratified"]["8x9"]["images"] == 70
        assert result["adoption_comparison_model377_vs_model374"]["specimen_group_delta_summary_pp"]["total_groups"] == 59
        assert set(result["adoption_comparison_model377_vs_model374"]["predicted_area_fraction_comparison"]) == set(MODULE.CLASS_NAMES)
        assert result["directional_mechanism_model377_vs_model376"]["rates"]
        assert "eutectic_gt_distance1_from_al3ni:Eutectic_to_Al3Ni" in result[
            "directional_mechanism_model377_vs_model376"
        ]["Eutectic_error_by_taxicab_distance_from_Al3Ni"]
    print("Model377 OOF evaluation smoke passed")


if __name__ == "__main__":
    main()
