#!/usr/bin/env python3
"""Synthetic-only checks for Model382 two-seed sensitivity v2 gates."""

from __future__ import annotations

import numpy as np

from model382_two_seed_mean_sensitivity import summarize_pairs
from model382_two_seed_mean_sensitivity_v2_lock import inventory_masks


def make_rows(seed1: np.ndarray, seed2: np.ndarray, mean12: np.ndarray):
    rows = {}
    images_per_fold = (18, 18, 17, 17)
    groups_per_fold = (15, 15, 14, 15)
    for fold in range(1, 5):
        image_count = images_per_fold[fold - 1]
        group_count = groups_per_fold[fold - 1]
        extra = image_count - group_count
        for image_index in range(image_count):
            if image_index < group_count - extra:
                group_index = image_index
            else:
                group_index = group_count - extra + (image_index - (group_count - extra)) // 2
            group = f"A{fold * 100 + group_index + 1}"
            image_id = f"{group}_step{fold}_sample{image_index + 1}"
            rows[image_id] = {
                "group": group,
                "fold": fold,
                "step": fold,
                "class_iou": {"seed1": seed1.copy(), "seed2": seed2.copy(), "mean12": mean12.copy()},
            }
    return rows


def main() -> None:
    # Positive overall gain (+0.75pp) still fails when one class loses 3pp.
    class_harm = summarize_pairs(
        make_rows(np.full(4, 0.80), np.full(4, 0.80),
                  np.array([0.82, 0.82, 0.82, 0.77])),
        draws=500,
    )
    assert np.isclose(class_harm["comparisons"]["seed1"]["delta_pp"], 0.75)
    assert np.isclose(class_harm["comparisons"]["seed1"]["class_delta_pp"]["Eutectic_Si"], -3.0)
    assert class_harm["comparisons"]["seed1"]["point_delta_at_least_0_20pp"] is True
    assert class_harm["comparisons"]["seed1"]["all_class_drops_within_1pp"] is False
    assert class_harm["secondary_gate_passed"] is False

    # Synthetic filename inventory checks coverage and deterministic ordering;
    # no mask pixel values are read by this inventory-only helper.
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        counts = (18, 18, 17, 17)
        groups = (15, 15, 14, 15)
        for fold, (image_count, group_count) in enumerate(zip(counts, groups), start=1):
            mask_dir = root / f"oof_fold{fold}" / "masks"
            mask_dir.mkdir(parents=True)
            extra = image_count - group_count
            for index in range(image_count):
                if index < group_count - extra:
                    group_index = index
                else:
                    group_index = group_count - extra + (index - (group_count - extra)) // 2
                group = "A71" if fold == 1 and group_index == 0 else f"A{fold * 100 + group_index + 1}"
                image_id = f"{group}_step{4 if index == 0 else fold}_sample{index + 1}"
                (mask_dir / f"{image_id}_mask.png").write_bytes(b"synthetic mask bytes")
        hashes, inventory = inventory_masks(root)
        assert len(hashes) == 70
        assert inventory["image_count"] == 70
        assert inventory["group_count"] == 59
        assert inventory["step4_image_count"] > 0
        assert inventory["a71_image_count"] > 0
    print("Model382 secondary sensitivity v2 synthetic smoke passed.")


if __name__ == "__main__":
    main()
