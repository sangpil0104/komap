#!/usr/bin/env python3
"""Synthetic-only checks for the locked Model382 two-seed sensitivity."""

from __future__ import annotations

import numpy as np

from model382_two_seed_mean_sensitivity import summarize_pairs
from model382_oof_ensemble_compare import class_ious


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
    # Equal per-class gains pass the paired gate and the strict three-fold rule.
    good = summarize_pairs(
        make_rows(np.full(4, 0.70), np.full(4, 0.71), np.full(4, 0.76)),
        draws=500,
    )
    assert good["secondary_gate_passed"] is True
    assert good["adoption_gate_passed"] is True
    assert good["folds_where_mean12_beats_both_seeds"] == 4

    # E beating only one singleton is insufficient, even with a positive average delta.
    one_seed_only = summarize_pairs(
        make_rows(np.full(4, 0.70), np.full(4, 0.80), np.full(4, 0.76)),
        draws=500,
    )
    assert one_seed_only["secondary_gate_passed"] is False
    assert one_seed_only["folds_where_mean12_beats_both_seeds"] == 0

    # A >1pp class drop relative to either reference fails class safety.
    class_harm = summarize_pairs(
        make_rows(np.array([0.80, 0.80, 0.80, 0.80]),
                  np.array([0.80, 0.80, 0.80, 0.80]),
                  np.array([0.80, 0.80, 0.80, 0.78])),
        draws=500,
    )
    assert class_harm["secondary_gate_passed"] is False

    # Official empty-union policy remains zero for a class absent in both masks.
    target = np.zeros((3, 3), dtype=np.uint8)
    probabilities = np.zeros((4, 3, 3), dtype=np.float32)
    probabilities[0] = 1.0
    ious = class_ious(probabilities, target)
    assert ious.tolist() == [1.0, 0.0, 0.0, 0.0]
    print("Model382 secondary sensitivity synthetic smoke passed.")


if __name__ == "__main__":
    main()
