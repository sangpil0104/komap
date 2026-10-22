#!/usr/bin/env python3
"""Synthetic smoke test for Model374 group-coverage audit."""

from __future__ import annotations

import numpy as np

from model374_train_oof_group_coverage_audit import analyze


def main() -> None:
    rows = []
    for fold in range(1, 5):
        for within in range(3):
            group = f"A{fold * 10 + within}"
            offset = fold * 0.2 + within * 0.03
            rows.append({
                "image_id": f"{group}_step{(fold + within) % 4 + 1}_sample1",
                "group": group,
                "fold": fold,
                "score": 0.9 - offset,
                "appearance": np.arange(15, dtype=np.float64) * 0.01 + offset,
                "step": f"step{(fold + within) % 4 + 1}",
                "resolution": "64x64" if within == 0 else "80x64",
                "class_fraction": np.asarray([0.55, 0.2, 0.15, 0.1]) +
                    np.asarray([0.0, offset * 0.1, 0.0, -offset * 0.1]),
            })
    result = analyze(rows, bootstrap_draws=200, seed=7)
    assert result["images"] == 12
    assert result["specimen_groups"] == 12
    assert set(result["distance_score_associations"]) == {
        "appearance", "step_resolution", "class_mix"
    }
    for fold, values in result["folds"].items():
        assert values["heldout_groups"] == 3
        for row in values["rows"]:
            assert row["group"] not in {
                row["nearest_train_group_appearance"],
                row["nearest_train_group_step"],
                row["nearest_train_group_resolution"],
                row["nearest_train_group_class_mix"],
            }
            assert 0 <= row["step_resolution_distance"] <= 1
            assert 0 <= row["class_mix_distance"] <= 1
    print("Model374 group-coverage synthetic smoke passed: heldout groups excluded, three fixed distance families summarized, group bootstrap completed.")


if __name__ == "__main__":
    main()
