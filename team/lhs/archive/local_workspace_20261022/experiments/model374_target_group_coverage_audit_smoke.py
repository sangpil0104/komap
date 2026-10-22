#!/usr/bin/env python3
"""Synthetic smoke test for label-free target group coverage audit."""

from __future__ import annotations

import numpy as np

from model374_target_group_coverage_audit import analyze


def row(group: str, value: float) -> dict:
    return {"group": group, "features": np.arange(15, dtype=np.float64) * 0.01 + value,
            "name": f"{group}_step1_sample1"}


def main() -> None:
    train_rows = [row(f"A{fold * 10 + offset}", float(fold * 10 + offset))
                  for fold in range(1, 5) for offset in (0, 1)]
    targets = {
        "valid": [row("A10", 10.1), row("A90", 19.0)],
        "test": [row("A91", 20.0)],
    }
    coverage = {"folds": {str(fold): {"rows": [
        {"group": f"A{fold * 10 + offset}", "appearance_distance": 0.3 + offset * 0.2}
        for offset in (0, 1)
    ]} for fold in range(1, 5)}}
    result = analyze(train_rows, targets, coverage)
    assert result["splits"]["valid"]["images"] == 2
    assert result["splits"]["valid"]["groups_sharing_prefix_with_train"] == 1
    rows = {item["group"]: item for item in result["splits"]["valid"]["per_group"]}
    assert rows["A10"]["train_prefix_seen"] is True
    assert rows["A90"]["train_prefix_seen"] is False
    assert len(rows["A10"]["fold_comparisons"]) == 4
    assert all(0 <= item["percentile_vs_fold_oof_group_distances"] <= 1
               for item in rows["A90"]["fold_comparisons"])
    print("Model374 target coverage synthetic smoke passed: image-only groups, same-prefix sensitivity, and fold-calibrated percentiles.")


if __name__ == "__main__":
    main()
