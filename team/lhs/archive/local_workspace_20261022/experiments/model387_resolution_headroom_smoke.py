#!/usr/bin/env python3
"""Synthetic checks for Model387 band counts, oracle upper bounds, and empty=0."""

from __future__ import annotations

import numpy as np

from model387_resolution_headroom_audit import (
    BIN_NAMES,
    diagnose_image,
    image_miou,
    summarize,
)


def main() -> None:
    target = np.zeros((40, 40), dtype=np.uint8)
    target[:20, 20:] = 1
    target[20:, :20] = 2
    target[20:, 20:] = 3
    prediction = target.copy()
    prediction[19, 10] = 1   # class-0 boundary-band false negative
    prediction[10, 10] = 2   # class-0 interior false negative

    diagnostic = diagnose_image(target, prediction)
    assert diagnostic["baseline_image_miou"] < 1.0
    assert diagnostic["bins"][BIN_NAMES[0]]["fn_by_class"][0] >= 1
    assert diagnostic["bins"][BIN_NAMES[0]]["oracle_miou_gain"] > 0.0
    assert diagnostic["bins"][BIN_NAMES[3]]["fn_by_class"][0] >= 1
    assert diagnostic["bins"][BIN_NAMES[3]]["oracle_miou_gain"] > 0.0
    assert sum(b["pixels"] for b in diagnostic["bins"].values()) == target.size

    one_class = np.zeros((8, 8), dtype=np.uint8)
    score, class_scores = image_miou(one_class, one_class.copy())
    assert score == 0.25
    assert class_scores == [1.0, 0.0, 0.0, 0.0]
    rows = [
        {
            "image_id": f"{group}_step1_image.png",
            "specimen_group": group,
            "fold": fold,
            "diagnostic": diagnose_image(target, prediction),
        }
        for fold, groups in ((3, ("A1", "A2")), (4, ("A3", "A4")))
        for group in groups
    ]
    summary = summarize(rows, (3, 4), draws=100)
    assert summary["images"] == 4 and summary["groups"] == 4
    assert summary["oracle_miou_gain_by_bin"][BIN_NAMES[0]]["group_bootstrap_95ci"] is not None
    assert summary["class_distance_rates"]["Al"]["near_minus_interior_rate_difference"] is not None
    print("Model387 synthetic smoke passed: distance bands partition pixels, near/interior errors detected, oracle gains positive, empty union=0.")


if __name__ == "__main__":
    main()
