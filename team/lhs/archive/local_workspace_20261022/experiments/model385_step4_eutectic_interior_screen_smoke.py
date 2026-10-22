#!/usr/bin/env python3
"""Synthetic checks for Model385 Stage-0 error-zone accounting."""

import numpy as np

from model385_step4_eutectic_interior_screen import eutectic_false_negative_counts


def main() -> None:
    target = np.full((25, 25), 3, dtype=np.uint8)
    target[0, :] = 0
    target[-1, :] = 0
    target[:, 0] = 0
    target[:, -1] = 0
    prediction = target.copy()
    prediction[12, 12] = 0  # well inside the Eutectic region
    prediction[2, 12] = 0   # within the boundary band

    counts = eutectic_false_negative_counts(target, prediction, k_px=8.0)
    assert counts["eutectic_false_negative_pixels"] == 2, counts
    assert counts["interior_fn_pixels_distance_ge_k"] == 1, counts
    assert counts["near_boundary_fn_pixels_distance_lt_k"] == 1, counts
    assert counts["interior_fn_prediction_counts_by_class"][0] == 1, counts
    assert counts["near_fn_prediction_counts_by_class"][0] == 1, counts
    print({"status": "pass", "counts": counts})


if __name__ == "__main__":
    main()
