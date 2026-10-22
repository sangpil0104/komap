#!/usr/bin/env python3
"""Synthetic test for Model386 8-connected interior components."""

import numpy as np

from model386_step4_interior_component_audit import components


def main() -> None:
    target = np.zeros((80, 120), dtype=np.uint8)
    target[15:55, 10:50] = 3
    target[15:55, 65:105] = 3
    prediction = target.copy()
    prediction[15:55, 10:50] = 2  # miss first island

    result = components(target, prediction)
    assert len(result) == 2, result
    assert sum(row["gt_eutectic_interior_pixels"] for row in result) == 1152, result
    assert sum(row["false_negative_pixels"] for row in result) == 576, result
    rates = sorted(row["false_negative_rate"] for row in result)
    assert rates == [0.0, 1.0], rates
    missed = next(row for row in result if row["false_negative_rate"] == 1.0)
    assert missed["prediction_destination_counts_by_class_id"][2] == 576, missed
    print({"status": "pass", "components": result})


if __name__ == "__main__":
    main()
