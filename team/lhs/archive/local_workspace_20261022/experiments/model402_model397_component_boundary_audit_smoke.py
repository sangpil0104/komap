#!/usr/bin/env python3
"""Synthetic-only checks for Model402 geometry and oracle calculations."""

import numpy as np

from model402_model397_component_boundary_audit import (
    AREA_BINS,
    BOOTSTRAP_DRAWS,
    _area_bin_index,
    _audit_arm_image,
    _component_map,
    _distance_bin_map,
    _group_bootstrap_ci,
    _iou_vector,
    _ious_from_confusion,
    _new_arm_cells,
)


def main() -> None:
    truth = np.zeros((8, 8), dtype=np.uint8)
    truth[2:4, 2:4] = 1
    truth[2:4, 5:7] = 3
    prediction = truth.copy()
    prediction[2, 2] = 0
    prediction[2, 5] = 1

    labels, areas, component_bins = _component_map(truth, 1)
    assert areas[1] == 4
    assert _area_bin_index(4) == 0
    assert np.all(component_bins[truth == 1] == 0)
    assert np.all(labels[truth == 1] == 1)
    distance_bins = _distance_bin_map(truth, 1)
    assert np.all(distance_bins[truth == 1] == 0)

    base_iou = _iou_vector(truth, prediction)
    confusion = np.bincount(truth.ravel().astype(np.int64) * 4 + prediction.ravel().astype(np.int64),
                            minlength=16).reshape(4, 4)
    selected = truth == 1
    selected_pred = np.bincount(prediction[selected], minlength=4)
    corrected_confusion = confusion.copy()
    corrected_confusion[1, :] -= selected_pred
    corrected_confusion[1, 1] += int(selected.sum())
    corrected = prediction.copy()
    corrected[selected] = 1
    assert np.allclose(_ious_from_confusion(corrected_confusion), _iou_vector(truth, corrected))
    assert np.all(_ious_from_confusion(corrected_confusion) >= base_iou - 1e-12)

    arm = _new_arm_cells()
    _audit_arm_image(arm, truth, prediction, "A1_step1_sample1", "A1", 1)
    cell = arm["component"][(1, AREA_BINS[0][2])]
    assert cell["support_pixels"] == 4
    assert cell["false_negative_pixels"] == 1
    assert cell["components"] == 1 and cell["components_with_any_fn"] == 1
    assert arm["oracle_component"][(1, AREA_BINS[0][2])][0]["delta_pp"] > 0

    rows = [{"group": f"A{fold}", "fold": fold, "delta_pp": 1.0 + fold / 10}
            for fold in range(1, 5)]
    ci = _group_bootstrap_ci(rows, 402081)
    assert len(ci) == 2 and ci[0] <= np.mean([row["delta_pp"] for row in rows]) <= ci[1]
    assert BOOTSTRAP_DRAWS == 20_000
    print("Model402 synthetic geometry/oracle/bootstrap smoke: PASS; no dataset accessed.")


if __name__ == "__main__":
    main()
