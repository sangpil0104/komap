#!/usr/bin/env python3
"""Smoke test for the grouped train-OOF error-type diagnostic helpers."""

import numpy as np

from model397_train_oof_error_type_audit import _confusion, _summarize


def main() -> None:
    truth = np.asarray([[0, 0, 1], [2, 3, 3]], dtype=np.uint8)
    prediction = np.asarray([[0, 1, 1], [2, 0, 3]], dtype=np.uint8)
    matrix = _confusion(truth, prediction)
    assert matrix[0, 0] == 1 and matrix[0, 1] == 1
    assert matrix[1, 1] == 1
    assert matrix[2, 2] == 1
    assert matrix[3, 0] == 1 and matrix[3, 3] == 1
    row = {"group": "A1", "step": 4, "confusion": matrix.tolist()}
    summary = _summarize([row])
    al = summary["classes"]["Al"]
    assert al["pooled_pixels"]["tp"] == 1
    assert al["pooled_pixels"]["fp"] == 1
    assert al["pooled_pixels"]["fn"] == 1
    eutectic = summary["classes"]["Eutectic Si"]
    assert eutectic["gt_present_image_means"]["recall"] == 0.5
    print("Model397 train-OOF error audit smoke: PASS")


if __name__ == "__main__":
    main()
