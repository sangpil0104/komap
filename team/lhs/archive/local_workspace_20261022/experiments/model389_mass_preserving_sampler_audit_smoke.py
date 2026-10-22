#!/usr/bin/env python3
"""Synthetic checks for Model389 mass preservation and OOF gates."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from model389_mass_preserving_sampler_audit import (
    PALETTE,
    component_areas,
    expected_component_anchors,
    exposure_audit,
    image_outcomes,
)


def make_mask() -> np.ndarray:
    ids = np.zeros((100, 100), dtype=np.uint8)
    ids[0:10, 0:10] = 1
    ids[20:40, 20:40] = 1
    ids[20:50, 50:80] = 1
    ids[60:70, 0:10] = 3
    ids[72:92, 12:32] = 3
    ids[0:4, 80:84] = 3
    ids[50:60, 80:90] = 2
    return ids


def save_mask(path: Path, ids: np.ndarray) -> None:
    Image.fromarray(PALETTE[ids]).save(path)


def main() -> None:
    areas = component_areas(make_mask(), 3)
    budget = 6.0
    expected = expected_component_anchors(areas, budget)
    tiny = areas < 64
    assert abs(expected["model389"][tiny].sum() - expected["current"][tiny].sum()) < 1e-9
    assert abs(expected["model389"].sum() - budget) < 1e-9
    target_bin = (areas >= 64) & (areas < 256)
    current_target = expected["current"][target_bin].sum()
    new_target = expected["model389"][target_bin].sum()
    assert new_target / current_target > 1.0

    target = make_mask()
    prediction = target.copy()
    prediction[60:66, 0:10] = 0  # only 40/100 pixels correct: target component is missed
    outcomes = image_outcomes(target, prediction)
    assert outcomes["target_components"] == 1
    assert outcomes["target_missed"] == 1
    assert outcomes["comparator_components"] == 1
    assert outcomes["target_oracle_gain"] > 0

    with tempfile.TemporaryDirectory(prefix="model389-smoke-") as temp:
        root = Path(temp)
        for fold in range(1, 5):
            mask_dir = root / f"oof_fold{fold}" / "masks"
            mask_dir.mkdir(parents=True)
            save_mask(mask_dir / f"A{fold}_step1_sample1_mask.png", target)
        result = exposure_audit(root, (3, 4))
    fold3 = result["per_outer_fold"]["3"]
    class3 = fold3["classes"]["Eutectic_Si"]
    assert fold3["max_absolute_per_image_class_total_anchor_delta_model389"] < 1e-9
    assert abs(class3["area_bins"]["1_to_lt64"]["model389_to_current_ratio"] - 1.0) < 1e-9
    assert fold3["training_images"] == 3
    assert result["oof_values_read"] is False
    print("Model389 synthetic smoke passed: tiny mass preserved, eligible mass redistributed, held-out fold excluded, component miss/oracle computed.")


if __name__ == "__main__":
    main()
