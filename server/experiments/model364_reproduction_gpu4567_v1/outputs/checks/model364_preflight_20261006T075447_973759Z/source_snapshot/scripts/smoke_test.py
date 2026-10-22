#!/usr/bin/env python3
"""Offline integrity smoke test; does not load data or score validation."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
recipe = json.loads((ROOT / "config/model364_recipe.json").read_text())
record = json.loads((ROOT / "results/outputs/model364_model361_entropy_into_360/results.json").read_text())
model360 = json.loads((ROOT / "results/outputs/model360_entropy_gated_d4_into_355/results.json").read_text())

assert recipe["model"] == 364
assert record["model"] == 364
assert abs(record["raw_best_valid_image_mean_miou_empty0"] * 100
           - recipe["recorded_validation"]["model364_raw_best_image_mean_miou_percent"]) < 1e-9
assert abs(model360["raw_best_valid_image_mean_miou_empty0"] * 100
           - recipe["recorded_validation"]["model360_image_mean_miou_percent"]) < 1e-9
assert record["raw_best_scale3_gate"] == [0.0, 0.25]
assert record["raw_best_expert_gate"] == [0.0, 0.25]
assert record["metric"] == "20-image mean of four class IoUs; empty union=0"
for relpath in (
    "experiments/model354_model349_bias_audit.py",
    "experiments/model355_model349_spatial_audit.py",
    "experiments/model360_entropy_gated_d4_into_355.py",
    "experiments/model364_model361_entropy_into_360.py",
    "experiments/model336_hrnet_best/train.py",
):
    assert (ROOT / relpath).is_file(), relpath

print("Model364 bundle integrity OK; no training or evaluation was run.")
