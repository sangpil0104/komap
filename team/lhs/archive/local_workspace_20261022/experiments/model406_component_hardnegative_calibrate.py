#!/usr/bin/env python3
"""Calibrate Model406's joint auxiliary-loss scale on outer-train batches only."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
loss_v1 = importlib.import_module("model406_component_hardnegative_loss")
sys.modules["model404_component_recall_loss"] = loss_v1
calibrator = importlib.import_module("model404_component_recall_calibrate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--probe-batches", type=int, default=8)
    parser.add_argument("--probe-seed", type=int, default=1)
    args = parser.parse_args()
    if args.probe_batches < 8:
        raise ValueError("at least 8 outer-train batches per fold are preregistered")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite calibration: {args.out}")
    folds = {}
    for fold in range(1, 5):
        record = calibrator.calibrate_fold(
            args.split_root / f"seed1_fold{fold}",
            args.baseline_root / f"fold{fold}" / "last.pt",
            torch.device(args.device),
            args.probe_batches,
            args.probe_seed,
        )
        folds[str(fold)] = {
            "training_split_only": record["training_split_only"],
            "optimizer_steps": record["optimizer_steps"],
            "valid_read": record["valid_read"],
            "test_read": record["test_read"],
            "probe_batches": record["probe_batches"],
            "nonempty_gradient_batches": record["nonempty_gradient_batches"],
            "component_counts_per_batch": record["component_counts_per_batch"],
            "joint_aux_to_main_decoder_gradient_norm_ratios":
                record["aux_to_full_main_decoder_gradient_norm_ratios"],
            "median_joint_gradient_norm_ratio": record["median_gradient_norm_ratio"],
            "target_joint_auxiliary_gradient_fraction":
                record["target_auxiliary_gradient_fraction"],
            "joint_aux_weight": record["component_weight"],
        }
        print(
            f"fold={fold} joint_aux_weight={record['component_weight']:.12g} "
            f"median_grad_ratio={record['median_gradient_norm_ratio']:.6g}",
            flush=True,
        )
    if any(
        not item["training_split_only"] or item["optimizer_steps"] != 0
        or item["valid_read"] or item["test_read"]
        or item["nonempty_gradient_batches"] < 6
        or not 1e-5 <= item["joint_aux_weight"] <= 1.0
        for item in folds.values()
    ):
        raise ValueError("outer-train calibration failed its locked safety checks")
    result = {
        "experiment": "Model406 component-recall plus hard-negative island loss calibration",
        "objective": (
            "component_recall_loss + hard_negative_island_loss; each target-class/crop "
            "uses top 0.1% negative probabilities outside a 1-pixel GT halo"
        ),
        "metric_used_for_calibration":
            "decoder-feature gradient norm only; no mIoU or held-out images",
        "formula":
            "joint_aux_weight = 0.10 / median(norm(grad(joint_aux))/norm(grad(CE+Dice+boundary)))",
        "scope": "four fixed outer-train folds only",
        "valid_read": False,
        "test_read": False,
        "optimizer_steps": 0,
        "folds": folds,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
