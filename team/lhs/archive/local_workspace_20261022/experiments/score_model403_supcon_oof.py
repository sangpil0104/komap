#!/usr/bin/env python3
"""Paired train-only OOF score for Model403 against Model392 seed 1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "model397_reproduction_bundle"
sys.path.insert(0, str(BUNDLE / "scripts"))
from score_model397_oof import (  # noqa: E402
    CLASS_NAMES,
    FOLD_FILE,
    bootstrap_indices,
    class_ious,
    compare,
    group_and_step,
    summarize,
)
from write_oof_manifest import sha256  # noqa: E402


PALETTE = np.asarray(
    [(153, 127, 76), (25, 76, 153), (76, 178, 76), (204, 204, 204)], dtype=np.uint8
)


def load_probability_archive(path: Path, expected_ids: set[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_ids:
            raise ValueError(
                f"OOF inventory mismatch at {path}: expected {len(expected_ids)}, got {len(archive.files)}"
            )
        return {image_id: np.asarray(archive[image_id], dtype=np.float32) for image_id in archive.files}


def score(data_root: Path, output_root: Path, baseline_root: Path) -> dict:
    data_root = data_root.resolve()
    output_root = output_root.resolve()
    baseline_root = baseline_root.resolve()
    images = sorted((data_root / "train/images").glob("*_image.png"))
    if len(images) != 70:
        raise ValueError(f"expected 70 train images, found {len(images)}")
    ids = [path.name.removesuffix("_image.png") for path in images]
    info = {image_id: group_and_step(image_id) for image_id in ids}
    if len({group for group, _ in info.values()}) != 59:
        raise ValueError("expected 59 train specimen groups")

    fold_cfg = json.loads(FOLD_FILE.read_text(encoding="utf-8"))
    fold_for_group: dict[str, int] = {}
    for fold_entry in fold_cfg["folds"]:
        fold = int(fold_entry["fold"])
        for group in fold_entry["heldout_groups"]:
            if group in fold_for_group:
                raise ValueError(f"group appears in multiple folds: {group}")
            fold_for_group[group] = fold
    if set(fold_for_group) != {group for group, _ in info.values()}:
        raise ValueError("fixed fold file does not exactly cover current train groups")

    candidate_rows: list[np.ndarray] = []
    baseline_rows: list[np.ndarray] = []
    rows: list[dict] = []
    for fold in range(1, 5):
        expected = {image_id for image_id in ids if fold_for_group[info[image_id][0]] == fold}
        candidate_path = output_root / f"fold{fold}_oof/valid_probabilities.npz"
        baseline_path = baseline_root / f"seed1/fold{fold}_oof/valid_probabilities.npz"
        candidate = load_probability_archive(candidate_path, expected)
        baseline = load_probability_archive(baseline_path, expected)
        for image_id in sorted(expected):
            mask_path = data_root / "train/masks" / f"{image_id}_mask.png"
            rgb = np.asarray(Image.open(mask_path).convert("RGB"), dtype=np.uint8)
            truth = np.full(rgb.shape[:2], 255, dtype=np.uint8)
            for class_id, color in enumerate(PALETTE):
                truth[np.all(rgb == color, axis=2)] = class_id
            if np.any(truth == 255):
                raise ValueError(f"unknown mask color: {mask_path}")
            candidate_iou = class_ious(candidate[image_id], truth)
            baseline_iou = class_ious(baseline[image_id], truth)
            group, step = info[image_id]
            rows.append({"fold": fold, "image_id": image_id, "group": group, "step": step})
            candidate_rows.append(candidate_iou)
            baseline_rows.append(baseline_iou)

    if len(rows) != 70 or len({row["image_id"] for row in rows}) != 70:
        raise ValueError("candidate did not score each of the 70 train images exactly once")
    candidate_scores = np.asarray(candidate_rows, dtype=np.float64)
    baseline_scores = np.asarray(baseline_rows, dtype=np.float64)
    samples = bootstrap_indices(rows, draws=20_000, seed=403)
    comparison = compare(candidate_scores, baseline_scores, rows, samples)
    checks = {
        "gain_at_least_0_20pp": comparison["delta_image_mean_pp"] >= 0.20,
        "paired_group_ci_lower_gt_zero": comparison["paired_group_bootstrap_95ci_pp"][0] > 0.0,
        "positive_at_least_3_of_4_folds": comparison["positive_folds"] >= 3,
        "eutectic_nonnegative": comparison["class_delta_pp"]["Eutectic Si"] >= 0.0,
        "step4_delta_at_least_minus_0_30pp": (
            comparison["step4_delta_pp"] is not None and comparison["step4_delta_pp"] >= -0.30
        ),
    }
    return {
        "experiment": "Model403 class-balanced local supervised contrastive auxiliary loss",
        "metric": "image-wise mean of four class IoUs; empty-union=0",
        "class_order": CLASS_NAMES,
        "images": len(rows),
        "specimen_groups": len({row["group"] for row in rows}),
        "folds": 4,
        "baseline": "Model392 seed1, same fixed specimen-group folds",
        "candidate": "Model392 recipe + 0.0062 local supervised contrastive loss",
        "baseline_scores": summarize(baseline_scores),
        "candidate_scores": summarize(candidate_scores),
        "paired_comparison": comparison,
        "preregistered_gate": {"checks": checks, "pass": all(checks.values())},
        "official_valid_read": False,
        "test_read": False,
        "fold_config_sha256": sha256(FOLD_FILE),
        "interpretation": "Train-only group-OOF development evidence; not an official valid/test score.",
        "per_image": [
            {**row, "candidate_class_iou": candidate_scores[index].tolist(),
             "baseline_class_iou": baseline_scores[index].tolist()}
            for index, row in enumerate(rows)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = score(args.data_root, args.output_root, args.baseline_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "result": str(args.out),
        "baseline_oof_percent": result["baseline_scores"]["image_mean_miou_percent"],
        "candidate_oof_percent": result["candidate_scores"]["image_mean_miou_percent"],
        "delta_pp": result["paired_comparison"]["delta_image_mean_pp"],
        "ci95_pp": result["paired_comparison"]["paired_group_bootstrap_95ci_pp"],
        "gate_pass": result["preregistered_gate"]["pass"],
        "official_valid_read": False,
        "test_read": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
