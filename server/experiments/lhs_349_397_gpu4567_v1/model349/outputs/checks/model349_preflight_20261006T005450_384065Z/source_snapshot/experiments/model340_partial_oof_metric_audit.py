#!/usr/bin/env python3
"""Exact empty-union=0 paired audit of completed Model329 OOF folds.

This diagnostic compares decoded prediction masks for Models 329 and 322. It
does not read validation labels, train a model, or run model inference.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def specimen_group(name: str) -> str:
    match = re.match(r"(A\d+)_", name)
    if not match:
        raise ValueError(f"cannot parse specimen group: {name}")
    return match.group(1)


def confusion_iou(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: prediction={pred.shape}, target={target.shape}")
    cm = np.bincount(target.astype(np.int64).ravel() * 4 + pred.astype(np.int64).ravel(),
                     minlength=16).reshape(4, 4)
    intersection = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - intersection
    # Official convention: empty union has IoU 0.
    return np.divide(intersection, union, out=np.zeros(4, dtype=np.float64), where=union > 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--folds", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1329)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    code_dir = args.root / "model326_hybrid_fullstats/KoMaP"
    sys.path.insert(0, str(code_dir))
    from common import list_split, load_mask  # pylint: disable=import-outside-toplevel

    out_root = args.root / "outputs"
    records = []
    fold_summaries = []
    for fold in args.folds:
        split_pairs = list_split(str(args.root / "data_model322_group_cv"),
                                 f"oof_fold{fold}")
        parent_dir = out_root / f"model322_team_b4_group_oof/fold{fold}_oof"
        candidate_dir = out_root / f"model329_fullstats_width4000_uniform_group_oof/fold{fold}_oof"
        fold_records = []
        for image_path, mask_path in split_pairs:
            name = Path(image_path).name.replace("_image.png", "")
            target = load_mask(mask_path)
            parent_pred = load_mask(str(parent_dir / f"{name}_mask.png"))
            candidate_pred = load_mask(str(candidate_dir / f"{name}_mask.png"))
            parent_iou = confusion_iou(parent_pred, target)
            candidate_iou = confusion_iou(candidate_pred, target)
            row = {
                "image": name,
                "specimen_group": specimen_group(name),
                "model322_miou_empty0": float(parent_iou.mean() * 100.0),
                "model329_miou_empty0": float(candidate_iou.mean() * 100.0),
                "delta_pp_empty0": float((candidate_iou.mean() - parent_iou.mean()) * 100.0),
                "class_delta_pp_empty0": ((candidate_iou - parent_iou) * 100.0).tolist(),
            }
            fold_records.append(row)
            records.append(row)
        fold_delta = np.asarray([row["delta_pp_empty0"] for row in fold_records])
        fold_summaries.append({
            "fold": fold,
            "image_count": len(fold_records),
            "model322_mean_miou_empty0": float(np.mean(
                [row["model322_miou_empty0"] for row in fold_records])),
            "model329_mean_miou_empty0": float(np.mean(
                [row["model329_miou_empty0"] for row in fold_records])),
            "paired_delta_pp_empty0": float(fold_delta.mean()),
            "positive_images": int((fold_delta > 0).sum()),
        })

    names = [row["image"] for row in records]
    if len(names) != len(set(names)):
        raise ValueError("duplicate image IDs across audited folds")
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(records):
        groups.setdefault(row["specimen_group"], []).append(index)
    delta = np.asarray([row["delta_pp_empty0"] for row in records], dtype=np.float64)
    class_delta = np.asarray([row["class_delta_pp_empty0"] for row in records],
                             dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    bootstrap = np.empty((args.bootstrap, 5), dtype=np.float64)
    group_names = list(groups)
    for draw in range(args.bootstrap):
        sampled = rng.choice(group_names, size=len(group_names), replace=True)
        indices = [index for group in sampled for index in groups[str(group)]]
        bootstrap[draw, 0] = delta[indices].mean()
        bootstrap[draw, 1:] = class_delta[indices].mean(axis=0)

    class_names = ("Al", "Al3Ni", "Primary Si", "Eutectic Si")
    summary = {
        "model": 340,
        "scope": "interim train-only grouped OOF; only the listed completed folds",
        "metric": "image-wise mean of four class IoUs; empty union=0",
        "class_order": list(class_names),
        "folds": fold_summaries,
        "image_count": len(records),
        "total_train_images": 70,
        "specimen_group_count": len(groups),
        "model322_mean_miou_empty0": float(np.mean([
            row["model322_miou_empty0"] for row in records])),
        "model329_mean_miou_empty0": float(np.mean([
            row["model329_miou_empty0"] for row in records])),
        "paired_delta_pp_empty0": float(delta.mean()),
        "positive_images": int((delta > 0).sum()),
        "group_bootstrap_95ci_delta_pp_empty0": [
            float(value) for value in np.quantile(bootstrap[:, 0], [0.025, 0.975])],
        "class_delta_pp_empty0": {
            name: {
                "mean": float(class_delta[:, index].mean()),
                "group_bootstrap_95ci": [float(value) for value in np.quantile(
                    bootstrap[:, index + 1], [0.025, 0.975])],
                "positive_images": int((class_delta[:, index] > 0).sum()),
            }
            for index, name in enumerate(class_names)
        },
        "images": records,
    }
    output = args.output or out_root / "model340_partial_oof_audit/folds1_2.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary["output"] = str(output)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "image_count", "specimen_group_count", "model322_mean_miou_empty0",
        "model329_mean_miou_empty0", "paired_delta_pp_empty0", "positive_images",
        "group_bootstrap_95ci_delta_pp_empty0", "class_delta_pp_empty0", "folds", "output")},
        ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
