#!/usr/bin/env python3
"""Paired, held-out OOF audit for the first Model322 fold (diagnostic only)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

TEAM_CODE = Path(__file__).resolve().parent / "model322_team_b4" / "KoMaP"
sys.path.insert(0, str(TEAM_CODE))

from common import list_split, load_mask
from evaluate_group_cv import image_score, probability_for, read_scores


TEAM_CLASSES = ("Al", "Al3Ni", "Primary Si", "Eutectic Si")


def specimen_group(name: str) -> str:
    match = re.match(r"(A\d+)_", name)
    if not match:
        raise ValueError(f"cannot parse specimen group from {name}")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cv-data", type=Path, required=True)
    parser.add_argument("--parent-oof", type=Path, required=True)
    parser.add_argument("--candidate-oof", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")

    train_pairs = list_split(str(args.data), "train")
    names, parent_groups, parent_zero, parent_skip, parent_ious = read_scores(
        args.parent_oof, train_pairs, convert_to_team_order=True)
    parent_by_name = {
        name: (parent_groups[i], parent_zero[i], parent_skip[i], parent_ious[i])
        for i, name in enumerate(names)
    }

    fold_pairs = list_split(str(args.cv_data), "oof_fold1")
    records = []
    with np.load(args.candidate_oof) as archive:
        for image_path, mask_path in fold_pairs:
            name = Path(image_path).name.replace("_image.png", "")
            if name not in parent_by_name:
                raise KeyError(f"{name} missing from parent OOF")
            probs = probability_for(archive, name)
            target = load_mask(mask_path)
            candidate_zero, candidate_skip, candidate_iou = image_score(probs, target)
            parent_group, base_zero, base_skip, base_iou = parent_by_name[name]
            group = specimen_group(name)
            if group != parent_group:
                raise ValueError(f"group mismatch for {name}: {group} vs {parent_group}")
            records.append({
                "image": name,
                "specimen_group": group,
                "parent_empty_zero": base_zero,
                "candidate_empty_zero": candidate_zero,
                "delta_empty_zero_pp": (candidate_zero - base_zero) * 100.0,
                "parent_nan_skip": base_skip,
                "candidate_nan_skip": candidate_skip,
                "delta_nan_skip_pp": (candidate_skip - base_skip) * 100.0,
                "parent_class_iou_empty_zero": np.nan_to_num(base_iou, nan=0.0).tolist(),
                "candidate_class_iou_empty_zero": np.nan_to_num(candidate_iou, nan=0.0).tolist(),
            })

    if len(records) != len(set(row["image"] for row in records)):
        raise ValueError("duplicate fold1 OOF image")
    with np.load(args.candidate_oof) as archive:
        if set(archive.files) != {row["image"] for row in records}:
            raise ValueError("fold1 OOF archive and held-out split differ")

    groups: dict[str, list[int]] = {}
    for i, row in enumerate(records):
        groups.setdefault(row["specimen_group"], []).append(i)
    deltas_zero = np.asarray([r["delta_empty_zero_pp"] for r in records])
    deltas_skip = np.asarray([r["delta_nan_skip_pp"] for r in records])
    rng = np.random.default_rng(3221)
    group_names = list(groups)
    draws = np.empty((10000, 2), dtype=np.float64)
    for d in range(len(draws)):
        sampled = rng.choice(group_names, size=len(group_names), replace=True)
        idx = [i for g in sampled for i in groups[str(g)]]
        draws[d, 0] = deltas_zero[idx].mean()
        draws[d, 1] = deltas_skip[idx].mean()

    output = {
        "audit": "first-fold held-out OOF only; diagnostic, not full 70-image result or validation tuning",
        "class_order": list(TEAM_CLASSES),
        "image_count": len(records),
        "specimen_group_count": len(groups),
        "parent_empty_zero_image_mean": float(np.mean([r["parent_empty_zero"] for r in records])),
        "candidate_empty_zero_image_mean": float(np.mean([r["candidate_empty_zero"] for r in records])),
        "delta_empty_zero_pp": float(deltas_zero.mean()),
        "group_bootstrap_95ci_delta_empty_zero_pp": np.quantile(
            draws[:, 0], [0.025, 0.975]).tolist(),
        "parent_nan_skip_image_mean": float(np.mean([r["parent_nan_skip"] for r in records])),
        "candidate_nan_skip_image_mean": float(np.mean([r["candidate_nan_skip"] for r in records])),
        "delta_nan_skip_pp": float(deltas_skip.mean()),
        "group_bootstrap_95ci_delta_nan_skip_pp": np.quantile(
            draws[:, 1], [0.025, 0.975]).tolist(),
        "positive_images_empty_zero": int((deltas_zero > 0).sum()),
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "records"},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
