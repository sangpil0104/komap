#!/usr/bin/env python3
"""Exact paired empty-union=0 audit for completed Model333 OOF folds."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import re
from pathlib import Path

import numpy as np


CLASS_ORDER = ("Al", "Al3Ni", "Primary Si", "Eutectic Si")


def image_group(name: str) -> str:
    match = re.match(r"(A\d+)_", Path(name).name)
    if not match:
        raise ValueError(f"cannot parse specimen group from {name}")
    return match.group(1)


def class_iou(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    matrix = np.bincount(target.astype(np.int64).ravel() * 4 +
                         prediction.astype(np.int64).ravel(), minlength=16).reshape(4, 4)
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(axis=0) + matrix.sum(axis=1) - intersection
    return np.divide(intersection, union, out=np.zeros(4, dtype=np.float64), where=union > 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--folds", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--arms", choices=("model322_parent", "control", "lovasz"),
                        nargs="+", default=("model322_parent", "control", "lovasz"))
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.folds or any(fold not in (1, 2, 3, 4) for fold in args.folds):
        raise ValueError("folds must be selected from 1,2,3,4")
    if "model322_parent" not in args.arms or len(args.arms) < 2:
        raise ValueError("include model322_parent and at least one Model333 arm")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    code_dir = args.root / "model326_hybrid_fullstats/KoMaP"
    import sys
    sys.path.insert(0, str(code_dir))
    from common import list_split, load_mask  # pylint: disable=import-outside-toplevel

    data = args.root / "data_model322_group_cv"
    rows = []
    for fold in args.folds:
        pairs = list_split(str(data), f"oof_fold{fold}")
        prob_paths = {}
        for arm in args.arms:
            if arm == "model322_parent":
                prob_paths[arm] = (args.root / "outputs/model322_team_b4_group_oof" /
                                   f"fold{fold}_oof/probabilities.npz")
            else:
                prob_paths[arm] = (
                    args.root / "outputs/model333_b4_imagewise_lovasz_group_oof" /
                    arm / f"fold{fold}_oof/probabilities.npz")
        for path in prob_paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        with ExitStack() as stack:
            archives = {arm: stack.enter_context(np.load(path))
                        for arm, path in prob_paths.items()}
            for image_path, mask_path in pairs:
                name = Path(image_path).name.replace("_image.png", "")
                target = load_mask(mask_path)
                row = {"image": name, "specimen_group": image_group(name)}
                for arm, archive in archives.items():
                    key = name if name in archive else f"{name}_image"
                    if key not in archive:
                        raise KeyError(f"missing {name} in {prob_paths[arm]}")
                    value = archive[key].astype(np.float32)
                    if value.shape != (4, *target.shape):
                        raise ValueError(f"unexpected {arm} shape for {name}: {value.shape}")
                    class_values = class_iou(value.argmax(axis=0), target)
                    row[f"{arm}_miou_empty0"] = float(class_values.mean() * 100.0)
                    row[f"{arm}_class_iou_empty0"] = (class_values * 100.0).tolist()
                rows.append(row)

    if not rows:
        raise ValueError("no OOF predictions were scored")
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["specimen_group"], []).append(index)
    all_comparisons = {
        "control_minus_model322_parent": ("model322_parent", "control"),
        "lovasz_minus_model322_parent": ("model322_parent", "lovasz"),
        "lovasz_minus_control": ("control", "lovasz"),
    }
    comparisons = {
        name: pair for name, pair in all_comparisons.items()
        if set(pair).issubset(args.arms)
    }
    deltas = {
        name: np.asarray([
            row[f"{right}_miou_empty0"] - row[f"{left}_miou_empty0"]
            for row in rows])
        for name, (left, right) in comparisons.items()
    }
    rng = np.random.default_rng(333)
    group_names = list(groups)
    bootstrap = {name: np.empty(args.bootstrap, dtype=np.float64) for name in comparisons}
    for draw in range(args.bootstrap):
        sampled = rng.choice(group_names, size=len(group_names), replace=True)
        indices = [index for group in sampled for index in groups[str(group)]]
        for name, values in deltas.items():
            bootstrap[name][draw] = values[indices].mean()

    def mean_class(key: str) -> list[float]:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        return values.mean(axis=0).tolist()

    paired_comparisons = {}
    for name, values in deltas.items():
        paired_comparisons[name] = {
            "mean_delta_pp": float(values.mean()),
            "positive_images": int((values > 0).sum()),
            "group_bootstrap_95ci_delta_pp": [
                float(value) for value in np.quantile(bootstrap[name], [0.025, 0.975])],
        }
    all_folds_complete = set(args.folds) == {1, 2, 3, 4}
    if all_folds_complete and set(args.arms) == {
            "model322_parent", "control", "lovasz"}:
        status = "complete four-fold paired train OOF audit; validation evaluation is separate"
    elif all_folds_complete and set(args.arms) == {"model322_parent", "control"}:
        status = ("complete Model322-parent versus CE+Dice-control train OOF audit; "
                 "paired Lovasz comparison is separate")
    else:
        status = "partial OOF audit; remaining requested folds/arms are not complete"
    result = {
        "model": 333,
        "status": status,
        "metric": "image-wise four-class mIoU; empty union=0",
        "class_order": list(CLASS_ORDER),
        "arms": list(args.arms),
        "completed_folds": list(args.folds),
        "image_count": len(rows),
        "specimen_group_count": len(groups),
        "arm_image_mean_miou_empty0": {
            arm: float(np.mean([row[f"{arm}_miou_empty0"] for row in rows]))
            for arm in args.arms
        },
        "paired_comparisons": paired_comparisons,
        "arm_class_mean_iou_empty0": {
            arm: mean_class(f"{arm}_class_iou_empty0") for arm in args.arms
        },
        "images": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "images"},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
