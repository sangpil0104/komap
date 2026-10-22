#!/usr/bin/env python3
"""Audit phase-pair component homogenization on Model329 grouped train OOF only.

This is a GT-informed feasibility diagnostic, not deployable postprocessing and
not an upper bound for arbitrary pixel-level corrections. It never opens the
official validation split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

PALETTE = np.array(
    [[153, 127, 76], [25, 76, 153], [76, 178, 76], [204, 204, 204]],
    dtype=np.uint8,
)
CLASS_ORDER = ("Al", "Al3Ni", "Primary_Si", "Eutectic_Si")
STRUCTURE_8 = np.ones((3, 3), dtype=bool)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mask(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    result = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        result[np.all(rgb == color, axis=-1)] = class_id
    if np.any(result == 255):
        raise ValueError(f"unknown mask color in {path}")
    return result


def class_iou(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    confusion = np.bincount(
        target.astype(np.int64).ravel() * 4 + prediction.astype(np.int64).ravel(),
        minlength=16,
    ).reshape(4, 4)
    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(axis=0) + confusion.sum(axis=1) - intersection
    return np.divide(intersection, union, out=np.zeros(4), where=union > 0)


def group_id(image_id: str) -> str:
    match = re.match(r"(A\d+)_", image_id)
    if match is None:
        raise ValueError(f"cannot parse specimen group from {image_id}")
    return match.group(1)


def homogenize_using_gt(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Assign each predicted Al3Ni/Eutectic component its GT pair majority."""
    pair = (prediction == 1) | (prediction == 3)
    labels, count = ndimage.label(pair, structure=STRUCTURE_8)
    gt_al3ni = np.bincount(labels[pair], weights=(target[pair] == 1), minlength=count + 1)
    gt_eutectic = np.bincount(labels[pair], weights=(target[pair] == 3), minlength=count + 1)
    component_ids = np.arange(1, count + 1)
    choose_eutectic = gt_eutectic[component_ids] > gt_al3ni[component_ids]
    component_class = np.ones(count + 1, dtype=np.uint8)  # ties go to Al3Ni
    component_class[component_ids[choose_eutectic]] = 3
    result = prediction.copy()
    result[pair] = component_class[labels[pair]]
    return result


def grouped_bootstrap(values: np.ndarray, groups: np.ndarray, draws: int,
                      seed: int) -> np.ndarray:
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([by_group[group] for group in selected])
        samples[index] = values[rows].mean()
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-data", type=Path, required=True)
    parser.add_argument("--model329-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=37_470)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    if args.bootstrap < 100 or not args.folds:
        raise ValueError("need at least 100 bootstrap draws and one fold")
    if len(set(args.folds)) != len(args.folds) or any(fold not in (1, 2, 3, 4)
                                                       for fold in args.folds):
        raise ValueError(f"invalid folds: {args.folds}")

    records = []
    input_hashes = {}
    for fold in args.folds:
        mask_dir = args.cv_data / f"oof_fold{fold}" / "masks"
        archive_path = args.model329_root / f"fold{fold}_oof" / "probabilities.npz"
        mask_paths = sorted(mask_dir.glob("*_mask.png"))
        if not mask_paths or not archive_path.is_file():
            raise FileNotFoundError(f"missing grouped-OOF fold {fold} inputs")
        input_hashes[str(fold)] = sha256(archive_path)
        with np.load(archive_path) as archive:
            expected = {path.name.removesuffix("_mask.png") for path in mask_paths}
            if set(archive.files) != expected:
                raise ValueError(
                    f"fold {fold} probability keys differ from held-out masks: "
                    f"maps={len(archive.files)} masks={len(expected)}")
            for mask_path in mask_paths:
                image_id = mask_path.name.removesuffix("_mask.png")
                target = load_mask(mask_path)
                probability = np.asarray(archive[image_id], dtype=np.float32)
                if (probability.shape != (4, *target.shape)
                        or not np.isfinite(probability).all()):
                    raise ValueError(f"invalid probability tensor for {image_id}: "
                                     f"{probability.shape}")
                prediction = probability.argmax(axis=0).astype(np.uint8)
                diagnostic = homogenize_using_gt(prediction, target)
                records.append({
                    "fold": fold,
                    "image_id": image_id,
                    "group_id": group_id(image_id),
                    "base_iou": class_iou(prediction, target).tolist(),
                    "gt_component_majority_iou": class_iou(diagnostic, target).tolist(),
                })

    base = np.asarray([row["base_iou"] for row in records], dtype=np.float64)
    diagnostic = np.asarray([row["gt_component_majority_iou"] for row in records],
                            dtype=np.float64)
    delta = diagnostic.mean(axis=1) - base.mean(axis=1)
    groups = np.asarray([row["group_id"] for row in records])
    draws = grouped_bootstrap(delta, groups, args.bootstrap, args.seed)
    per_fold = {}
    for fold in args.folds:
        indexes = np.asarray([row["fold"] == fold for row in records])
        per_fold[str(fold)] = {
            "images": int(indexes.sum()),
            "base_image_mean_miou": float(base[indexes].mean()),
            "gt_component_majority_image_mean_miou": float(diagnostic[indexes].mean()),
            "delta_pp": float(delta[indexes].mean() * 100),
            "positive_images": int((delta[indexes] > 0).sum()),
        }
    result = {
        "diagnostic": "GT-informed predicted Al3Ni/Eutectic component majority; not deployable",
        "metric": "image-wise mean of four class IoUs; empty union=0",
        "class_order": list(CLASS_ORDER),
        "folds": args.folds,
        "images": len(records),
        "specimen_groups": int(np.unique(groups).size),
        "base_image_mean_miou": float(base.mean()),
        "gt_component_majority_image_mean_miou": float(diagnostic.mean()),
        "delta_pp": float(delta.mean() * 100),
        "group_bootstrap_95_ci_delta_pp": [
            float(np.quantile(draws, 0.025) * 100),
            float(np.quantile(draws, 0.975) * 100),
        ],
        "positive_images": int((delta > 0).sum()),
        "base_mean_per_image_class_iou": base.mean(axis=0).tolist(),
        "diagnostic_mean_per_image_class_iou": diagnostic.mean(axis=0).tolist(),
        "per_fold": per_fold,
        "model329_probability_archive_sha256": input_hashes,
        "warning": (
            "The correction uses each held-out ground-truth mask and is only a "
            "feasibility diagnostic for whole-component phase homogenization. "
            "It is neither a deployable score nor an upper bound for other rules."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
