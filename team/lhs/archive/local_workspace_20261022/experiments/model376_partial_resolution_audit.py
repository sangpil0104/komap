#!/usr/bin/env python3
"""Resolution-stratified paired OOF deltas and raw valid/test size mix.

Uses only train masks for scores. Valid/test ZIPs are opened only to read image
dimensions, never masks or predictions. The default partial analysis uses folds
1-3 and is descriptive until all four OOF folds are complete.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from model374_train_oof_error_diagnostic import mask_to_ids
from model376_boundary_ce_paired_oof import image_iou


GROUP_RE = re.compile(r"^(A\d+)")


def zip_image_sizes(zip_path: Path) -> dict[str, tuple[int, int]]:
    sizes = {}
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist()
                   if "/images/" in name and name.endswith("_image.png")]
        for member in members:
            image_id = Path(member).name.removesuffix("_image.png")
            with Image.open(io.BytesIO(archive.read(member))) as image:
                sizes[image_id] = image.size
    return sizes


def resolution_counts(sizes: dict[str, tuple[int, int]]) -> Counter:
    return Counter(f"{width}x{height}" for width, height in sizes.values())


def score_rows(cv_data: Path, train_sizes: dict[str, tuple[int, int]],
               baseline_root: Path, candidate_root: Path,
               folds: tuple[int, ...]) -> tuple[list[dict], dict[str, int]]:
    rows = []
    fold_counts = {}
    seen = set()
    for fold in folds:
        mask_dir = cv_data / f"oof_fold{fold}" / "masks"
        masks = {p.name.removesuffix("_mask.png"): p for p in mask_dir.glob("*_mask.png")}
        base_path = baseline_root / f"fold{fold}_oof" / "valid_probabilities.npz"
        cand_path = candidate_root / f"fold{fold}_oof" / "valid_probabilities.npz"
        if not masks or not base_path.is_file() or not cand_path.is_file():
            raise FileNotFoundError(f"missing grouped OOF inputs for fold {fold}")
        with np.load(base_path, allow_pickle=False) as baseline, \
             np.load(cand_path, allow_pickle=False) as candidate:
            if set(baseline.files) != set(masks) or set(candidate.files) != set(masks):
                raise ValueError(f"fold {fold}: OOF image IDs do not exactly match masks")
            for image_id in sorted(masks):
                if image_id in seen:
                    raise ValueError(f"duplicate OOF image across folds: {image_id}")
                seen.add(image_id)
                match = GROUP_RE.match(image_id)
                if not match or image_id not in train_sizes:
                    raise ValueError(f"missing specimen group or train image dimensions: {image_id}")
                target = mask_to_ids(masks[image_id])
                base_iou = image_iou(np.asarray(baseline[image_id], dtype=np.float32), target)
                cand_iou = image_iou(np.asarray(candidate[image_id], dtype=np.float32), target)
                width, height = train_sizes[image_id]
                rows.append({
                    "image_id": image_id,
                    "group": match.group(1),
                    "fold": fold,
                    "resolution": f"{width}x{height}",
                    "delta_pp": float((np.mean(cand_iou) - np.mean(base_iou)) * 100.0),
                })
        fold_counts[str(fold)] = len(masks)
    return rows, fold_counts


def cluster_bootstrap(rows: list[dict], test_675_fraction: float,
                      valid_675_fraction: float, draws: int, seed: int) -> dict:
    groups = sorted({row["group"] for row in rows})
    by_group = {group: [r for r in rows if r["group"] == group] for group in groups}
    strata = {
        "675x480": lambda r: r["resolution"] == "675x480",
        "other_resolutions": lambda r: r["resolution"] != "675x480",
    }
    full = {}
    for name, predicate in strata.items():
        subset = [r for r in rows if predicate(r)]
        full[name] = {
            "images": len(subset),
            "groups": len({r["group"] for r in subset}),
            "mean_delta_pp": float(np.mean([r["delta_pp"] for r in subset])),
            "fold_mean_delta_pp": {
                str(fold): float(np.mean([r["delta_pp"] for r in subset if r["fold"] == fold]))
                for fold in sorted({r["fold"] for r in subset})
            },
        }

    rng = np.random.default_rng(seed)
    subgroup_draws = {name: np.empty(draws, dtype=np.float64) for name in strata}
    valid_mix = np.empty(draws, dtype=np.float64)
    test_mix = np.empty(draws, dtype=np.float64)
    interaction = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        selected = rng.integers(0, len(groups), size=len(groups))
        sample = [row for index in selected for row in by_group[groups[index]]]
        means = {}
        for name, predicate in strata.items():
            values = [row["delta_pp"] for row in sample if predicate(row)]
            if not values:
                raise RuntimeError(f"bootstrap draw {draw} has no rows in {name}")
            means[name] = float(np.mean(values))
            subgroup_draws[name][draw] = means[name]
        interaction[draw] = means["675x480"] - means["other_resolutions"]
        valid_mix[draw] = valid_675_fraction * means["675x480"] + (1 - valid_675_fraction) * means["other_resolutions"]
        test_mix[draw] = test_675_fraction * means["675x480"] + (1 - test_675_fraction) * means["other_resolutions"]

    for name in strata:
        full[name]["group_bootstrap_95ci_pp"] = np.percentile(subgroup_draws[name], [2.5, 97.5]).tolist()
    valid_mean = valid_675_fraction * full["675x480"]["mean_delta_pp"] + (1 - valid_675_fraction) * full["other_resolutions"]["mean_delta_pp"]
    test_mean = test_675_fraction * full["675x480"]["mean_delta_pp"] + (1 - test_675_fraction) * full["other_resolutions"]["mean_delta_pp"]
    return {
        "groups": len(groups),
        "subgroups": full,
        "675x480_minus_other_interaction_pp": {
            "mean": float(full["675x480"]["mean_delta_pp"] - full["other_resolutions"]["mean_delta_pp"]),
            "group_bootstrap_95ci_pp": np.percentile(interaction, [2.5, 97.5]).tolist(),
        },
        "resolution_mix_standardization": {
            "valid_675x480_fraction": valid_675_fraction,
            "test_675x480_fraction": test_675_fraction,
            "estimated_oof_delta_at_valid_mix_pp": float(valid_mean),
            "estimated_oof_delta_at_test_mix_pp": float(test_mean),
            "valid_mix_bootstrap_95ci_pp": np.percentile(valid_mix, [2.5, 97.5]).tolist(),
            "test_mix_bootstrap_95ci_pp": np.percentile(test_mix, [2.5, 97.5]).tolist(),
        },
    }


def run(cv_data: Path, train_zip: Path, valid_zip: Path, test_zip: Path,
        baseline_root: Path, candidate_root: Path, folds: tuple[int, ...],
        draws: int) -> dict:
    train_sizes = zip_image_sizes(train_zip)
    valid_sizes = zip_image_sizes(valid_zip)
    test_sizes = zip_image_sizes(test_zip)
    rows, fold_counts = score_rows(cv_data, train_sizes, baseline_root, candidate_root, folds)
    train_mix = resolution_counts(train_sizes)
    valid_mix = resolution_counts(valid_sizes)
    test_mix = resolution_counts(test_sizes)
    valid_fraction = valid_mix["675x480"] / len(valid_sizes)
    test_fraction = test_mix["675x480"] / len(test_sizes)
    complete = set(folds) == {1, 2, 3, 4}
    return {
        "scope": (
            "complete paired Model376-Model374 train OOF; official valid/test ZIPs used only for image dimensions"
            if complete else "partial paired Model376-Model374 train OOF; official valid/test ZIPs used only for image dimensions"
        ),
        "folds": list(folds),
        "images": len(rows),
        "images_by_fold": fold_counts,
        "resolution_counts": {
            "train": dict(sorted(train_mix.items())),
            "valid": dict(sorted(valid_mix.items())),
            "test": dict(sorted(test_mix.items())),
        },
        "bootstrap": {"unit": "specimen group", "draws": draws, "seed": 376},
        "paired_oof_by_resolution": cluster_bootstrap(rows, test_fraction, valid_fraction, draws, 376),
        "interpretation_guard": (
            "The resolution-stratified deltas are descriptive, not a separate adoption test. "
            "Reweighting applies the coarse 675x480-versus-other OOF effects to observed valid/test "
            "size mixes; it is not a test score estimate, and the pooled other-resolution category "
            "may hide heterogeneity."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-data", type=Path, required=True)
    parser.add_argument("--train-zip", type=Path, required=True)
    parser.add_argument("--valid-zip", type=Path, required=True)
    parser.add_argument("--test-zip", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.cv_data, args.train_zip, args.valid_zip, args.test_zip,
                 args.baseline_root, args.candidate_root, tuple(args.folds), args.draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
