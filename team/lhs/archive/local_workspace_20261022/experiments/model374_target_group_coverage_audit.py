#!/usr/bin/env python3
"""Label-free target image coverage vs fixed Model374 OOF distance ranges."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from valid_test_similarity_audit import FEATURES, image_features


GROUP_RE = re.compile(r"^(A\d+)_")
INTENSITY_FEATURES = set(range(7))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_images(zip_path: Path, temp_dir: Path) -> tuple[list[dict], dict[str, str]]:
    temp_dir.mkdir(parents=True, exist_ok=True)
    rows, hashes = [], {}
    with zipfile.ZipFile(zip_path) as archive:
        members = sorted(name for name in archive.namelist()
                         if "/images/" in name and name.endswith("_image.png"))
        for member in members:
            payload = archive.read(member)
            path = temp_dir / Path(member).name
            path.write_bytes(payload)
            row = image_features(path)
            match = GROUP_RE.match(row["name"])
            if not match:
                raise ValueError(f"unexpected image ID: {row['name']}")
            row["group"] = match.group(1)
            row["features"] = np.asarray(row["features"], dtype=np.float64)
            rows.append(row)
            hashes[row["name"]] = hashlib.sha256(payload).hexdigest()
    if not rows:
        raise ValueError(f"no images found in {zip_path}")
    return rows, hashes


def grouped_features(rows: list[dict]) -> dict[str, dict]:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)
    return {
        group: {
            "features": np.mean([row["features"] for row in group_rows], axis=0),
            "images": [row["name"] for row in group_rows],
        }
        for group, group_rows in sorted(by_group.items())
    }


def scaled_train_and_queries(train_groups: dict[str, dict], query_groups: dict[str, dict]):
    names = sorted(train_groups)
    train_x = np.asarray([train_groups[group]["features"] for group in names], dtype=np.float64)
    query_names = sorted(query_groups)
    query_x = np.asarray([query_groups[group]["features"] for group in query_names], dtype=np.float64)
    center = np.median(train_x, axis=0)
    q25, q75 = np.percentile(train_x, [25, 75], axis=0)
    scale = np.maximum((q75 - q25) / 1.349, np.std(train_x, axis=0) * 0.1)
    scale = np.maximum(scale, 1e-6)
    weights = np.asarray([
        1 / np.sqrt(7) if i in INTENSITY_FEATURES else 1 / np.sqrt(8)
        for i in range(len(FEATURES))
    ])
    train_z = (train_x - center) / scale * weights
    query_z = (query_x - center) / scale * weights
    return names, query_names, train_z, query_z


def nearest(query: np.ndarray, train: np.ndarray, train_names: list[str]) -> tuple[str, float]:
    distances = cdist(query[None, :], train[None, :] if train.ndim == 1 else train)[0]
    index = int(np.argmin(distances))
    return train_names[index], float(distances[index])


def analyze(train_rows: list[dict], targets: dict[str, list[dict]], coverage: dict) -> dict:
    train_groups = grouped_features(train_rows)
    target_groups = {name: grouped_features(rows) for name, rows in targets.items()}
    fold_assignment = {}
    oof_distances = {}
    for fold_key, data in coverage["folds"].items():
        fold = int(fold_key)
        oof_distances[fold] = np.asarray([row["appearance_distance"] for row in data["rows"]], dtype=np.float64)
        for row in data["rows"]:
            if row["group"] in fold_assignment and fold_assignment[row["group"]] != fold:
                raise ValueError(f"group crosses folds: {row['group']}")
            fold_assignment[row["group"]] = fold
    if set(fold_assignment) != set(train_groups):
        raise ValueError("train image groups and fixed OOF fold groups do not match")

    per_split = {}
    for split, query_groups in target_groups.items():
        split_rows = {}
        for fold in (1, 2, 3, 4):
            ref_groups = {group: value for group, value in train_groups.items()
                          if fold_assignment[group] != fold}
            ref_names, query_names, ref_z, query_z = scaled_train_and_queries(ref_groups, query_groups)
            baseline = np.sort(oof_distances[fold])
            for query_index, group in enumerate(query_names):
                distances = cdist(query_z[query_index:query_index + 1], ref_z)[0]
                nearest_index = int(np.argmin(distances))
                nearest_group = ref_names[nearest_index]
                distance = float(distances[nearest_index])
                percentile = float(np.searchsorted(baseline, distance, side="right") / len(baseline))

                unseen_ref_groups = {name: value for name, value in ref_groups.items() if name != group}
                if not unseen_ref_groups:
                    raise ValueError("no reference groups after excluding same-prefix group")
                unseen_names, unseen_query_names, unseen_ref_z, unseen_query_z = scaled_train_and_queries(
                    unseen_ref_groups, {group: query_groups[group]}
                )
                if unseen_query_names != [group]:
                    raise ValueError("unexpected query ordering in leave-prefix-out scaling")
                unseen_distances = cdist(unseen_query_z, unseen_ref_z)[0]
                unseen_index = int(np.argmin(unseen_distances))
                unseen_group = unseen_names[unseen_index]
                unseen_distance = float(unseen_distances[unseen_index])
                unseen_percentile = float(np.searchsorted(baseline, unseen_distance, side="right") / len(baseline))

                split_rows.setdefault(group, {
                    "group": group,
                    "images": query_groups[group]["images"],
                    "train_prefix_seen": group in train_groups,
                    "fold_comparisons": [],
                })["fold_comparisons"].append({
                    "outer_fold": fold,
                    "nearest_train_group": nearest_group,
                    "nearest_distance": distance,
                    "percentile_vs_fold_oof_group_distances": percentile,
                    "nearest_train_group_excluding_same_prefix": unseen_group,
                    "distance_excluding_same_prefix": unseen_distance,
                    "percentile_excluding_same_prefix_vs_fold_oof": unseen_percentile,
                })

        for row in split_rows.values():
            row["median_oof_distance_percentile"] = float(np.median([
                item["percentile_vs_fold_oof_group_distances"] for item in row["fold_comparisons"]
            ]))
            row["median_unseen_prefix_distance_percentile"] = float(np.median([
                item["percentile_excluding_same_prefix_vs_fold_oof"] for item in row["fold_comparisons"]
            ]))
        values = list(split_rows.values())
        per_split[split] = {
            "images": sum(len(row["images"]) for row in values),
            "specimen_groups": len(values),
            "groups_sharing_prefix_with_train": sum(row["train_prefix_seen"] for row in values),
            "median_fold_oof_distance_percentile_including_seen_prefix":
                float(np.median([row["median_oof_distance_percentile"] for row in values])),
            "median_fold_oof_distance_percentile_excluding_seen_prefix":
                float(np.median([row["median_unseen_prefix_distance_percentile"] for row in values])),
            "per_group": values,
        }

    return {
        "scope": "raw train/valid/test images only; no masks, predictions, or target scores read",
        "distance": "fixed 15D intensity/texture feature distance; robust scaling fit on each fold training-group pool",
        "reference": "for each fixed outer fold, target group nearest-neighbor distance is percentile-ranked against that fold's heldout Model374 OOF group-distance distribution",
        "same_prefix_sensitivity": "reports actual reference pool including an A-prefix match and a second distance excluding that prefix; the latter is closer to fully unseen-group CV",
        "splits": per_split,
        "interpretation_guard": (
            "Input-domain coverage only; the percentile is not a predicted mIoU. Valid/test group labels, masks, "
            "OOF error values, and target predictions are not used. The benchmark distribution is small and post-hoc."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-zip", type=Path, required=True)
    parser.add_argument("--valid-zip", type=Path, required=True)
    parser.add_argument("--test-zip", type=Path, required=True)
    parser.add_argument("--coverage-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="model374-target-coverage-") as temp:
        temp_path = Path(temp)
        train_rows, train_hashes = load_images(args.train_zip, temp_path / "train")
        valid_rows, valid_hashes = load_images(args.valid_zip, temp_path / "valid")
        test_rows, test_hashes = load_images(args.test_zip, temp_path / "test")
    coverage = json.loads(args.coverage_json.read_text(encoding="utf-8"))
    result = analyze(train_rows, {"valid": valid_rows, "test": test_rows}, coverage)
    result["source_sha256"] = {
        "train_zip": sha256(args.train_zip),
        "valid_zip": sha256(args.valid_zip),
        "test_zip": sha256(args.test_zip),
        "coverage_json_read_for_distance_columns_only": sha256(args.coverage_json),
        "train_image_hashes": train_hashes,
        "valid_image_hashes": valid_hashes,
        "test_image_hashes": test_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({name: {
        key: value for key, value in split.items() if key != "per_group"
    } for name, split in result["splits"].items()}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
