#!/usr/bin/env python3
"""Group-aware, label-free valid/test image-domain comparison.

Images are read from the official ZIPs. Masks and test scores are never read.
Bootstrap and permutation units are specimen-prefix groups, not individual
images. The overlapping A-prefix group is also excluded in a sensitivity test.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from valid_test_similarity_audit import FEATURES, image_features, standardized


def zip_images(zip_path: Path, folder: Path) -> tuple[list[dict], dict[str, str]]:
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    digests = {}
    import hashlib

    with zipfile.ZipFile(zip_path) as archive:
        members = sorted(name for name in archive.namelist()
                         if "/images/" in name and name.endswith("_image.png"))
        for member in members:
            payload = archive.read(member)
            target = folder / Path(member).name
            target.write_bytes(payload)
            row = image_features(target)
            rows.append(row)
            digests[row["name"]] = hashlib.sha256(payload).hexdigest()
    return rows, digests


def resolution_counts(rows: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(f"{row['width']}x{row['height']}" for row in rows).items()))


def step_counts(rows: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(row["step"] for row in rows).items()))


def group_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    result: dict[str, list[list[float]]] = defaultdict(list)
    for row in rows:
        result[row["group"]].append(row["features"])
    return {group: np.asarray(values, dtype=np.float64) for group, values in result.items()}


def clustered_bootstrap_difference(valid_rows: list[dict], test_rows: list[dict],
                                   draws: int, seed: int) -> dict:
    valid_groups = group_arrays(valid_rows)
    test_groups = group_arrays(test_rows)
    valid_names = sorted(valid_groups)
    test_names = sorted(test_groups)
    observed = (np.mean([row["features"] for row in test_rows], axis=0)
                - np.mean([row["features"] for row in valid_rows], axis=0))
    rng = np.random.default_rng(seed)
    boot = np.empty((draws, len(FEATURES)), dtype=np.float64)
    for draw in range(draws):
        selected_valid = rng.integers(0, len(valid_names), size=len(valid_names))
        selected_test = rng.integers(0, len(test_names), size=len(test_names))
        valid_sample = np.concatenate([valid_groups[valid_names[i]] for i in selected_valid])
        test_sample = np.concatenate([test_groups[test_names[i]] for i in selected_test])
        boot[draw] = test_sample.mean(axis=0) - valid_sample.mean(axis=0)
    return {
        feature: {
            "test_minus_valid": float(observed[i]),
            "group_bootstrap_95ci": np.percentile(boot[:, i], [2.5, 97.5]).tolist(),
        }
        for i, feature in enumerate(FEATURES)
    }


def group_disjoint_centroid_permutation(valid_rows: list[dict], test_rows: list[dict],
                                        train_rows: list[dict], draws: int,
                                        seed: int) -> dict:
    valid_groups = group_arrays(valid_rows)
    test_groups = group_arrays(test_rows)
    overlap = set(valid_groups) & set(test_groups)
    valid_groups = {key: value for key, value in valid_groups.items() if key not in overlap}
    test_groups = {key: value for key, value in test_groups.items() if key not in overlap}
    if not valid_groups or not test_groups:
        raise ValueError("no group-disjoint valid/test data remain")

    all_rows = {"train": train_rows, "valid": valid_rows, "test": test_rows}
    z = standardized(all_rows)
    valid_names, test_names = sorted(valid_groups), sorted(test_groups)
    valid_by_group: dict[str, list[np.ndarray]] = defaultdict(list)
    test_by_group: dict[str, list[np.ndarray]] = defaultdict(list)
    for i, row in enumerate(valid_rows):
        if row["group"] not in overlap:
            valid_by_group[row["group"]].append(z["valid"][i])
    for i, row in enumerate(test_rows):
        if row["group"] not in overlap:
            test_by_group[row["group"]].append(z["test"][i])
    valid_z = np.asarray([np.mean(valid_by_group[group], axis=0) for group in valid_names])
    test_z = np.asarray([np.mean(test_by_group[group], axis=0) for group in test_names])
    pooled = np.concatenate((valid_z, test_z))
    valid_n = len(valid_z)
    observed = float(np.linalg.norm(valid_z.mean(axis=0) - test_z.mean(axis=0)))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(draws):
        order = rng.permutation(len(pooled))
        distance = np.linalg.norm(pooled[order[:valid_n]].mean(axis=0)
                                  - pooled[order[valid_n:]].mean(axis=0))
        exceed += distance >= observed
    return {
        "overlap_groups_removed_from_both_splits": sorted(overlap),
        "valid_disjoint_groups": len(valid_z),
        "test_disjoint_groups": len(test_z),
        "robust_scaled_centroid_distance": observed,
        "group_permutation_p_value": (exceed + 1) / (draws + 1),
        "draws": draws,
    }


def run(train_zip: Path, valid_zip: Path, test_zip: Path, draws: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="komap-group-similarity-") as directory:
        temp = Path(directory)
        train_rows, train_hashes = zip_images(train_zip, temp / "train_images")
        valid_rows, valid_hashes = zip_images(valid_zip, temp / "valid_images")
        test_rows, test_hashes = zip_images(test_zip, temp / "test_images")

    valid_groups = {row["group"] for row in valid_rows}
    test_groups = {row["group"] for row in test_rows}
    overlap = sorted(valid_groups & test_groups)
    valid_without_overlap = [row for row in valid_rows if row["group"] not in overlap]
    test_without_overlap = [row for row in test_rows if row["group"] not in overlap]
    shared_group_features = {}
    for group in overlap:
        shared_group_features[group] = {
            "valid_images": [row["name"] for row in valid_rows if row["group"] == group],
            "test_images": [row["name"] for row in test_rows if row["group"] == group],
        }

    duplicates = []
    valid_by_hash = {digest: name for name, digest in valid_hashes.items()}
    for name, digest in test_hashes.items():
        if digest in valid_by_hash:
            duplicates.append({"valid": valid_by_hash[digest], "test": name})

    all_rows = {"train": train_rows, "valid": valid_rows, "test": test_rows}
    z = standardized(all_rows)
    valid_z = z["valid"]
    test_z = z["test"]
    distance = cdist(test_z, valid_z)
    a71_index = next(i for i, row in enumerate(valid_rows) if row["name"] == "A71_step2_sample1")
    a71_top3 = sum(a71_index in np.argsort(row)[:3] for row in distance)
    nearest_for_a71 = min(
        ((float(np.linalg.norm(valid_z[a71_index] - test_z[i])), row["name"])
         for i, row in enumerate(test_rows)), key=lambda item: item[0]
    )

    return {
        "scope": "raw input images from train/valid/test ZIPs only; no masks or test labels read",
        "counts": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
        "specimen_groups": {
            "train": len({row["group"] for row in train_rows}),
            "valid": len(valid_groups),
            "test": len(test_groups),
        },
        "group_overlap": {
            "valid_test": overlap,
            "train_test": sorted({row["group"] for row in train_rows} & test_groups),
        },
        "shared_group_images": shared_group_features,
        "exact_image_hash_duplicates_valid_test": duplicates,
        "step_counts": {"valid": step_counts(valid_rows), "test": step_counts(test_rows)},
        "resolution_counts": {"valid": resolution_counts(valid_rows), "test": resolution_counts(test_rows)},
        "group_bootstrap_feature_differences_including_overlap":
            clustered_bootstrap_difference(valid_rows, test_rows, draws, 410),
        "group_bootstrap_feature_differences_excluding_overlap":
            clustered_bootstrap_difference(valid_without_overlap, test_without_overlap, draws, 411),
        "group_disjoint_centroid_permutation": group_disjoint_centroid_permutation(
            valid_rows, test_rows, train_rows, draws, 412
        ),
        "a71_raw_feature_similarity": {
            "test_images_where_A71_is_among_three_nearest_valid": int(a71_top3),
            "test_images_total": len(test_rows),
            "nearest_test_image": nearest_for_a71[1],
            "distance_to_nearest_test": nearest_for_a71[0],
            "valid_rank_for_nearest_test_image": int(np.argsort(distance[
                next(i for i, row in enumerate(test_rows) if row["name"] == nearest_for_a71[1])
            ]).tolist().index(a71_index) + 1),
        },
        "interpretation_guard": (
            "Group-aware intervals and centroid permutation have low power at 18-19 valid versus 9-10 test groups. "
            "Failure to reject a centroid difference is not evidence of distributional equivalence or similar mIoU."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-zip", type=Path, required=True)
    parser.add_argument("--valid-zip", type=Path, required=True)
    parser.add_argument("--test-zip", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.train_zip, args.valid_zip, args.test_zip, args.draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
