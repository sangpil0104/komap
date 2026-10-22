#!/usr/bin/env python3
"""Compare per-image scale-selection headroom with an equal-size 1x null oracle."""

from __future__ import annotations

import argparse
import contextlib
import json
import pickle
from pathlib import Path

import numpy as np


def image_mean_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    matrix = np.bincount(
        target.reshape(-1).astype(np.int64) * 4 + prediction.reshape(-1).astype(np.int64),
        minlength=16,
    ).reshape(4, 4)
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(0) + matrix.sum(1) - intersection
    return float((intersection / np.maximum(union, 1.0)).mean())


def load_archive_group(paths: list[str], group_name: str, stack: contextlib.ExitStack):
    if not paths:
        raise ValueError(f"no archives provided for {group_name}")
    archives = {}
    orders = set()
    for path in paths:
        archive = stack.enter_context(np.load(path))
        name = Path(path).parent.name
        if name in archives:
            name = Path(path).stem
        archives[name] = archive
        orders.add(tuple(archive.files))
    if len(orders) != 1:
        raise ValueError(f"archive keys/order differ within {group_name}")
    return archives, tuple(next(iter(orders)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", required=True)
    parser.add_argument("--scale-archives", nargs="+", required=True)
    parser.add_argument("--null-archives", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path(args.entries).open("rb") as stream:
        entries = pickle.load(stream)

    with contextlib.ExitStack() as stack:
        scale_archives, scale_keys = load_archive_group(args.scale_archives, "scale", stack)
        null_archives, null_keys = load_archive_group(args.null_archives, "null", stack)
        if scale_keys != null_keys or len(scale_keys) != len(entries):
            raise ValueError("scale/null probability archives do not align with cache entries")

        scale_scores = {name: np.zeros(len(entries), dtype=np.float64)
                        for name in scale_archives}
        null_scores = {name: np.zeros(len(entries), dtype=np.float64)
                       for name in null_archives}
        image_groups = []
        for index, (key, item) in enumerate(zip(scale_keys, entries)):
            target = item["target"].astype(np.uint8)
            image_groups.append(key.split("_", 1)[0])
            for name, archive in scale_archives.items():
                scale_scores[name][index] = image_mean_iou(
                    archive[key].argmax(axis=0).astype(np.uint8), target)
            for name, archive in null_archives.items():
                null_scores[name][index] = image_mean_iou(
                    archive[key].argmax(axis=0).astype(np.uint8), target)
            if (index + 1) % 5 == 0:
                print(f"scored {index + 1}/{len(entries)} validation images", flush=True)

    scale_matrix = np.stack(list(scale_scores.values()))
    null_matrix = np.stack(list(null_scores.values()))
    scale_oracle = scale_matrix.max(axis=0)
    null_oracle = null_matrix.max(axis=0)
    delta = scale_oracle - null_oracle
    groups: dict[str, list[int]] = {}
    for index, group in enumerate(image_groups):
        groups.setdefault(group, []).append(index)
    group_names = list(groups)
    rng = np.random.default_rng(290)
    bootstrap = np.empty(20000, dtype=np.float64)
    for draw in range(len(bootstrap)):
        sampled = rng.choice(group_names, size=len(group_names), replace=True)
        indices = [index for group in sampled for index in groups[str(group)]]
        bootstrap[draw] = delta[indices].mean() * 100.0

    best_single_scale = max(scale_scores, key=lambda name: scale_scores[name].mean())
    best_single_null = max(null_scores, key=lambda name: null_scores[name].mean())
    result = {
        "metric": "per-image mean of four class IoUs; empty union is 0",
        "image_count": len(entries),
        "specimen_group_count": len(groups),
        "scale_candidates": list(scale_scores),
        "null_candidates": list(null_scores),
        "candidate_count_matched": len(scale_scores) == len(null_scores),
        "individual_scale_means": {name: float(value.mean())
                                    for name, value in scale_scores.items()},
        "individual_null_means": {name: float(value.mean())
                                  for name, value in null_scores.items()},
        "best_single_scale": {"name": best_single_scale,
                              "score": float(scale_scores[best_single_scale].mean())},
        "best_single_null": {"name": best_single_null,
                             "score": float(null_scores[best_single_null].mean())},
        "scale_oracle_mean": float(scale_oracle.mean()),
        "scale_oracle_gain_vs_best_single_pp": float(
            (scale_oracle.mean() - scale_scores[best_single_scale].mean()) * 100.0),
        "null_oracle_mean": float(null_oracle.mean()),
        "null_oracle_gain_vs_best_single_pp": float(
            (null_oracle.mean() - null_scores[best_single_null].mean()) * 100.0),
        "scale_oracle_delta_vs_null_pp": float(delta.mean() * 100.0),
        "group_bootstrap_95ci_delta_pp": [
            float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "scale_winner_counts": {name: int((scale_matrix.argmax(0)
                                            == index).sum())
                                for index, name in enumerate(scale_scores)},
        "null_winner_counts": {name: int((null_matrix.argmax(0)
                                           == index).sum())
                               for index, name in enumerate(null_scores)},
        "per_image": [{
            "name": key, "specimen": image_groups[index],
            "scale_oracle": float(scale_oracle[index]),
            "null_oracle": float(null_oracle[index]),
            "delta_pp": float(delta[index] * 100.0),
            "scale_winner": list(scale_scores)[int(scale_matrix[:, index].argmax())],
            "null_winner": list(null_scores)[int(null_matrix[:, index].argmax())],
        } for index, key in enumerate(scale_keys)],
        "caveat": "GT-best scale is an unattainable oracle; equal-count 1x oracle estimates max-selection bias.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "individual_scale_means", "individual_null_means", "scale_oracle_mean",
        "scale_oracle_gain_vs_best_single_pp", "null_oracle_mean",
        "null_oracle_gain_vs_best_single_pp", "scale_oracle_delta_vs_null_pp",
        "group_bootstrap_95ci_delta_pp")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
