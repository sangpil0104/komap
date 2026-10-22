#!/usr/bin/env python3
"""Search label-free, image-conditioned selection among model-197 candidates.

The candidate scores are read from the already completed validation sweep, but
the selector uses only image statistics and model-output statistics as its
features.  This is useful for checking whether one global gamma is hiding a
simple brightness/uncertainty-conditioned route.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from improve_five_causes import collect_split
from model187_schedule_branch_blend import undo_model181_primary_blend


def build_features(entries: list[dict], samples: list[tuple[Path, Path]]) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    for item, (image_path, _) in zip(entries, samples):
        raw = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        base = undo_model181_primary_blend(item)
        prediction = base.argmax(axis=0)
        entropy = -(base * np.log(np.maximum(base, 1e-7))).sum(axis=0)
        row = {
            "gray_mean": float(raw.mean()),
            "gray_std": float(raw.std()),
            "gray_p10": float(np.percentile(raw, 10.0)),
            "gray_p50": float(np.percentile(raw, 50.0)),
            "gray_p90": float(np.percentile(raw, 90.0)),
            "mean_entropy": float(entropy.mean()),
            "mean_confidence": float(base.max(axis=0).mean()),
            "area_al3ni": float((prediction == 0).mean()),
            "area_primary": float((prediction == 1).mean()),
            "area_al": float((prediction == 2).mean()),
            "area_eutectic": float((prediction == 3).mean()),
            "mean_p_al3ni": float(base[0].mean()),
            "mean_p_primary": float(base[1].mean()),
            "mean_p_al": float(base[2].mean()),
            "mean_p_eutectic": float(base[3].mean()),
            "margin_al3ni_primary": float((base[0] - base[1]).mean()),
            "margin_al3ni_eutectic": float((base[0] - base[3]).mean()),
            "margin_primary_eutectic": float((base[1] - base[3]).mean()),
            "high_weight": float(item["params"]["high_weight"]),
            "pair_weight": float(item["params"]["pair_weight"]),
            "adaptive_beta": float(item["params"]["beta"]),
            "eutectic_bias": float(item["params"]["eut_bias"]),
        }
        for key, value in row.items():
            values.setdefault(key, []).append(value)
    return {key: np.asarray(value, dtype=np.float64) for key, value in values.items()}


def describe(record: dict) -> dict:
    result = {
        key: record[key]
        for key in ("primary_gamma", "mode", "rare_gamma", "source", "stage",
                    "global_miou", "image_mean_miou", "image_mean_miou_exclude",
                    "_result_path")
        if key in record
    }
    return result


def load_candidates(path: Path, expected_count: int) -> list[dict]:
    """Load either one result file or all metric-bearing records below a dir."""
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["records"]

    candidates: list[dict] = []

    def walk(value: object, source: Path) -> None:
        if isinstance(value, dict):
            if (len(value.get("image_scores", [])) == expected_count
                    and len(value.get("image_scores_exclude", [])) == expected_count
                    and "image_mean_miou" in value):
                row = dict(value)
                row["_result_path"] = str(source)
                candidates.append(row)
            for child in value.values():
                if isinstance(child, (dict, list)):
                    walk(child, source)
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, (dict, list)):
                    walk(child, source)

    for result_path in sorted(path.rglob("*.json")):
        try:
            walk(json.loads(result_path.read_text(encoding="utf-8")), result_path)
        except (OSError, json.JSONDecodeError):
            continue
    return candidates


def thresholds_for(feature: np.ndarray) -> list[float]:
    order = np.argsort(feature)
    return [
        float((feature[order[i - 1]] + feature[order[i]]) / 2.0)
        for i in range(1, len(feature))
        if feature[order[i - 1]] != feature[order[i]]
    ]


def best_two_bin(matrix: np.ndarray, indices: np.ndarray,
                 features: dict[str, np.ndarray], pool: np.ndarray) -> tuple[float, dict, np.ndarray]:
    """Find the best two-leaf selector restricted to a sample subset."""
    best = (-1.0, None, None)
    baseline = int(pool[np.argmax(matrix[pool][:, indices].mean(axis=1))])
    selected = np.full(len(matrix[0]), np.nan, dtype=np.float64)
    selected[indices] = matrix[baseline, indices]
    best = (float(selected[indices].mean()), {"candidate": baseline}, selected)
    for feature_name, feature in features.items():
        local = feature[indices]
        for threshold in thresholds_for(local):
            left = indices[local < threshold]
            right = indices[local >= threshold]
            if len(left) == 0 or len(right) == 0:
                continue
            left_candidate = int(pool[np.argmax(matrix[pool][:, left].mean(axis=1))])
            right_candidate = int(pool[np.argmax(matrix[pool][:, right].mean(axis=1))])
            values = np.empty(len(indices), dtype=np.float64)
            values[local < threshold] = matrix[left_candidate, left]
            values[local >= threshold] = matrix[right_candidate, right]
            score = float(values.mean())
            if score > best[0]:
                full = np.full(len(matrix[0]), np.nan, dtype=np.float64)
                full[left] = matrix[left_candidate, left]
                full[right] = matrix[right_candidate, right]
                best = (score, {"feature": feature_name, "threshold": threshold,
                                "left_candidate": left_candidate,
                                "right_candidate": right_candidate}, full)
    return best


def main(args: argparse.Namespace) -> None:
    with Path(args.entries).open("rb") as handle:
        entries = pickle.load(handle)
    candidates = load_candidates(Path(args.candidates), len(entries))
    candidates = [row for row in candidates
                  if len(row.get("image_scores", [])) == len(entries)
                  and len(row.get("image_scores_exclude", [])) == len(entries)]
    samples = collect_split(args.data_dir, "valid")
    features = build_features(entries, samples)
    include = np.asarray([row["image_scores"] for row in candidates], dtype=np.float64)
    exclude = np.asarray([row["image_scores_exclude"] for row in candidates], dtype=np.float64)
    top_count = min(args.top_count, len(candidates))

    output: dict[str, object] = {"candidate_count": len(candidates), "features": {}}
    for key, value in features.items():
        output["features"][key] = value.tolist()  # type: ignore[index]

    for metric_name, matrix in (("include", include), ("exclude", exclude)):
        top_indices = np.argsort(matrix.mean(axis=1))[-top_count:]
        best_pair = (-1.0, None)
        best_bins = (-1.0, None)
        for feature_name, feature in features.items():
            order = np.argsort(feature)
            thresholds = [
                float((feature[order[i - 1]] + feature[order[i]]) / 2.0)
                for i in range(1, len(feature))
                if feature[order[i - 1]] != feature[order[i]]
            ]
            for threshold in thresholds:
                mask = feature < threshold
                left = np.flatnonzero(mask)
                right = np.flatnonzero(~mask)
                if len(left) == 0 or len(right) == 0:
                    continue
                left_index = int(top_indices[np.argmax(matrix[top_indices][:, left].mean(axis=1))])
                right_index = int(top_indices[np.argmax(matrix[top_indices][:, right].mean(axis=1))])
                selected = np.where(mask, matrix[left_index], matrix[right_index])
                score = float(selected.mean())
                if score > best_pair[0]:
                    best_pair = (score, {
                        "feature": feature_name, "threshold": threshold,
                        "left_candidate": left_index, "right_candidate": right_index,
                        "left_count": int(len(left)), "scores": selected.tolist(),
                    })

            # Three bins using one input-only feature.  The best candidate in
            # each bin is chosen independently; no target-derived feature is
            # passed to the final selector.
            for i, t1 in enumerate(thresholds):
                for t2 in thresholds[i + 1:]:
                    masks = (feature < t1, (feature >= t1) & (feature < t2), feature >= t2)
                    if any(int(mask.sum()) == 0 for mask in masks):
                        continue
                    selected_indices = []
                    for mask in masks:
                        indices = np.flatnonzero(mask)
                        selected_indices.append(int(
                            top_indices[np.argmax(matrix[top_indices][:, indices].mean(axis=1))]))
                    selected = np.empty(len(feature), dtype=np.float64)
                    for mask, index in zip(masks, selected_indices):
                        selected[mask] = matrix[index, mask]
                    score = float(selected.mean())
                    if score > best_bins[0]:
                        best_bins = (score, {
                            "feature": feature_name, "threshold1": t1, "threshold2": t2,
                            "candidates": selected_indices, "scores": selected.tolist(),
                        })

        assert best_pair[1] is not None and best_bins[1] is not None
        pair = best_pair[1]
        bins = best_bins[1]
        # A depth-2 input-only tree: split once, then independently optimize
        # a two-bin selector inside each child.  This remains label-free at
        # inference time while checking whether two simple image statistics
        # can recover complementary hard-case routes.
        pool = top_indices
        all_indices = np.arange(len(features[next(iter(features))]))
        best_tree = (-1.0, None)
        for root_name, root_feature in features.items():
            for root_threshold in thresholds_for(root_feature):
                left_indices = all_indices[root_feature < root_threshold]
                right_indices = all_indices[root_feature >= root_threshold]
                if len(left_indices) < 2 or len(right_indices) < 2:
                    continue
                left = best_two_bin(matrix, left_indices, features, pool)
                right = best_two_bin(matrix, right_indices, features, pool)
                selected = np.full(len(all_indices), np.nan, dtype=np.float64)
                selected[left_indices] = left[2][left_indices]
                selected[right_indices] = right[2][right_indices]
                score = float(selected.mean())
                if score > best_tree[0]:
                    best_tree = (score, {
                        "root_feature": root_name, "root_threshold": root_threshold,
                        "left": left[1], "right": right[1],
                        "scores": selected.tolist(),
                    })
        tree = best_tree[1]
        output[metric_name] = {
            "global_best": int(np.argmax(matrix.mean(axis=1))),
            "global_best_description": describe(candidates[int(np.argmax(matrix.mean(axis=1)))]),
            "pair_score": best_pair[0],
            "pair": pair,
            "pair_left_description": describe(candidates[pair["left_candidate"]]),
            "pair_right_description": describe(candidates[pair["right_candidate"]]),
            "bins_score": best_bins[0],
            "bins": bins,
            "bin_descriptions": [describe(candidates[index]) for index in bins["candidates"]],
            "tree_score": best_tree[0],
            "tree": tree,
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    for metric in ("include", "exclude"):
        section = output[metric]
        print(metric, "global", section["global_best_description"],
              "pair", section["pair_score"] * 100, section["pair_left_description"],
              section["pair_right_description"],
              "bins", section["bins_score"] * 100, section["bin_descriptions"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--entries", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-count", type=int, default=40)
    main(parser.parse_args())
