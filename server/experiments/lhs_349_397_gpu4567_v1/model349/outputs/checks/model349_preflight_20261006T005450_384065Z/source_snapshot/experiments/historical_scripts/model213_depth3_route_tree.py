#!/usr/bin/env python3
"""Diagnostic depth-3 input-only route selector over completed candidates."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from improve_five_causes import collect_split
from model203_conditional_calibration import (
    best_two_bin,
    build_features,
    describe,
    load_candidates,
    thresholds_for,
)


def best_depth2(matrix, indices, features, pool):
    """Best one-split selector inside a node (used for each depth-1 child)."""
    all_indices = np.asarray(indices, dtype=np.int64)
    baseline = int(pool[np.argmax(matrix[pool][:, all_indices].mean(axis=1))])
    best = (float(matrix[baseline, all_indices].mean()),
            {"candidate": baseline},
            np.full(matrix.shape[1], np.nan, dtype=np.float64))
    best[2][all_indices] = matrix[baseline, all_indices]
    for feature_name, feature in features.items():
        local = feature[all_indices]
        for threshold in thresholds_for(local):
            left = all_indices[local < threshold]
            right = all_indices[local >= threshold]
            if len(left) == 0 or len(right) == 0:
                continue
            left_candidate = int(pool[np.argmax(matrix[pool][:, left].mean(axis=1))])
            right_candidate = int(pool[np.argmax(matrix[pool][:, right].mean(axis=1))])
            selected = np.full(matrix.shape[1], np.nan, dtype=np.float64)
            selected[left] = matrix[left_candidate, left]
            selected[right] = matrix[right_candidate, right]
            score = float(selected[all_indices].mean())
            if score > best[0]:
                best = (score, {"feature": feature_name, "threshold": threshold,
                                "left_candidate": left_candidate,
                                "right_candidate": right_candidate}, selected)
    return best


def main(args):
    with Path(args.entries).open("rb") as handle:
        entries = pickle.load(handle)
    candidates = load_candidates(Path(args.candidates), len(entries))
    samples = collect_split(args.data_dir, "valid")
    features = build_features(entries, samples)
    top_count = min(args.top_count, len(candidates))
    output = {"candidate_count": len(candidates), "metrics": {}}
    all_indices = np.arange(len(entries), dtype=np.int64)
    pool_cache = {}
    for metric_name in ("image_scores", "image_scores_exclude"):
        matrix = np.asarray([row[metric_name] for row in candidates], dtype=np.float64)
        pool = np.argsort(matrix.mean(axis=1))[-top_count:]
        best = (-1.0, None)
        for root_name, root_feature in features.items():
            for root_threshold in thresholds_for(root_feature):
                left_indices = all_indices[root_feature < root_threshold]
                right_indices = all_indices[root_feature >= root_threshold]
                if len(left_indices) < 3 or len(right_indices) < 3:
                    continue
                left = best_depth2(matrix, left_indices, features, pool)
                right = best_depth2(matrix, right_indices, features, pool)
                selected = np.full(len(entries), np.nan, dtype=np.float64)
                selected[left_indices] = left[2][left_indices]
                selected[right_indices] = right[2][right_indices]
                score = float(selected.mean())
                if score > best[0]:
                    best = (score, {"root_feature": root_name,
                                    "root_threshold": root_threshold,
                                    "left": left[1], "right": right[1],
                                    "scores": selected.tolist()})
        tree = best[1]
        def descriptions(node):
            if "candidate" in node:
                return describe(candidates[node["candidate"]])
            return {
                "feature": node["feature"], "threshold": node["threshold"],
                "left": describe(candidates[node["left_candidate"]]),
                "right": describe(candidates[node["right_candidate"]]),
            }
        output["metrics"][metric_name] = {
            "score": best[0], "tree": tree,
            "left": descriptions(tree["left"]),
            "right": descriptions(tree["right"]),
        }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    for metric, result in output["metrics"].items():
        print(metric, result["score"] * 100, result["tree"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--entries", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-count", type=int, default=60)
    main(parser.parse_args())
