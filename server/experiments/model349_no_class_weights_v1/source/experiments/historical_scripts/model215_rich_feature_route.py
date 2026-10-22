#!/usr/bin/env python3
"""Search an input-only route tree with richer texture and geometry features.

The earlier route searches used only global gray statistics and class area.
This diagnostic adds label-free features that are available for test images as
well: gradient/texture summaries, prediction boundary density, connected
component counts, class-wise confidence, margins, and spatial moments.  The
tree is fit only to candidate image scores for validation analysis; the
reported rules are useful only if their split is physically interpretable and
stable.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from improve_five_causes import collect_split
from model187_schedule_branch_blend import undo_model181_primary_blend
from model203_conditional_calibration import load_candidates, thresholds_for


def _summary(values: dict[str, list[float]], name: str, array: np.ndarray) -> None:
    flat = array.astype(np.float32).reshape(-1)
    for suffix, value in (
        ("mean", flat.mean()),
        ("std", flat.std()),
        ("p10", np.percentile(flat, 10.0)),
        ("p50", np.percentile(flat, 50.0)),
        ("p90", np.percentile(flat, 90.0)),
    ):
        values.setdefault(f"{name}_{suffix}", []).append(float(value))


def build_features(entries: list[dict], samples: list[tuple[Path, Path]]) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    for item, (image_path, _) in zip(entries, samples):
        raw = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        base = undo_model181_primary_blend(item).astype(np.float32)
        prediction = base.argmax(axis=0).astype(np.uint8)
        entropy = -(base * np.log(np.maximum(base, 1e-7))).sum(axis=0)
        sorted_prob = np.sort(base, axis=0)
        margin = sorted_prob[-1] - sorted_prob[-2]

        gy, gx = np.gradient(raw)
        gradient = np.hypot(gx, gy)
        local_mean = ndimage.uniform_filter(raw, size=9, mode="reflect")
        local_contrast = np.abs(raw - local_mean)
        edge = gradient > np.percentile(gradient, 75.0)
        _summary(values, "gray", raw)
        _summary(values, "gradient", gradient)
        _summary(values, "local_contrast", local_contrast)
        _summary(values, "entropy", entropy)
        _summary(values, "margin", margin)

        height, width = raw.shape
        yy, xx = np.mgrid[:height, :width]
        xx = (xx.astype(np.float32) / max(width - 1, 1)) - 0.5
        yy = (yy.astype(np.float32) / max(height - 1, 1)) - 0.5
        values.setdefault("aspect", []).append(float(width / max(height, 1)))
        values.setdefault("edge_fraction", []).append(float(edge.mean()))
        values.setdefault("border_gray_mean", []).append(float(np.concatenate(
            (raw[0, :], raw[-1, :], raw[:, 0], raw[:, -1])).mean()))
        values.setdefault("center_gray_mean", []).append(float(raw[
            height // 4:3 * height // 4, width // 4:3 * width // 4].mean()))

        for class_id, name in enumerate(("al3ni", "primary", "al", "eutectic")):
            mask = prediction == class_id
            values.setdefault(f"area_{name}", []).append(float(mask.mean()))
            values.setdefault(f"confidence_{name}", []).append(float(base[class_id].mean()))
            values.setdefault(f"margin_{name}", []).append(float((
                base[class_id] - np.max(np.delete(base, class_id, axis=0), axis=0)
            ).mean()))
            labels, count = ndimage.label(mask)
            sizes = np.bincount(labels.reshape(-1))[1:]
            values.setdefault(f"components_{name}", []).append(float(count))
            values.setdefault(f"large_components_{name}", []).append(float(
                (sizes >= 64).sum()))
            values.setdefault(f"boundary_{name}", []).append(float(
                (mask ^ ndimage.binary_erosion(mask, iterations=1)).mean()))
            if mask.any():
                values.setdefault(f"x_center_{name}", []).append(float(xx[mask].mean()))
                values.setdefault(f"y_center_{name}", []).append(float(yy[mask].mean()))
                values.setdefault(f"x_spread_{name}", []).append(float(xx[mask].std()))
                values.setdefault(f"y_spread_{name}", []).append(float(yy[mask].std()))
            else:
                for suffix in ("x_center", "y_center", "x_spread", "y_spread"):
                    values.setdefault(f"{suffix}_{name}", []).append(0.0)

        # Per-class uncertainty and texture are often more informative than a
        # global entropy when a rare phase is only present in a narrow band.
        for class_id, name in enumerate(("al3ni", "primary", "al", "eutectic")):
            weighted_entropy = entropy * base[class_id]
            _summary(values, f"weighted_entropy_{name}", weighted_entropy)
            _summary(values, f"class_gradient_{name}", gradient * base[class_id])

    return {key: np.asarray(value, dtype=np.float64) for key, value in values.items()}


def leaf(matrix: np.ndarray, indices: np.ndarray, pool: np.ndarray) -> tuple[int, float]:
    scores = matrix[pool][:, indices].mean(axis=1)
    best = int(pool[np.argmax(scores)])
    return best, float(scores.max())


def fit_tree(matrix: np.ndarray, features: dict[str, np.ndarray], pool_size: int,
             max_depth: int, min_leaf: int) -> tuple[float, dict]:
    all_indices = np.arange(matrix.shape[1], dtype=np.int64)
    pool = np.argsort(matrix.mean(axis=1))[-min(pool_size, matrix.shape[0]):]

    def recurse(indices: np.ndarray, depth: int) -> tuple[float, dict]:
        candidate, score = leaf(matrix, indices, pool)
        best_score = score
        best_node: dict = {"candidate": candidate, "count": int(len(indices))}
        if depth >= max_depth or len(indices) < 2 * min_leaf:
            return best_score, best_node
        for feature_name, feature in features.items():
            local = feature[indices]
            for threshold in thresholds_for(local):
                left = indices[local < threshold]
                right = indices[local >= threshold]
                if len(left) < min_leaf or len(right) < min_leaf:
                    continue
                left_score, left_node = recurse(left, depth + 1)
                right_score, right_node = recurse(right, depth + 1)
                weighted = (left_score * len(left) + right_score * len(right)) / len(indices)
                if weighted > best_score + 1e-12:
                    best_score = weighted
                    best_node = {
                        "feature": feature_name,
                        "threshold": float(threshold),
                        "left": left_node,
                        "right": right_node,
                        "count": int(len(indices)),
                    }
        return best_score, best_node

    return recurse(all_indices, 0)


def main(args: argparse.Namespace) -> None:
    entries = pickle.load(Path(args.entries).open("rb"))
    candidates = load_candidates(Path(args.candidates), len(entries))
    samples = collect_split(args.data_dir, "valid")
    features = build_features(entries, samples)
    output: dict[str, object] = {
        "candidate_count": len(candidates),
        "feature_names": sorted(features),
        "features": {key: value.tolist() for key, value in features.items()},
        "metrics": {},
    }
    for metric_name in ("image_scores", "image_scores_exclude"):
        matrix = np.asarray([row[metric_name] for row in candidates], dtype=np.float64)
        metric_output = {}
        for depth in range(1, args.max_depth + 1):
            score, tree = fit_tree(matrix, features, args.pool_size, depth, args.min_leaf)
            metric_output[str(depth)] = {"score": score, "tree": tree}
            print(metric_name, "depth", depth, "score", score * 100, flush=True)
        output["metrics"][metric_name] = metric_output
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--entries", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pool-size", type=int, default=80)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-leaf", type=int, default=2)
    main(parser.parse_args())
