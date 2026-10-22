#!/usr/bin/env python3
"""Train-only screen for an unlabeled Al-matrix-mode intensity anchor.

Reads the 70 grouped-OOF training images/masks only. Image transforms use
grayscale pixels alone; labels are used exclusively for class-separation
diagnostics. No official-validation file is accessed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

PALETTE = np.array(
    [[153, 127, 76], [25, 76, 153], [76, 178, 76], [204, 204, 204]],
    dtype=np.uint8,
)
METHODS = ("raw", "zscore", "cdf", "al_mode_anchor")


def sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
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
        raise ValueError(f"unknown palette color in {path}")
    return result


def score_luts(gray: np.ndarray) -> dict[str, np.ndarray]:
    levels = np.arange(256, dtype=np.float32)
    x = levels / 255.0
    image = gray.astype(np.float32) / 255.0
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    cumulative = np.cumsum(histogram) / max(float(histogram.sum()), 1.0)
    mean, std = float(image.mean()), float(image.std())
    mode = int(np.argmax(histogram))
    q05, q95 = np.percentile(gray, [5, 95])
    spread = max(float(q95 - q05), 1.0)
    return {
        "raw": (x - 0.5) / 0.25,
        "zscore": np.clip((x - mean) / (std + 1e-6), -4.0, 4.0) / 3.0,
        "cdf": (cumulative - 0.5) / 0.288675,
        "al_mode_anchor": (levels - float(mode)) / spread,
    }


def score_histogram(score_lut: np.ndarray, pixel_histogram: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    used = np.flatnonzero(pixel_histogram)
    scores, inverse = np.unique(score_lut[used], return_inverse=True)
    counts = np.zeros(len(scores), dtype=np.float64)
    np.add.at(counts, inverse, pixel_histogram[used])
    return scores.astype(np.float64), counts


def pair_auc_numerator(pos_scores: np.ndarray, pos_counts: np.ndarray,
                       neg_scores: np.ndarray, neg_counts: np.ndarray) -> float:
    cumulative_neg = np.concatenate(([0.0], np.cumsum(neg_counts)))
    left = np.searchsorted(neg_scores, pos_scores, side="left")
    right = np.searchsorted(neg_scores, pos_scores, side="right")
    below = cumulative_neg[left]
    tied = cumulative_neg[right] - cumulative_neg[left]
    return float(np.sum(pos_counts * (below + 0.5 * tied)))


def orientation_free_auc(numerator: float, positives: float,
                         negatives: float) -> float:
    if positives <= 0 or negatives <= 0:
        return float("nan")
    auc = numerator / (positives * negatives)
    return max(auc, 1.0 - auc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-data", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=37_474)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    if args.bootstrap < 100:
        raise ValueError("need at least 100 group-bootstrap draws")

    examples = []
    all_input_paths = []
    for fold in args.folds:
        image_dir = args.cv_data / f"oof_fold{fold}" / "images"
        mask_dir = args.cv_data / f"oof_fold{fold}" / "masks"
        image_paths = sorted(image_dir.glob("*_image.png"))
        if not image_paths:
            raise FileNotFoundError(f"no held-out images in {image_dir}")
        for image_path in image_paths:
            image_id = image_path.name.removesuffix("_image.png")
            mask_path = mask_dir / f"{image_id}_mask.png"
            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)
            match = re.match(r"(A\d+)_", image_id)
            if match is None:
                raise ValueError(f"cannot parse filename group from {image_id}")
            gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
            target = load_mask(mask_path)
            if gray.shape != target.shape:
                raise ValueError(f"image/mask size mismatch: {image_id}")

            total_hist = np.bincount(gray.ravel(), minlength=256)
            class1_hist = np.bincount(gray[target == 1], minlength=256)
            class3_hist = np.bincount(gray[target == 3], minlength=256)
            mode = int(np.argmax(total_hist))
            mode_al_fraction = float(np.count_nonzero((gray == mode) & (target == 0))
                                     / max(int(total_hist[mode]), 1))
            luts = score_luts(gray)
            histograms = {}
            for method in METHODS:
                histograms[method] = (
                    *score_histogram(luts[method], class1_hist),
                    *score_histogram(luts[method], class3_hist),
                )
            examples.append({
                "fold": fold,
                "image_id": image_id,
                "group_id": match.group(1),
                "mode_al_fraction": mode_al_fraction,
                "histograms": histograms,
            })
            all_input_paths.extend((image_path, mask_path))

    groups, group_index = np.unique(
        [row["group_id"] for row in examples], return_inverse=True)
    group_count = len(groups)
    rng = np.random.default_rng(args.seed)
    result_methods = {}
    bootstrap_metrics = {}

    for method in METHODS:
        per_image_auc = []
        positive_totals = np.zeros(len(examples), dtype=np.float64)
        negative_totals = np.zeros(len(examples), dtype=np.float64)
        score_histograms = []
        for index, row in enumerate(examples):
            pos_scores, pos_counts, neg_scores, neg_counts = row["histograms"][method]
            numerator = pair_auc_numerator(pos_scores, pos_counts, neg_scores, neg_counts)
            p_total, n_total = float(pos_counts.sum()), float(neg_counts.sum())
            per_image_auc.append(orientation_free_auc(numerator, p_total, n_total))
            positive_totals[index], negative_totals[index] = p_total, n_total
            score_histograms.append((pos_scores, pos_counts, neg_scores, neg_counts))

        pair_contributions = np.zeros((len(examples), len(examples)), dtype=np.float64)
        for i, (pos_scores, pos_counts, _, _) in enumerate(score_histograms):
            for j, (_, _, neg_scores, neg_counts) in enumerate(score_histograms):
                pair_contributions[i, j] = pair_auc_numerator(
                    pos_scores, pos_counts, neg_scores, neg_counts)

        weights = np.ones(len(examples), dtype=np.float64)
        pooled_num = float(np.einsum("i,ij,j->", weights, pair_contributions, weights))
        pooled = orientation_free_auc(
            pooled_num, float(positive_totals.sum()), float(negative_totals.sum()))
        valid_images = np.isfinite(per_image_auc)
        macro = float(np.mean(np.asarray(per_image_auc)[valid_images]))
        result_methods[method] = {
            "pooled_pair_separability_auc": pooled,
            "image_macro_pair_separability_auc": macro,
            "absolute_pooled_vs_image_macro_gap": abs(pooled - macro),
            "images_with_both_phases": int(valid_images.sum()),
        }
        bootstrap_metrics[method] = {
            "contributions": pair_contributions,
            "positive_totals": positive_totals,
            "negative_totals": negative_totals,
            "per_image_auc": np.asarray(per_image_auc),
        }

    draws = np.empty((args.bootstrap, len(METHODS)), dtype=np.float64)
    gap_draws = np.empty((args.bootstrap, len(METHODS)), dtype=np.float64)
    for draw in range(args.bootstrap):
        sampled = rng.integers(0, group_count, size=group_count)
        group_weights = np.bincount(sampled, minlength=group_count)
        image_weights = group_weights[group_index].astype(np.float64)
        for method_index, method in enumerate(METHODS):
            audit = bootstrap_metrics[method]
            numerator = float(np.einsum(
                "i,ij,j->", image_weights, audit["contributions"], image_weights))
            positives = float(np.dot(image_weights, audit["positive_totals"]))
            negatives = float(np.dot(image_weights, audit["negative_totals"]))
            pooled = orientation_free_auc(numerator, positives, negatives)
            per_auc = audit["per_image_auc"]
            usable = np.isfinite(per_auc)
            macro = float(np.dot(image_weights[usable], per_auc[usable])
                          / image_weights[usable].sum())
            draws[draw, method_index] = pooled
            gap_draws[draw, method_index] = abs(pooled - macro)

    method_index = {name: i for i, name in enumerate(METHODS)}
    cdf_i, anchor_i = method_index["cdf"], method_index["al_mode_anchor"]
    pooled_delta = draws[:, anchor_i] - draws[:, cdf_i]
    gap_reduction = gap_draws[:, cdf_i] - gap_draws[:, anchor_i]
    cdf_gap = result_methods["cdf"]["absolute_pooled_vs_image_macro_gap"]
    anchor_gap = result_methods["al_mode_anchor"]["absolute_pooled_vs_image_macro_gap"]
    modal_al_fractions = np.asarray([row["mode_al_fraction"] for row in examples])
    gates = {
        "modal_bin_al_fraction_gt_0_5_in_at_least_70pct_images": bool(
            np.mean(modal_al_fractions > 0.5) >= 0.70),
        "anchor_reduces_cdf_gap_by_at_least_half": bool(
            anchor_gap <= 0.5 * cdf_gap) if cdf_gap > 0 else anchor_gap == 0,
        "anchor_auc_gain_at_least_0_02": bool(
            result_methods["al_mode_anchor"]["pooled_pair_separability_auc"]
            - result_methods["cdf"]["pooled_pair_separability_auc"] >= 0.02),
        "group_bootstrap_auc_gain_ci_lower_gt_0": bool(np.quantile(pooled_delta, 0.025) > 0),
    }
    result = {
        "purpose": "train-only grayscale class-separability screen; not a segmentation score",
        "metric": "orientation-free pooled Al3Ni-vs-Eutectic pixel AUC",
        "class_pair": ["Al3Ni", "Eutectic_Si"],
        "folds": args.folds,
        "images": len(examples),
        "filename_proxy_groups": group_count,
        "model329_input_reference": "raw + full-image z-score + full-image empirical CDF",
        "al_mode_anchor_definition": "(gray - lowest argmax-bin of full-image histogram) / max(q95-q05,1); no clipping",
        "modal_bin_al_fraction_median": float(np.median(modal_al_fractions)),
        "modal_bin_al_fraction_images_gt_0_5": int(np.count_nonzero(modal_al_fractions > 0.5)),
        "methods": result_methods,
        "anchor_minus_cdf_pooled_auc_delta": float(
            result_methods["al_mode_anchor"]["pooled_pair_separability_auc"]
            - result_methods["cdf"]["pooled_pair_separability_auc"]),
        "anchor_minus_cdf_pooled_auc_group_bootstrap_95_ci": [
            float(np.quantile(pooled_delta, 0.025)),
            float(np.quantile(pooled_delta, 0.975)),
        ],
        "cdf_minus_anchor_gap_reduction_group_bootstrap_95_ci": [
            float(np.quantile(gap_reduction, 0.025)),
            float(np.quantile(gap_reduction, 0.975)),
        ],
        "predeclared_gates": gates,
        "all_gates_pass": bool(all(gates.values())),
        "input_sha256": sha256(all_input_paths),
        "warning": (
            "Intensity AUC is a train-mask diagnostic, not a model mIoU estimate. "
            "A pass only justifies a separate grouped-OOF model experiment."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
