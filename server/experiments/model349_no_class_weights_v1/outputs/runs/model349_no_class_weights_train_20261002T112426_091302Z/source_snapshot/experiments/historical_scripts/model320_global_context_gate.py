#!/usr/bin/env python3
"""Train-only group-OOF gate for image-level context in Al3Ni/Eutectic errors.

This is a cheap falsification test before training a global-context decoder.
It compares a pixel-only binary expert with the same expert augmented by
per-image grayscale and prediction-distribution summaries. Base probabilities
are Model297 train-only specimen-group OOF predictions; outer evaluation keeps
specimen groups disjoint at the meta-classifier level as well.
"""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel, uniform_filter
from scipy.optimize import minimize
from scipy.special import expit

from improve_five_causes import collect_split, decode_mask


def specimen_id(path: Path) -> str:
    match = re.match(r"^(A\d+)_", path.name)
    return match.group(1) if match else path.stem


def feature_maps(image_path: Path, probability: np.ndarray):
    """Return local pixel features, broadcastable image context, and base mask."""
    image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
    if image.shape != probability.shape[1:]:
        raise ValueError(f"image/probability shape mismatch for {image_path.name}: "
                         f"{image.shape} != {probability.shape[1:]}")
    probability = probability.astype(np.float32)
    probability /= np.maximum(probability.sum(axis=0, keepdims=True), 1e-7)
    log_probability = np.log(np.maximum(probability, 1e-7))
    prediction = probability.argmax(axis=0).astype(np.uint8)

    local_mean = uniform_filter(image, size=15, mode="reflect")
    local_sq_mean = uniform_filter(image * image, size=15, mode="reflect")
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 1e-6))
    highpass = image - gaussian_filter(image, sigma=1.5, mode="reflect")
    gradient = np.hypot(sobel(image, axis=0, mode="reflect"),
                        sobel(image, axis=1, mode="reflect"))
    gradient /= max(float(np.percentile(gradient, 99.0)), 1e-6)
    entropy = -(probability * log_probability).sum(axis=0)

    pixel = np.stack([
        *[log_probability[c] for c in range(4)],
        image, local_mean, local_std, highpass, gradient, entropy,
    ], axis=-1).reshape(-1, 10).astype(np.float32)

    fractions = np.bincount(prediction.reshape(-1), minlength=4).astype(np.float32)
    fractions /= float(prediction.size)
    probability_means = probability.mean(axis=(1, 2)).astype(np.float32)
    raw_stats = np.percentile(image, [10, 50, 90]).astype(np.float32)
    predicted_phase_stats = []
    for class_id in (0, 3):
        region = prediction == class_id
        if region.any():
            predicted_phase_stats.extend([
                float(image[region].mean()), float(local_std[region].mean()),
            ])
        else:
            predicted_phase_stats.extend([0.0, 0.0])
    context = np.asarray([
        float(image.mean()), float(image.std()), *raw_stats.tolist(),
        *fractions.tolist(), *probability_means.tolist(), *predicted_phase_stats,
    ], dtype=np.float32)
    return pixel, context, prediction


def choose_binary_pixels(target: np.ndarray, prediction: np.ndarray,
                         samples_per_class: int, rng: np.random.Generator):
    eligible = np.isin(prediction, (0, 3))
    indices = []
    labels = []
    for class_id in (0, 3):
        current = np.flatnonzero((target.reshape(-1) == class_id) & eligible.reshape(-1))
        if len(current) == 0:
            return None
        chosen = rng.choice(current, min(samples_per_class, len(current)), replace=False)
        indices.append(chosen)
        labels.append(np.full(len(chosen), 1 if class_id == 3 else 0, dtype=np.uint8))
    return np.concatenate(indices), np.concatenate(labels)


def fit_logistic(features: np.ndarray, labels: np.ndarray, l2: float):
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-5] = 1.0
    x = np.clip((features - mean) / scale, -10.0, 10.0).astype(np.float64)
    y = labels.astype(np.float64)
    dimension = x.shape[1]

    def objective(parameters):
        weight, bias = parameters[:dimension], parameters[-1]
        logits = np.clip(x @ weight + bias, -40.0, 40.0)
        probability = expit(logits)
        loss = np.logaddexp(0.0, logits).mean() - np.mean(y * logits)
        loss += 0.5 * l2 * float(weight @ weight)
        residual = (probability - y) / len(y)
        gradient = np.concatenate([
            x.T @ residual + l2 * weight,
            np.asarray([residual.sum()]),
        ])
        return loss, gradient

    result = minimize(objective, np.zeros(dimension + 1), jac=True,
                      method="L-BFGS-B", options={"maxiter": 250, "ftol": 1e-10})
    return {"mean": mean, "scale": scale, "parameters": result.x,
            "converged": bool(result.success), "message": str(result.message)}


def predict_binary(model, features: np.ndarray) -> np.ndarray:
    dimension = len(model["mean"])
    x = np.clip((features - model["mean"]) / model["scale"], -10.0, 10.0)
    parameters = model["parameters"]
    return expit(np.clip(x @ parameters[:dimension] + parameters[-1], -40.0, 40.0))


def image_iou(prediction: np.ndarray, target: np.ndarray):
    bins = target.reshape(-1).astype(np.int64) * 4 + prediction.reshape(-1).astype(np.int64)
    matrix = np.bincount(bins, minlength=16).reshape(4, 4)
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(0) + matrix.sum(1) - intersection
    iou = intersection / np.maximum(union, 1.0)
    return float(iou.mean()), iou.tolist(), matrix.tolist()


def bootstrap_group_delta(records, baseline_key, candidate_key, draws=10000, seed=320):
    groups: dict[str, list[float]] = {}
    for row in records:
        groups.setdefault(row["group"], []).append(
            row[candidate_key] - row[baseline_key])
    group_names = sorted(groups)
    group_delta = np.asarray([np.mean(groups[name]) for name in group_names], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(group_delta, size=(draws, len(group_delta)), replace=True).mean(axis=1)
    return {
        "mean_delta_pp": float(np.mean([row[candidate_key] - row[baseline_key]
                                       for row in records]) * 100.0),
        "group_bootstrap_95ci_delta_pp": [
            float(value * 100.0) for value in np.quantile(sampled, [0.025, 0.975])],
        "positive_images": int(sum(row[candidate_key] > row[baseline_key] for row in records)),
        "image_count": len(records), "specimen_group_count": len(group_names),
    }


def main(args):
    started = time.time()
    samples = collect_split(args.data_dir, "train")
    archive = np.load(args.oof_probs)
    names = [path.stem for path, _ in samples]
    if list(archive.files) != names:
        raise ValueError("OOF probability key order does not match sorted train samples")

    fold_payload = json.loads(Path(args.fold_results).read_text(encoding="utf-8"))
    folds = [set(row["heldout_specimen_groups"]) for row in fold_payload["folds"]]
    sample_groups = [specimen_id(image_path) for image_path, _ in samples]
    flat_folds = [group for fold in folds for group in fold]
    if len(flat_folds) != len(set(flat_folds)) or set(flat_folds) != set(sample_groups):
        raise ValueError("fold groups do not form a disjoint cover of train specimen IDs")

    rng = np.random.default_rng(args.seed)
    train_records = []
    for index, (image_path, mask_path) in enumerate(_local_progress(samples, 'model320_global_context_gate')):
        probability = archive[image_path.stem].astype(np.float32)
        target = decode_mask(Image.open(mask_path).convert("RGB")).astype(np.uint8)
        pixel, context, prediction = feature_maps(image_path, probability)
        selected = choose_binary_pixels(target, prediction, args.samples_per_class, rng)
        if selected is None:
            continue
        indices, labels = selected
        train_records.append({
            "name": image_path.name, "group": sample_groups[index],
            "pixel_x": pixel[indices],
            "context_x": np.concatenate([
                pixel[indices], np.broadcast_to(context, (len(indices), len(context)))
            ], axis=1).astype(np.float32),
            "labels": labels,
        })
        if args.verbose:
            print(f"sampled {image_path.name}: {len(indices)} pixels", flush=True)

    result_records = []
    fold_summaries = []
    for fold_index, heldout_groups in enumerate(folds):
        train_rows = [row for row in train_records if row["group"] not in heldout_groups]
        test_indices = [i for i, group in enumerate(sample_groups) if group in heldout_groups]
        if len(train_rows) < 2 or not test_indices:
            raise ValueError(f"empty meta fold {fold_index + 1}")
        x_pixel = np.concatenate([row["pixel_x"] for row in train_rows])
        x_context = np.concatenate([row["context_x"] for row in train_rows])
        y = np.concatenate([row["labels"] for row in train_rows])
        pixel_model = fit_logistic(x_pixel, y, args.l2)
        context_model = fit_logistic(x_context, y, args.l2)

        fold_scores = {"baseline": [], "pixel_only": [], "global_context": []}
        for index in test_indices:
            image_path, mask_path = samples[index]
            probability = archive[image_path.stem].astype(np.float32)
            target = decode_mask(Image.open(mask_path).convert("RGB")).astype(np.uint8)
            pixel, context, base_prediction = feature_maps(image_path, probability)
            eligible = np.flatnonzero(np.isin(base_prediction.reshape(-1), (0, 3)))
            if len(eligible):
                pixel_score = predict_binary(pixel_model, pixel[eligible])
                context_x = np.concatenate([
                    pixel[eligible],
                    np.broadcast_to(context, (len(eligible), len(context))),
                ], axis=1).astype(np.float32)
                context_score = predict_binary(context_model, context_x)
            else:
                pixel_score = np.empty(0, dtype=np.float64)
                context_score = np.empty(0, dtype=np.float64)

            corrected_pixel = base_prediction.reshape(-1).copy()
            corrected_context = base_prediction.reshape(-1).copy()
            corrected_pixel[eligible] = np.where(pixel_score >= args.threshold, 3, 0)
            corrected_context[eligible] = np.where(context_score >= args.threshold, 3, 0)
            base_score = image_iou(base_prediction, target)[0]
            pixel_iou, pixel_classes, _ = image_iou(
                corrected_pixel.reshape(target.shape), target)
            context_iou, context_classes, _ = image_iou(
                corrected_context.reshape(target.shape), target)
            row = {
                "image": image_path.name, "group": sample_groups[index],
                "baseline": base_score, "pixel_only": pixel_iou,
                "global_context": context_iou,
                "pixel_only_class_iou": pixel_classes,
                "global_context_class_iou": context_classes,
                "eligible_pixels": int(len(eligible)),
            }
            result_records.append(row)
            fold_scores["baseline"].append(base_score)
            fold_scores["pixel_only"].append(pixel_iou)
            fold_scores["global_context"].append(context_iou)

        fold_summaries.append({
            "fold": fold_index + 1,
            "train_images": len(train_rows),
            "heldout_images": len(test_indices),
            "baseline_miou": float(np.mean(fold_scores["baseline"])),
            "pixel_only_miou": float(np.mean(fold_scores["pixel_only"])),
            "global_context_miou": float(np.mean(fold_scores["global_context"])),
            "pixel_model_converged": pixel_model["converged"],
            "context_model_converged": context_model["converged"],
        })
        print(json.dumps(fold_summaries[-1], ensure_ascii=False), flush=True)

    if len(result_records) != len(samples):
        raise ValueError(f"OOF coverage mismatch: {len(result_records)} != {len(samples)}")
    summary = {
        "method": "Model297 OOF probabilities + specimen-group cross-fitted Al3Ni/Eutectic logistic expert",
        "metric": "per-image mean of four class IoUs; empty union=0",
        "n_images": len(samples), "n_specimen_groups": len(set(sample_groups)),
        "samples_per_class_per_image": args.samples_per_class,
        "decision_threshold": args.threshold,
        "baseline_oof_image_mean_miou": float(np.mean([r["baseline"] for r in result_records])),
        "pixel_only_oof_image_mean_miou": float(np.mean([r["pixel_only"] for r in result_records])),
        "global_context_oof_image_mean_miou": float(np.mean([r["global_context"] for r in result_records])),
        "pixel_only_vs_baseline": bootstrap_group_delta(result_records, "baseline", "pixel_only"),
        "global_context_vs_baseline": bootstrap_group_delta(result_records, "baseline", "global_context"),
        "global_context_vs_pixel_only": bootstrap_group_delta(result_records, "pixel_only", "global_context"),
        "folds": fold_summaries,
        "images": result_records,
        "elapsed_seconds": time.time() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "baseline_oof_image_mean_miou", "pixel_only_oof_image_mean_miou",
        "global_context_oof_image_mean_miou", "pixel_only_vs_baseline",
        "global_context_vs_baseline", "global_context_vs_pixel_only",
        "elapsed_seconds")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--oof-probs", required=True)
    parser.add_argument("--fold-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-class", type=int, default=384)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--l2", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=320)
    parser.add_argument("--verbose", action="store_true")
    main(parser.parse_args())
