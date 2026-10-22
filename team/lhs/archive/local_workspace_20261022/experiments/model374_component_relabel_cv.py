#!/usr/bin/env python3
"""Train a component relabeler on train-only grouped Model374 OOF predictions.

All features are available at inference. Ground truth supplies component labels
and evaluation scores only. The default run performs a two-way fold-1/2 check;
folds 3/4 are reserved for a later single holdout check.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import lightgbm as lgb
import numpy as np
from PIL import Image
from sklearn.model_selection import GroupKFold

from model374_train_oof_error_diagnostic import image_ious, mask_to_ids


N_CLASSES = 4
TAUS = (0.1, 0.2, 0.3)
GROUP_RE = re.compile(r"^(A\d+)")
FEATURE_NAMES = (
    "pred_class", "log_area", "log_width", "log_height", "bbox_extent", "touches_image_edge",
    "mean_margin", "mean_entropy", "gray_mean_z", "gray_std_z", "gray_gradient_over_std",
    "gray_local_variance", "gray_gradient", "neighbor_boundary_edges_log",
    *(f"mean_probability_{i}" for i in range(N_CLASSES)),
    *(f"std_probability_{i}" for i in range(N_CLASSES)),
    *(f"neighbor_edge_fraction_{i}" for i in range(N_CLASSES)),
    *(f"image_predicted_fraction_{i}" for i in range(N_CLASSES)),
)


@dataclass
class ImageRecord:
    image_id: str
    fold: int
    group: str
    features: np.ndarray
    targets: np.ndarray
    original_classes: np.ndarray
    areas: np.ndarray
    component_map: np.ndarray
    prediction: np.ndarray
    mask: np.ndarray


def component_features(probabilities: np.ndarray, gray: np.ndarray,
                       mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return features, GT-majority labels, source labels, and pixel component IDs."""
    height, width = gray.shape
    pred = probabilities.argmax(axis=0).astype(np.uint8)
    gray = gray.astype(np.float32)
    gray = (gray - float(gray.mean())) / max(float(gray.std()), 1e-5)
    gray_local_var = np.maximum(
        cv2.blur(gray * gray, (5, 5)) - cv2.blur(gray, (5, 5)) ** 2, 0.0
    )
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-7, 1.0)), axis=0)
    image_fractions = np.bincount(pred.ravel(), minlength=N_CLASSES).astype(np.float32) / pred.size
    component_map = np.full((height, width), -1, dtype=np.int32)
    feature_blocks = []
    target_blocks = []
    class_blocks = []
    area_blocks = []
    offset = 0

    for class_id in range(N_CLASSES):
        region = (pred == class_id).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
        if count <= 1:
            continue
        n = count - 1
        active = region != 0
        ids = labels[active].astype(np.int64)
        component_map[active] = (offset + ids - 1).astype(np.int32)
        area = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
        safe_area = np.maximum(area, 1.0)

        def means(values: np.ndarray) -> np.ndarray:
            sums = np.bincount(ids, weights=values[active].astype(np.float64), minlength=count)
            return (sums[1:] / safe_area).astype(np.float32)

        votes = np.bincount(ids * N_CLASSES + mask[active].astype(np.int64),
                            minlength=count * N_CLASSES).reshape(count, N_CLASSES)
        majority = votes.argmax(axis=1).astype(np.uint8)
        majority[votes[np.arange(count), class_id] == votes.max(axis=1)] = class_id
        target_blocks.append(majority[1:])
        class_blocks.append(np.full(n, class_id, dtype=np.uint8))
        area_blocks.append(area)

        other = probabilities.copy()
        other[class_id] = -np.inf
        margin = probabilities[class_id] - other.max(axis=0)
        gray_mean = means(gray)
        gray_var = np.maximum(means(gray * gray) - gray_mean * gray_mean, 0.0)

        # Count unlike-class 4-neighbor edges for every component at once.
        edge_votes = np.zeros((count, N_CLASSES), dtype=np.int64)
        for source, neighbor in (
            (labels[1:, :], pred[:-1, :]), (labels[:-1, :], pred[1:, :]),
            (labels[:, 1:], pred[:, :-1]), (labels[:, :-1], pred[:, 1:]),
        ):
            boundary = (source != 0) & (neighbor != class_id)
            edge_votes += np.bincount(
                source[boundary].astype(np.int64) * N_CLASSES + neighbor[boundary].astype(np.int64),
                minlength=count * N_CLASSES,
            ).reshape(count, N_CLASSES)
        edges = edge_votes[1:].sum(axis=1).astype(np.float32)
        edge_fractions = edge_votes[1:] / np.maximum(edges[:, None], 1.0)

        x = stats[1:, cv2.CC_STAT_LEFT].astype(np.float32)
        y = stats[1:, cv2.CC_STAT_TOP].astype(np.float32)
        box_w = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float32)
        box_h = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float32)
        columns = [
            np.full(n, class_id, np.float32), np.log1p(area), np.log1p(box_w), np.log1p(box_h),
            area / np.maximum(box_w * box_h, 1.0),
            ((x == 0) | (y == 0) | (x + box_w >= width) | (y + box_h >= height)).astype(np.float32),
            means(margin), means(entropy), gray_mean, np.sqrt(gray_var),
            means(gradient) / (1.0 + np.sqrt(gray_var)), means(gray_local_var),
            means(gradient), np.log1p(edges),
        ]
        prob_means = []
        prob_stds = []
        for phase_id in range(N_CLASSES):
            p = probabilities[phase_id]
            mean_p = means(p)
            prob_means.append(mean_p)
            prob_stds.append(np.sqrt(np.maximum(means(p * p) - mean_p * mean_p, 0.0)))
        columns.extend(prob_means)
        columns.extend(prob_stds)
        columns.extend(edge_fractions[:, phase_id].astype(np.float32)
                       for phase_id in range(N_CLASSES))
        columns.extend(np.full(n, image_fractions[phase_id], np.float32)
                       for phase_id in range(N_CLASSES))
        feature_blocks.append(np.stack(columns, axis=1))
        offset += n

    if np.any(component_map < 0):
        raise ValueError("component map did not cover every pixel")
    features = np.concatenate(feature_blocks)
    if features.shape[1] != len(FEATURE_NAMES) or not np.isfinite(features).all():
        raise ValueError(f"invalid component feature matrix: {features.shape}")
    return (features, np.concatenate(target_blocks), np.concatenate(class_blocks),
            np.concatenate(area_blocks), component_map)


def load_records(cv_data: Path, oof_root: Path, image_dir: Path, folds: tuple[int, ...]) -> list[ImageRecord]:
    records = []
    seen = set()
    for fold in folds:
        masks = {p.name.removesuffix("_mask.png"): p
                 for p in (cv_data / f"oof_fold{fold}" / "masks").glob("*_mask.png")}
        archive_path = oof_root / f"fold{fold}_oof" / "valid_probabilities.npz"
        with np.load(archive_path, allow_pickle=False) as archive:
            if set(archive.files) != set(masks):
                raise ValueError(f"fold {fold}: mask and probability IDs differ")
            for image_id in sorted(masks):
                if image_id in seen:
                    raise ValueError(f"duplicate OOF image: {image_id}")
                seen.add(image_id)
                group_match = GROUP_RE.match(image_id)
                if not group_match:
                    raise ValueError(f"missing specimen ID: {image_id}")
                target = mask_to_ids(masks[image_id])
                probabilities = np.asarray(archive[image_id], np.float32)
                image_path = image_dir / f"{image_id}_image.png"
                gray = np.asarray(Image.open(image_path).convert("L"), np.float32)
                if probabilities.shape != (N_CLASSES, *target.shape) or gray.shape != target.shape:
                    raise ValueError(f"image shapes differ for {image_id}")
                if not np.isfinite(probabilities).all():
                    raise ValueError(f"non-finite probabilities: {image_id}")
                features, targets, source, areas, component_map = component_features(probabilities, gray, target)
                records.append(ImageRecord(image_id, fold, group_match.group(1), features, targets,
                                           source, areas, component_map,
                                           probabilities.argmax(axis=0).astype(np.uint8), target))
    return records


def fit(records: list[ImageRecord]) -> lgb.LGBMClassifier:
    features = np.concatenate([r.features for r in records])
    targets = np.concatenate([r.targets for r in records])
    weights = np.concatenate([np.sqrt(r.areas) for r in records])
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=N_CLASSES, num_leaves=15,
        learning_rate=0.05, n_estimators=300, min_child_samples=50,
        colsample_bytree=0.8, reg_lambda=1.0, verbosity=-1,
        random_state=374, n_jobs=4,
    )
    model.fit(features, targets, sample_weight=weights, feature_name=list(FEATURE_NAMES))
    if set(model.classes_) != set(range(N_CLASSES)):
        raise ValueError(f"training subset lacks a phase: {model.classes_}")
    return model


def predict_map(record: ImageRecord, model: lgb.LGBMClassifier, tau: float) -> np.ndarray:
    probabilities = model.predict_proba(record.features)
    source = record.original_classes.astype(np.int64)
    proposal = probabilities.argmax(axis=1).astype(np.uint8)
    margin = probabilities[np.arange(len(source)), proposal] - probabilities[np.arange(len(source)), source]
    chosen = np.where((proposal != source) & (margin >= tau), proposal, source).astype(np.uint8)
    return chosen[record.component_map]


def score(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(image_ious(pred, target)))


def choose_tau_inner(records: list[ImageRecord]) -> tuple[float, dict[str, float]]:
    groups = np.asarray([r.group for r in records])
    if len(set(groups)) < 3:
        raise ValueError("need at least 3 specimen groups for inner CV")
    splitter = GroupKFold(n_splits=3)
    scores = {tau: [] for tau in TAUS}
    indices = np.arange(len(records))
    for train_idx, val_idx in splitter.split(indices, groups=groups):
        train = [records[i] for i in train_idx]
        val = [records[i] for i in val_idx]
        if set(r.group for r in train) & set(r.group for r in val):
            raise ValueError("specimen leakage in inner CV")
        model = fit(train)
        for record in val:
            for tau in TAUS:
                scores[tau].append(score(predict_map(record, model, tau), record.mask))
    means = {tau: float(np.mean(values)) for tau, values in scores.items()}
    chosen = max(TAUS, key=lambda tau: (means[tau], tau))
    return chosen, {str(tau): value for tau, value in means.items()}


def run(cv_data: Path, oof_root: Path, image_dir: Path) -> dict:
    records = load_records(cv_data, oof_root, image_dir, (1, 2))
    if len(records) != 36:
        raise ValueError(f"expected 36 fold-1/2 OOF images, found {len(records)}")
    outer = []
    for train_fold, eval_fold in ((1, 2), (2, 1)):
        train = [r for r in records if r.fold == train_fold]
        heldout = [r for r in records if r.fold == eval_fold]
        if set(r.group for r in train) & set(r.group for r in heldout):
            raise ValueError("specimen leakage across outer folds")
        tau, inner_scores = choose_tau_inner(train)
        model = fit(train)
        rows = []
        for record in heldout:
            base = score(record.prediction, record.mask)
            corrected = score(predict_map(record, model, tau), record.mask)
            rows.append({"image_id": record.image_id, "base": base, "corrected": corrected,
                         "gain_percentage_points": 100 * (corrected - base)})
        gains = np.asarray([r["gain_percentage_points"] for r in rows])
        outer.append({
            "train_fold": train_fold, "eval_fold": eval_fold,
            "tau_selected_by_inner_group_cv": tau,
            "inner_mean_scores_by_tau": inner_scores,
            "eval_images": len(rows),
            "eval_base_image_mean_miou_empty0": float(np.mean([r["base"] for r in rows])),
            "eval_corrected_image_mean_miou_empty0": float(np.mean([r["corrected"] for r in rows])),
            "eval_gain_percentage_points": float(gains.mean()),
            "eval_improved_images": int(np.count_nonzero(gains > 1e-9)),
            "eval_worsened_images": int(np.count_nonzero(gains < -1e-9)),
            "eval_worst_image_gain_percentage_points": float(gains.min()),
            "per_image": rows,
        })
    gain = sum(row["eval_gain_percentage_points"] * row["eval_images"] for row in outer) / 36
    return {
        "scope": "train-only Model374 OOF folds 1 and 2, outer fold holdout, inner specimen-group CV",
        "metric": "image mean of 4 class IoUs; empty union=0",
        "features": list(FEATURE_NAMES),
        "candidate_taus": list(TAUS),
        "outer_gain_percentage_points": gain,
        "outer": outer,
        "interpretation_guard": "No valid/test images were used; folds 3/4 remain unexamined holdout data.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-data", type=Path, required=True)
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.cv_data, args.oof_root, args.image_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("outer", "features")}, indent=2))
    for row in result["outer"]:
        print(json.dumps({k: v for k, v in row.items() if k != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
