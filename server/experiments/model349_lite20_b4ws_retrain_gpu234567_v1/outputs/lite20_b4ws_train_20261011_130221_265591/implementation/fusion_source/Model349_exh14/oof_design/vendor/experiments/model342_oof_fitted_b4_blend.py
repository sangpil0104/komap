#!/usr/bin/env python3
"""Fit a class-wise Model322/329 probability blend on train-only grouped OOF.

The blend weights are selected without looking at validation labels. The chosen
weights are then applied once to valid probabilities and scored with the
competition's image-wise four-class mIoU, assigning zero to empty unions.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
from pathlib import Path

import numpy as np


CLASS_NAMES = ("Al", "Al3Ni", "Primary Si", "Eutectic Si")


def image_group(name: str) -> str:
    match = re.match(r"(A\d+)_", Path(name).name)
    if not match:
        raise ValueError(f"cannot parse specimen group from {name}")
    return match.group(1)


def class_iou(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    if prediction.shape != target.shape:
        raise ValueError(f"shape mismatch: {prediction.shape} vs {target.shape}")
    matrix = np.bincount(target.astype(np.int64).ravel() * 4 +
                         prediction.astype(np.int64).ravel(), minlength=16).reshape(4, 4)
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(axis=0) + matrix.sum(axis=1) - intersection
    return np.divide(intersection, union, out=np.zeros(4, dtype=np.float64), where=union > 0)


def blended_prediction(base: np.ndarray, candidate: np.ndarray,
                       gamma: tuple[float, ...]) -> np.ndarray:
    if base.shape != candidate.shape or base.ndim != 3 or base.shape[0] != 4:
        raise ValueError(f"expected matching 4xHxW arrays, got {base.shape}/{candidate.shape}")
    scores = np.empty(base.shape, dtype=np.float32)
    for class_id, weight in enumerate(gamma):
        scores[class_id] = ((1.0 - weight) * base[class_id].astype(np.float32) +
                            weight * candidate[class_id].astype(np.float32))
    # Per-pixel probability renormalization cannot change argmax.
    return scores.argmax(axis=0).astype(np.uint8)


def score_grid(base: np.ndarray, candidate: np.ndarray, target: np.ndarray,
               settings: list[tuple[float, ...]]) -> tuple[np.ndarray, np.ndarray]:
    means = np.empty(len(settings), dtype=np.float64)
    per_class = np.empty((len(settings), 4), dtype=np.float64)
    for index, gamma in enumerate(settings):
        ious = class_iou(blended_prediction(base, candidate, gamma), target)
        per_class[index] = ious
        means[index] = ious.mean()
    return means, per_class


def grouped_cv(scores: np.ndarray, groups: list[str], seed: int,
               bootstrap_draws: int) -> dict:
    # Candidate 0 is Model322 unchanged; remaining gamma tuples are fixed in a
    # train-only grid and selected anew while holding out each specimen group.
    reference = scores[0]
    group_to_indices: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        group_to_indices.setdefault(group, []).append(index)
    selected = np.empty(len(groups), dtype=np.int64)
    choices = []
    for held_group, held_indices in group_to_indices.items():
        train_indices = [i for group, indices in group_to_indices.items()
                         if group != held_group for i in indices]
        selected_index = int(np.argmax(scores[:, train_indices].mean(axis=1)))
        selected[held_indices] = selected_index
        choices.append({"heldout_specimen": held_group,
                        "selected_setting_index": selected_index})
    held_scores = scores[selected, np.arange(len(groups))]
    delta = held_scores - reference
    rng = np.random.default_rng(seed)
    group_names = list(group_to_indices)
    bootstrap = np.empty(bootstrap_draws, dtype=np.float64)
    for draw in range(bootstrap_draws):
        sampled = rng.choice(group_names, size=len(group_names), replace=True)
        indices = [index for group in sampled for index in group_to_indices[str(group)]]
        bootstrap[draw] = delta[indices].mean() * 100.0
    return {
        "reference": "Model322 (gamma 0 for all classes)",
        "image_mean_delta_pp": float(delta.mean() * 100.0),
        "positive_images": int((delta > 0).sum()),
        "image_count": len(groups),
        "specimen_group_count": len(group_to_indices),
        "group_bootstrap_95ci_delta_pp": [
            float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "fold_choices": choices,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--gammas", type=float, nargs="+", default=(0.0, 0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not args.gammas or any(gamma < 0 or gamma > 1 for gamma in args.gammas):
        raise ValueError("gamma candidates must be within [0, 1]")

    root = args.root
    code_dir = root / "model326_hybrid_fullstats/KoMaP"
    sys.path.insert(0, str(code_dir))
    from common import list_split, load_mask  # pylint: disable=import-outside-toplevel

    outputs = root / "outputs"
    parent_oof = outputs / "model322_team_b4_group_oof/train_oof_probs.npz"
    candidate_oof = outputs / "model329_fullstats_width4000_uniform_group_oof/train_oof_probs.npz"
    parent_valid = outputs / "model322_team_b4_group_oof/valid_ensemble/probabilities.npz"
    candidate_valid = outputs / "model329_fullstats_width4000_uniform_group_oof/valid_ensemble/probabilities.npz"
    output = args.output or outputs / "model342_oof_fitted_b4_blend/results.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output}")
    for path in (parent_oof, candidate_oof, parent_valid, candidate_valid):
        if not path.is_file():
            raise FileNotFoundError(path)

    pairs = []
    for fold in range(1, 5):
        pairs.extend(list_split(str(root / "data_model322_group_cv"), f"oof_fold{fold}"))
    pairs.sort(key=lambda pair: Path(pair[0]).name)
    names = [Path(image).name.replace("_image.png", "") for image, _ in pairs]
    if len(names) != 70 or len(set(names)) != len(names):
        raise ValueError(f"expected 70 unique train OOF images, found {len(names)}")
    groups = [image_group(name) for name in names]
    settings = list(itertools.product(args.gammas, repeat=4))
    scores = np.empty((len(settings), len(names)), dtype=np.float64)
    class_scores = np.empty((len(settings), len(names), 4), dtype=np.float64)

    start = time.monotonic()
    with (np.load(parent_oof) as base_archive,
          np.load(candidate_oof) as candidate_archive):
        base_names, candidate_names = set(base_archive.files), set(candidate_archive.files)
        if base_names != set(names) or candidate_names != set(names):
            raise ValueError("OOF probability keys do not match the 70 held-out train images")
        for index, (name, (_, mask_path)) in enumerate(zip(names, pairs)):
            base = base_archive[name].astype(np.float32)
            candidate = candidate_archive[name].astype(np.float32)
            if base.shape != candidate.shape or base.shape[0] != 4:
                raise ValueError(f"probability shape mismatch for {name}: {base.shape}/{candidate.shape}")
            target = load_mask(mask_path)
            image_scores, image_class_scores = score_grid(base, candidate, target, settings)
            scores[:, index] = image_scores
            class_scores[:, index] = image_class_scores
            if (index + 1) % 10 == 0 or index + 1 == len(names):
                print(f"scored OOF images {index + 1}/{len(names)}; "
                      f"elapsed {(time.monotonic() - start) / 60.0:.1f} min", flush=True)

    best_index = int(np.argmax(scores.mean(axis=1)))
    cv = grouped_cv(scores, groups, seed=342, bootstrap_draws=args.bootstrap)
    valid_pairs = list_split(str(root / "data"), "valid")
    valid_scores: dict[str, list[float]] = {"Model322": [], "Model329": [], "Model342": []}
    valid_class_scores: dict[str, list[list[float]]] = {
        "Model322": [], "Model329": [], "Model342": []}
    valid_rows = []
    with (np.load(parent_valid) as base_archive,
          np.load(candidate_valid) as candidate_archive):
        base_names, candidate_names = set(base_archive.files), set(candidate_archive.files)
        expected_names = {Path(image).name.replace("_image.png", "")
                          for image, _ in valid_pairs}
        if base_names != expected_names or candidate_names != expected_names:
            raise ValueError("valid probability keys do not match the valid image split")
        for image_path, mask_path in valid_pairs:
            if mask_path is None:
                raise ValueError("valid masks are required to score Model342")
            name = Path(image_path).name.replace("_image.png", "")
            base = base_archive[name].astype(np.float32)
            candidate = candidate_archive[name].astype(np.float32)
            target = load_mask(mask_path)
            predictions = {
                "Model322": base.argmax(axis=0),
                "Model329": candidate.argmax(axis=0),
                "Model342": blended_prediction(base, candidate, settings[best_index]),
            }
            row = {"image": name}
            for model_name, prediction in predictions.items():
                ious = class_iou(prediction, target)
                score = float(ious.mean() * 100.0)
                valid_scores[model_name].append(score)
                valid_class_scores[model_name].append((ious * 100.0).tolist())
                row[model_name] = {
                    "image_miou_empty0": score,
                    "class_iou_empty0": (ious * 100.0).tolist(),
                }
            valid_rows.append(row)

    result = {
        "model": 342,
        "metric": "image-wise mean of four class IoUs; empty union=0",
        "class_order": list(CLASS_NAMES),
        "data_selection": "fit gamma only on exactly-once train grouped-OOF predictions; valid labels used only for final scoring",
        "gamma_grid": list(args.gammas),
        "candidate_count": len(settings),
        "selected_gamma_by_class": dict(zip(CLASS_NAMES, settings[best_index])),
        "selected_train_oof_miou_empty0": float(scores[best_index].mean() * 100.0),
        "model322_train_oof_miou_empty0": float(scores[0].mean() * 100.0),
        "grouped_cv_vs_model322": cv,
        "valid_image_mean_miou_empty0": {
            name: float(np.mean(values)) for name, values in valid_scores.items()},
        "valid_class_mean_iou_empty0": {
            name: np.mean(np.asarray(values), axis=0).tolist()
            for name, values in valid_class_scores.items()},
        "valid_images": valid_rows,
        "settings_train_oof_mean_miou_empty0": [
            float(scores[index].mean() * 100.0) for index in range(len(settings))],
        "settings_train_oof_class_mean_iou_empty0": [
            np.mean(class_scores[index], axis=0).tolist() for index in range(len(settings))],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "candidate_count": len(settings),
        "selected_gamma_by_class": result["selected_gamma_by_class"],
        "selected_train_oof_miou_empty0": result["selected_train_oof_miou_empty0"],
        "grouped_cv_vs_model322": {
            key: cv[key] for key in ("image_mean_delta_pp", "positive_images",
                                     "group_bootstrap_95ci_delta_pp")},
        "valid_image_mean_miou_empty0": result["valid_image_mean_miou_empty0"],
        "output": str(output),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
