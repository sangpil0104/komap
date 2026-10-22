#!/usr/bin/env python3
"""Search a Model342 probability expert on top of the Model341 route.

The fixed baseline is Model341's already selected Model339+Model329 route.
Model342 probabilities are a train-OOF-fitted class-wise Model322/Model329
blend. A second class-wise gamma grid is evaluated, and route selection is
audited by holding out one specimen group at a time on valid. CPU-only.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np

import model341_model329_joint_route_search as model341


GAMMAS = (0.0, 0.25, 0.50, 0.75, 1.0)
CLASS_NAMES = ("Al3Ni", "Primary Si", "Al", "Eutectic Si")
# Model342 records gamma in [Al, Al3Ni, Primary Si, Eutectic Si] order.
MODEL342_GAMMA_MAIN_ORDER = (1.0, 0.25, 0.50, 0.75)


def image_group(name: str) -> str:
    match = re.match(r"(A\d+)_", Path(name).name)
    if not match:
        raise ValueError(f"cannot parse specimen group from {name}")
    return match.group(1)


def class_iou(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    matrix = np.bincount(target.astype(np.int64).ravel() * 4 +
                         prediction.astype(np.int64).ravel(), minlength=16).reshape(4, 4)
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(0) + matrix.sum(1) - intersection
    return np.divide(intersection, union, out=np.zeros(4, dtype=np.float64), where=union > 0)


def build_model342_probabilities(base: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    if base.shape != candidate.shape or base.ndim != 3 or base.shape[0] != 4:
        raise ValueError(f"expected matching 4xHxW archives, got {base.shape}/{candidate.shape}")
    result = np.empty_like(base, dtype=np.float32)
    for class_id, gamma in enumerate(MODEL342_GAMMA_MAIN_ORDER):
        result[class_id] = (1.0 - gamma) * base[class_id] + gamma * candidate[class_id]
    return result


def candidate_prediction(row: dict, gamma342: tuple[float, ...],
                         base_route: dict, gamma329: tuple[float, ...],
                         guide_weights: list[np.ndarray]) -> np.ndarray:
    route = model341.route
    mixed = route.make_mixed(row, base_route)
    for class_id, gamma in enumerate(gamma329):
        if gamma:
            mixed = route.replace_classes(mixed, row["sources"]["329"], (class_id,), gamma)
    for class_id, gamma in enumerate(gamma342):
        if gamma:
            mixed = route.replace_classes(mixed, row["sources"]["342"], (class_id,), gamma)
    return route.predict(row["item"], mixed, base_route, guide_weights, exact=True)


def grouped_cv(scores: np.ndarray, names: list[str], seed: int,
               bootstrap_draws: int) -> dict:
    # Route 0 is the unchanged Model341 reference.
    reference = scores[0]
    group_to_indices: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        group_to_indices.setdefault(image_group(name), []).append(index)

    selected = np.empty(len(names), dtype=np.int64)
    fold_choices = []
    for held_group, held_indices in group_to_indices.items():
        train_indices = [index for group, indices in group_to_indices.items()
                         if group != held_group for index in indices]
        route_index = int(np.argmax(scores[:, train_indices].mean(axis=1)))
        selected[held_indices] = route_index
        fold_choices.append({
            "heldout_specimen": held_group,
            "selected_gamma342": list(SETTINGS[route_index]),
        })

    held_scores = scores[selected, np.arange(len(names))]
    delta = held_scores - reference
    rng = np.random.default_rng(seed)
    group_names = list(group_to_indices)
    bootstrap = np.empty(bootstrap_draws, dtype=np.float64)
    for draw in range(bootstrap_draws):
        sampled = rng.choice(group_names, size=len(group_names), replace=True)
        indices = [index for group in sampled for index in group_to_indices[str(group)]]
        bootstrap[draw] = delta[indices].mean() * 100.0

    return {
        "reference": "Model341 fixed strict-best route",
        "image_mean_delta_pp": float(delta.mean() * 100.0),
        "positive_images": int((delta > 0).sum()),
        "image_count": len(names),
        "specimen_group_count": len(group_to_indices),
        "group_bootstrap_95ci_delta_pp": [
            float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "folds": fold_choices,
    }


SETTINGS: list[tuple[float, ...]] = []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--max-routes", type=int, default=0,
                        help="prefix limit for smoke validation; 0 evaluates all 625 settings")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    global SETTINGS
    SETTINGS = list(itertools.product(GAMMAS, repeat=4))
    root = args.root
    outputs = root / "outputs"
    output = args.output or outputs / "model343_model342_into_341_route/results.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output}")
    if args.max_routes:
        if not 2 <= args.max_routes <= len(SETTINGS):
            raise ValueError("--max-routes must be between 2 and 625")
        SETTINGS = SETTINGS[:args.max_routes]

    model339_result = json.loads(
        (outputs / "model339_joint_gamma_route_search/results.json").read_text(encoding="utf-8"))
    model341_result = json.loads(
        (outputs / "model341_model329_joint_route_search/results.json").read_text(encoding="utf-8"))
    base_route = model339_result["best_image_empty_zero"]
    baseline_record = model341_result["best_image_empty_zero"]
    gamma329 = tuple(float(value) for value in baseline_record["gamma329"])
    if len(gamma329) != 4:
        raise ValueError("Model341 strict-best record has no four-class Model329 gamma")

    route = model341.route
    source_paths = {
        "195": (outputs / "model195_native_resume/valid_probs.npz", False),
        "217": (outputs / "model220_native_tent_extended/valid_probs_06.npz", False),
        "269": (outputs / "model269_fullstats_1x_extended/valid_probs.npz", False),
        "322": (outputs / "model322_team_b4_group_oof/valid_ensemble/probabilities.npz", True),
        "326": (outputs / "model326_hybrid_fullstats_group_oof/valid_ensemble/probabilities.npz", True),
        "329": (outputs / "model329_fullstats_width4000_uniform_group_oof/valid_ensemble/probabilities.npz", True),
    }
    archives = {}
    names = None
    for source, (path, team_order) in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        archive_names, values = route.load_archive(path, team_order=team_order)
        if names is None:
            names = archive_names
        elif set(archive_names) != set(names):
            raise ValueError(f"validation image-key mismatch for Model{source}")
        archives[source] = values
    with (outputs / "model183_metric_calibration/entries.pkl").open("rb") as stream:
        entries = pickle.load(stream)
    if names is None or len(entries) != len(names) or len(names) != 20:
        raise ValueError("expected the 20 validation images and their calibrated entries")

    rows = []
    for index, name in enumerate(names):
        item = entries[index]
        sources = {source: values[name] for source, values in archives.items()}
        sources["b4mean"] = 0.5 * (sources["322"] + sources["326"])
        sources["342"] = build_model342_probabilities(sources["322"], sources["329"])
        rows.append({
            "index": index,
            "item": item,
            "base": route.undo_model181_primary_blend(item),
            "sources": {"base": None, "181": item["extra"].astype(np.float32), **sources},
        })

    scores = np.empty((len(SETTINGS), len(rows)), dtype=np.float64)
    class_scores = np.empty((len(SETTINGS), len(rows), 4), dtype=np.float64)
    start = time.monotonic()
    for image_index, row in enumerate(rows):
        guide_weights = route.bilateral_weights(row["item"]["z"])
        target = row["item"]["target"]
        for route_index, gamma342 in enumerate(SETTINGS):
            prediction = candidate_prediction(row, gamma342, base_route, gamma329,
                                              guide_weights)
            ious = class_iou(prediction, target)
            class_scores[route_index, image_index] = ious
            scores[route_index, image_index] = ious.mean()
        print(f"scored image {image_index + 1}/{len(rows)}; "
              f"elapsed {(time.monotonic() - start) / 60.0:.1f} min", flush=True)

    # Assert exact parity before trusting this extension's baseline comparison.
    baseline_scores = np.asarray(baseline_record["image_scores"], dtype=np.float64)
    if baseline_scores.shape != (len(rows),) or not np.allclose(
            scores[0], baseline_scores, atol=1e-10, rtol=0):
        raise AssertionError("gamma342=0 failed to reproduce Model341's recorded image scores")

    best_index = int(np.argmax(scores.mean(axis=1)))
    cv = grouped_cv(scores, names, seed=343, bootstrap_draws=args.bootstrap)
    result = {
        "model": 343,
        "metric": "image-wise four-class mIoU; empty union=0",
        "baseline": "Model341 fixed strict-best route",
        "model341_gamma329_class_order": list(model341.CLASS_NAMES),
        "model341_gamma329": list(gamma329),
        "model342_source_gamma_class_order": ["Al", "Al3Ni", "Primary Si", "Eutectic Si"],
        "model342_source_gamma": [0.50, 1.00, 0.25, 0.75],
        "model342_source_gamma_main_order": list(MODEL342_GAMMA_MAIN_ORDER),
        "class_order": list(CLASS_NAMES),
        "gamma_grid": list(GAMMAS),
        "candidate_count": len(SETTINGS),
        "selection_note": "valid route selection is exploratory; use grouped held-out result, not raw maximum, for adoption",
        "baseline_valid_image_mean_miou_empty0": float(scores[0].mean()),
        "best_image_empty0": {
            "gamma342": list(SETTINGS[best_index]),
            "image_mean_miou": float(scores[best_index].mean()),
            "class_mean_iou": class_scores[best_index].mean(axis=0).tolist(),
            "image_scores": scores[best_index].tolist(),
        },
        "grouped_cv_vs_model341": cv,
        "settings_image_mean_miou_empty0": scores.mean(axis=1).tolist(),
        "settings_class_mean_iou_empty0": class_scores.mean(axis=1).tolist(),
        "image_names": names,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "candidate_count": len(SETTINGS),
        "best_gamma342": result["best_image_empty0"]["gamma342"],
        "baseline_valid_image_mean_miou_empty0": result["baseline_valid_image_mean_miou_empty0"],
        "best_valid_image_mean_miou_empty0": result["best_image_empty0"]["image_mean_miou"],
        "grouped_cv_vs_model341": {
            key: cv[key] for key in ("image_mean_delta_pp", "positive_images",
                                     "group_bootstrap_95ci_delta_pp")},
        "output": str(output),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
