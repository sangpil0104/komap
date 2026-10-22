#!/usr/bin/env python3
"""Exact grouped-CV route search adding the completed Model329 branch.

Model339's class experts remain the base route. For each class, independently
blend in the new Model329 probability at gamma 0, .25, .5, .75, or 1.0. The
625-setting grid includes Model339 unchanged; a fixed Model338 route is also
included as a conservative comparison. CPU-only; no model inference/training.
"""
from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import itertools
import json
import multiprocessing as mp
import pickle
import re
import time
from pathlib import Path

import numpy as np

import model335_exhaustive_b4_route_search as route


GAMMAS = (0.0, 0.25, 0.50, 0.75, 1.0)
CLASS_NAMES = ("Al3Ni", "Primary Si", "Al", "Eutectic Si")
_WORKER_ROWS = None
_WORKER_SETTINGS = None


def build_settings(model338: dict, model339: dict) -> list[dict]:
    settings = [{
        "stage": "model338_reference",
        "candidate": "Model338 conservative reference",
        "base_route": model338,
        "gamma329": [0.0, 0.0, 0.0, 0.0],
    }]
    for gammas in itertools.product(GAMMAS, repeat=4):
        all_zero = all(value == 0.0 for value in gammas)
        settings.append({
            "stage": "model339_parent" if all_zero else "model341_model329_grid",
            "candidate": ("Model339 unchanged" if all_zero else
                          "Model339 base + Model329 gamma " +
                          ",".join(f"{value:.2f}" for value in gammas)),
            "base_route": model339,
            "gamma329": list(gammas),
        })
    if len(settings) != 626:
        raise AssertionError(f"expected 626 settings (Model338 + 625 grid), got {len(settings)}")
    return settings


def make_prediction(row: dict, setting: dict, guide_weights: list[np.ndarray]) -> np.ndarray:
    mixed = route.make_mixed(row, setting["base_route"])
    candidate = row["sources"]["329"]
    for class_id, gamma in enumerate(setting["gamma329"]):
        if gamma:
            mixed = route.replace_classes(mixed, candidate, (class_id,), gamma)
    return route.predict(row["item"], mixed, setting["base_route"],
                         guide_weights, exact=True)


def _init_worker(rows: list[dict], settings: list[dict]) -> None:
    global _WORKER_ROWS, _WORKER_SETTINGS
    _WORKER_ROWS = rows
    _WORKER_SETTINGS = settings


def _evaluate_image(image_index: int):
    row = _WORKER_ROWS[image_index]
    settings = _WORKER_SETTINGS
    count = len(settings)
    confusion = np.zeros((count, 4, 4), dtype=np.int64)
    scores = np.zeros(count, dtype=np.float64)
    scores_skip = np.zeros(count, dtype=np.float64)
    weights = route.bilateral_weights(row["item"]["z"])
    target = row["item"]["target"]
    for index, setting in enumerate(_local_progress(settings, 'model341_model329_joint_route_search')):
        prediction = make_prediction(row, setting, weights)
        matrix = np.bincount(target.reshape(-1) * 4 + prediction.reshape(-1),
                             minlength=16).reshape(4, 4)
        confusion[index] = matrix
        intersection = np.diag(matrix).astype(np.float64)
        union = matrix.sum(0) + matrix.sum(1) - intersection
        iou = intersection / np.maximum(union, 1.0)
        scores[index] = iou.mean()
        scores_skip[index] = iou[union > 0].mean()
        if (index + 1) % 200 == 0:
            print(f"image {row['index'] + 1}: routes {index + 1}/{count}", flush=True)
    return image_index, confusion, scores, scores_skip


def evaluate(rows: list[dict], settings: list[dict], workers: int):
    count, image_count = len(settings), len(rows)
    total_confusion = np.zeros((count, 4, 4), dtype=np.int64)
    image_confusion = np.zeros((count, image_count, 4, 4), dtype=np.int64)
    image_scores = np.zeros((count, image_count), dtype=np.float64)
    image_scores_skip = np.zeros_like(image_scores)
    start = time.monotonic()
    if workers > 1:
        context = mp.get_context("fork")
        with context.Pool(workers, initializer=_init_worker,
                          initargs=(rows, settings)) as pool:
            completed = pool.imap_unordered(_evaluate_image, range(image_count), chunksize=1)
            for done, (image_index, matrices, scores, scores_skip) in enumerate(_local_progress(completed, 'model341_model329_joint_route_search', total=image_count), 1):
                total_confusion += matrices
                image_confusion[:, image_index] = matrices
                image_scores[:, image_index] = scores
                image_scores_skip[:, image_index] = scores_skip
                print(f"completed image {done}/{image_count}; "
                      f"elapsed {(time.monotonic() - start) / 60:.1f} min", flush=True)
    else:
        _init_worker(rows, settings)
        for image_index in _local_progress(range(image_count), 'model341_model329_joint_route_search'):
            _, matrices, scores, scores_skip = _evaluate_image(image_index)
            total_confusion += matrices
            image_confusion[:, image_index] = matrices
            image_scores[:, image_index] = scores
            image_scores_skip[:, image_index] = scores_skip
            print(f"completed image {image_index + 1}/{image_count}; "
                  f"elapsed {(time.monotonic() - start) / 60:.1f} min", flush=True)

    records = []
    for index, setting in enumerate(_local_progress(settings, 'model341_model329_joint_route_search')):
        matrix = total_confusion[index]
        per_image = image_confusion[index]
        intersection = np.diagonal(per_image, axis1=1, axis2=2).astype(np.float64)
        union = per_image.sum(1) + per_image.sum(2) - intersection
        image_class_iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0).mean(0)
        records.append({
            "stage": setting["stage"],
            "candidate": setting["candidate"],
            "base_model": (338 if setting["stage"] == "model338_reference" else 339),
            "gamma329": setting["gamma329"],
            "image_mean_miou": float(image_scores[index].mean()),
            "image_mean_miou_exclude": float(image_scores_skip[index].mean()),
            "image_miou_compat": float(image_class_iou.mean()),
            "image_scores": image_scores[index].tolist(),
            "image_scores_exclude": image_scores_skip[index].tolist(),
            "confusion": matrix.tolist(),
        })
    return records, image_confusion


def grouped_cv(records: list[dict], names: list[str], score_key: str,
               reference_stage: str, seed: int) -> dict:
    reference = next(row for row in records if row["stage"] == reference_stage)
    reference_scores = np.asarray(reference[score_key], dtype=np.float64)
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        match = re.match(r"(A\d+)_", Path(name).name)
        if not match:
            raise ValueError(f"cannot parse specimen group from {name}")
        groups.setdefault(match.group(1), []).append(index)
    delta = np.zeros(len(names), dtype=np.float64)
    folds = []
    for held_group, held_indices in groups.items():
        train_indices = [index for group, indices in groups.items() if group != held_group
                         for index in indices]
        selected = max(records, key=lambda row: float(
            np.asarray(row[score_key], dtype=np.float64)[train_indices].mean()))
        selected_scores = np.asarray(selected[score_key], dtype=np.float64)
        delta[held_indices] = selected_scores[held_indices] - reference_scores[held_indices]
        folds.append({"heldout_specimen": held_group,
                      "selected_route": selected["candidate"]})
    rng = np.random.default_rng(seed)
    group_names = list(groups)
    bootstrap = np.empty(10000, dtype=np.float64)
    for draw in range(len(bootstrap)):
        sampled = rng.choice(group_names, size=len(group_names), replace=True)
        indices = [index for group in sampled for index in groups[str(group)]]
        bootstrap[draw] = delta[indices].mean() * 100.0
    return {
        "reference": reference["candidate"],
        "image_mean_delta_pp": float(delta.mean() * 100.0),
        "positive_images": int((delta > 0).sum()),
        "image_count": len(names),
        "specimen_group_count": len(groups),
        "group_bootstrap_95ci_delta_pp": [
            float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "folds": folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--candidate-npz", type=Path, default=None,
                        help="defaults to Model329's valid_ensemble archive")
    parser.add_argument("--candidate-order", choices=("team", "main"), default="team")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-routes", type=int, default=0,
                        help="prefix limit for smoke validation; 0 evaluates all settings")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root
    candidate_path = args.candidate_npz or root / (
        "outputs/model329_fullstats_width4000_uniform_group_oof/"
        "valid_ensemble/probabilities.npz")
    output = args.output or root / "outputs/model341_model329_joint_route_search/results.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output}")

    model339 = json.loads((root / "outputs/model339_joint_gamma_route_search/results.json")
                          .read_text(encoding="utf-8"))
    model338_config = model339["fixed_experts_from_model338_strict"]
    model339_config = model339["best_image_empty_zero"]
    with (root / "outputs/model183_metric_calibration/entries.pkl").open("rb") as stream:
        entries = pickle.load(stream)
    source_paths = {
        "195": (root / "outputs/model195_native_resume/valid_probs.npz", False),
        "217": (root / "outputs/model220_native_tent_extended/valid_probs_06.npz", False),
        "269": (root / "outputs/model269_fullstats_1x_extended/valid_probs.npz", False),
        "322": (root / "outputs/model322_team_b4_group_oof/valid_ensemble/probabilities.npz", True),
        "326": (root / "outputs/model326_hybrid_fullstats_group_oof/valid_ensemble/probabilities.npz", True),
        "329": (candidate_path, args.candidate_order == "team"),
    }
    archives = {}
    names = None
    for source, (path, team_order) in source_paths.items():
        archive_names, values = route.load_archive(path, team_order=team_order)
        if names is None:
            names = archive_names
        elif set(archive_names) != set(names):
            raise ValueError(f"validation image-key mismatch for Model{source}")
        archives[source] = values
    if names is None or len(entries) != len(names):
        raise ValueError("entry count and validation probability image count differ")
    for name in names:
        candidate = archives["329"][name]
        if candidate.shape != archives["322"][name].shape:
            raise ValueError(f"Model329 probability shape mismatch for {name}")
        if not np.allclose(candidate.sum(axis=0), 1.0, atol=3e-3):
            raise ValueError(f"Model329 probability sum check failed for {name}")
    archives["b4mean"] = {
        name: 0.5 * (archives["322"][name] + archives["326"][name])
        for name in names
    }
    rows = []
    for index, name in enumerate(names):
        item = entries[index]
        rows.append({
            "index": index,
            "item": item,
            "base": route.undo_model181_primary_blend(item),
            "sources": {
                "base": None,
                "181": item["extra"].astype(np.float32),
                **{source: values[name] for source, values in archives.items()},
            },
        })

    settings = build_settings(model338_config, model339_config)
    if args.max_routes:
        if args.max_routes < 2 or args.max_routes > len(settings):
            raise ValueError("--max-routes must be between 2 and 626")
        settings = settings[:args.max_routes]
    records, _ = evaluate(rows, settings, args.workers)
    strict_best = max(records, key=lambda row: row["image_mean_miou"])
    skip_best = max(records, key=lambda row: row["image_mean_miou_exclude"])
    cv338 = grouped_cv(records, names, "image_scores", "model338_reference", 341)
    cv339 = grouped_cv(records, names, "image_scores", "model339_parent", 342)
    result = {
        "model": 341,
        "metric": "Exact per-image 4-class mIoU; empty union=0 local empty=0 selection",
        "candidate_count": len(records),
        "full_candidate_count": 626,
        "gamma_grid": list(GAMMAS),
        "class_order": list(CLASS_NAMES),
        "best_image_empty_zero": strict_best,
        "best_image_empty_skipped": skip_best,
        "grouped_cv_vs_model338": cv338,
        "grouped_cv_vs_model339": cv339,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "candidate_count": len(records),
        "strict_best": strict_best["image_mean_miou"],
        "strict_route": strict_best["candidate"],
        "skip_best": skip_best["image_mean_miou_exclude"],
        "grouped_cv_vs_model338": {
            key: cv338[key] for key in ("image_mean_delta_pp", "positive_images",
                                        "group_bootstrap_95ci_delta_pp")},
        "grouped_cv_vs_model339": {
            key: cv339[key] for key in ("image_mean_delta_pp", "positive_images",
                                        "group_bootstrap_95ci_delta_pp")},
        "output": str(output),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
