#!/usr/bin/env python3
"""Test whether the train-OOF-fitted Model344 expert improves Model341.

Model344's class-wise gamma is fixed from train OOF. A second class-wise grid
then adds that fixed expert to Model341's route; validation labels are used
only for the grouped held-out route audit and final score. CPU-only.
"""
from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import itertools
import json
import multiprocessing as mp
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
for path in (HERE / "historical_scripts", HERE):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import model341_model329_joint_route_search as model341  # noqa: E402
import model343_model342_into_341_route as model343  # noqa: E402


GAMMAS = (0.0, 0.25, 0.50, 0.75, 1.0)
CLASS_ORDER = ("Al3Ni", "Primary Si", "Al", "Eutectic Si")
TEAM_TO_MAIN = (1, 2, 0, 3)
_WORKER_ROWS: list[dict] = []
_WORKER_SETTINGS: list[tuple[float, ...]] = []
_WORKER_BASE_ROUTE: dict = {}
_WORKER_GAMMA329: tuple[float, ...] = ()


def _init_worker(rows: list[dict], settings: list[tuple[float, ...]],
                 base_route: dict, gamma329: tuple[float, ...]) -> None:
    global _WORKER_ROWS, _WORKER_SETTINGS, _WORKER_BASE_ROUTE, _WORKER_GAMMA329
    _WORKER_ROWS = rows
    _WORKER_SETTINGS = settings
    _WORKER_BASE_ROUTE = base_route
    _WORKER_GAMMA329 = gamma329
    model343.SETTINGS = settings


def _evaluate_image(image_index: int):
    row = _WORKER_ROWS[image_index]
    target = row["item"]["target"]
    guide = model341.route.bilateral_weights(row["item"]["z"])
    image_scores = np.zeros(len(_WORKER_SETTINGS), dtype=np.float64)
    image_class_scores = np.zeros((len(_WORKER_SETTINGS), 4), dtype=np.float64)
    for route_index in range(1, len(_WORKER_SETTINGS)):
        prediction = model343.candidate_prediction(
            row, _WORKER_SETTINGS[route_index], _WORKER_BASE_ROUTE,
            _WORKER_GAMMA329, guide)
        iou = model343.class_iou(prediction, target)
        image_scores[route_index] = iou.mean()
        image_class_scores[route_index] = iou
        if route_index % 200 == 0:
            print(f"image {row['index'] + 1}: routes {route_index + 1}/"
                  f"{len(_WORKER_SETTINGS)}", flush=True)
    return image_index, image_scores, image_class_scores


def load_model344_expert(root: Path, names: list[str], gamma_team: tuple[float, ...]) -> dict[str, np.ndarray]:
    route = model341.route
    outputs = root / "outputs"
    parent_path = outputs / "model322_team_b4_group_oof/valid_ensemble/probabilities.npz"
    control_path = outputs / "model333_b4_imagewise_lovasz_group_oof/control/valid_ensemble/probabilities.npz"
    parent_names, parent = route.load_archive(parent_path, team_order=True)
    control_names, control = route.load_archive(control_path, team_order=True)
    if set(parent_names) != set(names) or set(control_names) != set(names):
        raise ValueError("Model344 constituent valid probability keys do not match Model341 images")

    # Convert gamma from team order [Al, Al3Ni, Primary Si, Eutectic Si]
    # into the route's main order [Al3Ni, Primary Si, Al, Eutectic Si].
    gamma_main = tuple(gamma_team[index] for index in TEAM_TO_MAIN)
    expert = {}
    for name in names:
        if parent[name].shape != control[name].shape:
            raise ValueError(f"Model322/333 probability shape mismatch for {name}")
        mixed = np.empty_like(parent[name], dtype=np.float32)
        for class_id, gamma in enumerate(gamma_main):
            mixed[class_id] = ((1.0 - gamma) * parent[name][class_id]
                               + gamma * control[name][class_id])
        expert[name] = mixed
    return expert


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-routes", type=int, default=0,
                        help="prefix limit for smoke validation; 0 evaluates all 625 settings")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    root = args.root
    outputs = root / "outputs"
    output = args.output or outputs / "model345_model344_into_341_route/results.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output}")

    settings = list(itertools.product(GAMMAS, repeat=4))
    if args.max_routes:
        if not 2 <= args.max_routes <= len(settings):
            raise ValueError(f"--max-routes must be between 2 and {len(settings)}")
        settings = settings[:args.max_routes]

    model339_result = json.loads(
        (outputs / "model339_joint_gamma_route_search/results.json").read_text(encoding="utf-8"))
    model341_result = json.loads(
        (outputs / "model341_model329_joint_route_search/results.json").read_text(encoding="utf-8"))
    model344_oof = json.loads(
        (outputs / "model344_model333_control_blend/oof_results.json").read_text(encoding="utf-8"))
    gamma_team = tuple(float(value) for value in model344_oof["selected_gamma"])
    if len(gamma_team) != 4:
        raise ValueError("Model344 OOF result has no four-class gamma")

    route = model341.route
    baseline_record = model341_result["best_image_empty_zero"]
    base_route = model339_result["best_image_empty_zero"]
    gamma329 = tuple(float(value) for value in baseline_record["gamma329"])
    source_paths = {
        "195": (outputs / "model195_native_resume/valid_probs.npz", False),
        "217": (outputs / "model220_native_tent_extended/valid_probs_06.npz", False),
        "269": (outputs / "model269_fullstats_1x_extended/valid_probs.npz", False),
        "322": (outputs / "model322_team_b4_group_oof/valid_ensemble/probabilities.npz", True),
        "326": (outputs / "model326_hybrid_fullstats_group_oof/valid_ensemble/probabilities.npz", True),
        "329": (outputs / "model329_fullstats_width4000_uniform_group_oof/valid_ensemble/probabilities.npz", True),
    }
    archives: dict[str, dict[str, np.ndarray]] = {}
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
    if names is None or len(names) != 20:
        raise ValueError("expected all 20 validation images")
    expert344 = load_model344_expert(root, names, gamma_team)
    with (outputs / "model183_metric_calibration/entries.pkl").open("rb") as stream:
        entries = pickle.load(stream)
    if len(entries) != len(names):
        raise ValueError("Model183 entry count and valid probability image count differ")

    rows = []
    for index, name in enumerate(names):
        item = entries[index]
        sources = {source: values[name] for source, values in archives.items()}
        sources["b4mean"] = 0.5 * (sources["322"] + sources["326"])
        sources["342"] = expert344[name]  # shared candidate helper's expert slot
        rows.append({"index": index, "item": item,
                     "base": route.undo_model181_primary_blend(item),
                     "sources": {"base": None, "181": item["extra"].astype(np.float32),
                                 **sources}})

    scores = np.empty((len(settings), len(rows)), dtype=np.float64)
    class_scores = np.empty((len(settings), len(rows), 4), dtype=np.float64)
    baseline_scores = np.asarray(baseline_record["image_scores"], dtype=np.float64)
    if baseline_scores.shape != (len(rows),):
        raise ValueError("Model341 baseline and Model345 validation image counts differ")

    model343.SETTINGS = settings
    for image_index, row in enumerate(rows):
        guide = route.bilateral_weights(row["item"]["z"])
        prediction = model343.candidate_prediction(
            row, settings[0], base_route, gamma329, guide)
        iou = model343.class_iou(prediction, row["item"]["target"])
        scores[0, image_index] = iou.mean()
        class_scores[0, image_index] = iou
    if not np.allclose(scores[0], baseline_scores, atol=1e-10, rtol=0):
        raise AssertionError("gamma344=0 failed to reproduce Model341 baseline per image")
    baseline_class_scores = class_scores[0].copy()
    print(f"baseline parity passed: {scores[0].mean() * 100:.6f}%", flush=True)

    start = time.monotonic()
    if args.workers > 1:
        context = mp.get_context("fork")
        with context.Pool(args.workers, initializer=_init_worker,
                          initargs=(rows, settings, base_route, gamma329)) as pool:
            for done, (image_index, image_scores, image_class_scores) in enumerate(
                    _local_progress(pool.imap_unordered(_evaluate_image, range(len(rows)), chunksize=1), 'model345_model344_into_341_route', total=len(rows)), 1):
                scores[:, image_index] = image_scores
                class_scores[:, image_index] = image_class_scores
                print(f"completed image {done}/{len(rows)}; elapsed "
                      f"{(time.monotonic() - start) / 60.0:.1f} min", flush=True)
    else:
        _init_worker(rows, settings, base_route, gamma329)
        for image_index in _local_progress(range(len(rows)), 'model345_model344_into_341_route'):
            _, image_scores, image_class_scores = _evaluate_image(image_index)
            scores[:, image_index] = image_scores
            class_scores[:, image_index] = image_class_scores
            print(f"completed image {image_index + 1}/{len(rows)}; elapsed "
                  f"{(time.monotonic() - start) / 60.0:.1f} min", flush=True)
    scores[0] = baseline_scores
    class_scores[0] = baseline_class_scores

    cv = model343.grouped_cv(scores, names, seed=345, bootstrap_draws=args.bootstrap)
    best = int(np.argmax(scores.mean(axis=1)))
    for fold in cv["folds"]:
        fold["selected_gamma344"] = fold.pop("selected_gamma342")
    folds = Counter(tuple(fold["selected_gamma344"]) for fold in cv["folds"])
    majority_gamma = list(folds.most_common(1)[0][0])
    majority_index = settings.index(tuple(majority_gamma))
    result = {
        "model": 345,
        "metric": "image-wise four-class mIoU; empty union=0",
        "baseline": "Model341 fixed strict-best route",
        "expert": "Model344 fixed train-OOF-fitted blend of Model322 and Model333 CE+Dice control",
        "model344_gamma_team_order": list(gamma_team),
        "model344_gamma_main_order": [gamma_team[index] for index in TEAM_TO_MAIN],
        "class_order": list(CLASS_ORDER),
        "gamma_grid": list(GAMMAS),
        "candidate_count": len(settings),
        "baseline_valid_image_mean_miou_empty0": float(scores[0].mean()),
        "raw_best_gamma344": list(settings[best]),
        "raw_best_valid_image_mean_miou_empty0": float(scores[best].mean()),
        "raw_best_class_mean_iou_empty0": class_scores[best].mean(axis=0).tolist(),
        "grouped_cv_vs_model341": cv,
        "majority_group_cv_gamma344": majority_gamma,
        "majority_group_cv_gamma_valid_image_mean_miou_empty0": float(scores[majority_index].mean()),
        "settings_valid_image_mean_miou_empty0": scores.mean(axis=1).tolist(),
        "settings_valid_class_mean_iou_empty0": class_scores.mean(axis=1).tolist(),
        "image_names": names,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if not key.startswith("settings_")}, ensure_ascii=False, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
