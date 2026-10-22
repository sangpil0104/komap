#!/usr/bin/env python3
"""Fine-sweep the two active Model336 coefficients in the Model346 route.

Model346's Al3Ni and Primary Si coefficients stay at zero. Al is searched
from 0.65 to 1.00 and Eutectic Si from 0.00 to 0.50 in 0.025 increments.
The exact Model346 route is row zero and must pass score/class-mean parity
before the grouped leave-one-specimen-out comparison is accepted.
"""
from __future__ import annotations

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
from PIL import Image

HERE = Path(__file__).resolve().parent
for path in (HERE / "historical_scripts", HERE):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import model341_model329_joint_route_search as model341  # noqa: E402
import model343_model342_into_341_route as model343  # noqa: E402
import model346_model336_into_341_route as model346  # noqa: E402
from model1_unet_resnet34 import decode_mask  # noqa: E402
from model323_model270_branch_route_probe import canonical_image_key  # noqa: E402

CLASS_ORDER = ("Al3Ni", "Primary Si", "Al", "Eutectic Si")
AL_GAMMAS = tuple(float(value) for value in np.round(np.arange(0.65, 1.0001, 0.025), 3))
EUTECTIC_GAMMAS = tuple(float(value) for value in np.round(np.arange(0.00, 0.5001, 0.025), 3))
BASELINE_GAMMA336 = (0.0, 0.0, 1.0, 0.25)


def build_settings(max_routes: int = 0) -> list[tuple[float, ...]]:
    candidates = [(0.0, 0.0, al_gamma, eutectic_gamma)
                  for al_gamma, eutectic_gamma in itertools.product(
                      AL_GAMMAS, EUTECTIC_GAMMAS)]
    settings = [BASELINE_GAMMA336]
    settings.extend(candidate for candidate in candidates
                    if candidate != BASELINE_GAMMA336)
    if len(settings) != 315:
        raise AssertionError(f"expected 315 unique routes, got {len(settings)}")
    if max_routes:
        if not 2 <= max_routes <= len(settings):
            raise ValueError("--max-routes must be between 2 and 315")
        settings = settings[:max_routes]
    return settings


def load_rows(root: Path, expert_path: Path):
    outputs = root / "outputs"
    model339_result = json.loads(
        (outputs / "model339_joint_gamma_route_search/results.json")
        .read_text(encoding="utf-8"))
    model341_result = json.loads(
        (outputs / "model341_model329_joint_route_search/results.json")
        .read_text(encoding="utf-8"))
    model346_result = json.loads(
        (outputs / "model346_model336_into_341_route/results.json")
        .read_text(encoding="utf-8"))
    gamma336 = tuple(float(value) for value in model346_result["raw_best_gamma336"])
    if gamma336 != BASELINE_GAMMA336:
        raise ValueError(f"unexpected Model346 selected gamma: {gamma336}")

    base_route = model339_result["best_image_empty_zero"]
    baseline_record = model341_result["best_image_empty_zero"]
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
    for source, (path, team_order) in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        _, archives[source] = model341.route.load_archive(path, team_order=team_order)
    if not expert_path.is_file():
        raise FileNotFoundError(expert_path)
    _, archives["336"] = model341.route.load_archive(expert_path, team_order=True)

    names = sorted(archives["336"])
    if len(names) != 20:
        raise ValueError(f"expected 20 Model336 validation images, got {len(names)}")
    for source, values in archives.items():
        if set(values) != set(names):
            raise ValueError(f"validation image-key mismatch for Model{source}")
    mask_paths = sorted((root / "data/valid/masks").glob("*_mask.png"))
    mask_names = [canonical_image_key(path.name[:-len("_mask.png")])
                  for path in mask_paths]
    if mask_names != names:
        raise ValueError("sorted validation masks and probability names/order differ")
    with (outputs / "model183_metric_calibration/entries.pkl").open("rb") as stream:
        entries = pickle.load(stream)
    if len(entries) != len(names):
        raise ValueError("Model183 route-cache entry count differs from validation image count")

    rows = []
    for index, (name, item, mask_path) in enumerate(zip(names, entries, mask_paths)):
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        if not np.array_equal(item["target"], target):
            raise ValueError(f"cached target order mismatch for {mask_path.name}")
        for source, values in archives.items():
            probs = values[name]
            if probs.shape != (4, *target.shape):
                raise ValueError(f"Model{source} shape mismatch for {name}: {probs.shape}")
            if not np.allclose(probs.sum(axis=0), 1.0, atol=3e-3):
                raise ValueError(f"Model{source} probabilities are not normalized for {name}")
        sources = {source: values[name] for source, values in archives.items()}
        sources["b4mean"] = 0.5 * (sources["322"] + sources["326"])
        sources["342"] = sources["336"]
        rows.append({
            "index": index,
            "item": item,
            "base": model341.route.undo_model181_primary_blend(item),
            "sources": {"base": None, "181": item["extra"].astype(np.float32), **sources},
        })
    return rows, names, base_route, gamma329, model346_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--expert-probabilities", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--max-routes", type=int, default=0,
                        help="prefix limit for smoke tests; 0 evaluates all 315 routes")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    root, outputs = args.root, args.root / "outputs"
    expert_path = args.expert_probabilities or (
        outputs / "model336_hrnet_best/valid_eval/valid_probabilities.npz")
    output = args.output or outputs / "model350_model346_fine_gamma/results.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output}")
    rows, names, base_route, gamma329, model346_result = load_rows(root, expert_path)
    settings = build_settings(args.max_routes)
    scores = np.empty((len(settings), len(rows)), dtype=np.float64)
    class_scores = np.empty((len(settings), len(rows), 4), dtype=np.float64)

    # Route zero must reproduce the saved Model346 raw-best route.
    for image_index, row in enumerate(rows):
        guide = model341.route.bilateral_weights(row["item"]["z"])
        prediction = model346._candidate_prediction(
            row, settings[0], base_route, gamma329, guide, (0.0, 0.0, 0.0, 0.0))
        target = row["item"]["target"]
        matrix = np.bincount(target.astype(np.int64).ravel() * 4
                             + prediction.astype(np.int64).ravel(),
                             minlength=16).reshape(4, 4)
        intersection = np.diag(matrix).astype(np.float64)
        union = matrix.sum(0) + matrix.sum(1) - intersection
        iou = np.divide(intersection, union, out=np.zeros(4), where=union > 0)
        scores[0, image_index] = iou.mean()
        class_scores[0, image_index] = iou
    if not np.isclose(scores[0].mean(),
                      model346_result["raw_best_valid_image_mean_miou_empty0"],
                      atol=1e-10, rtol=0):
        raise AssertionError("Model346 baseline mIoU parity failed")
    if not np.allclose(class_scores[0].mean(axis=0),
                       model346_result["raw_best_class_mean_iou_empty0"],
                       atol=1e-10, rtol=0):
        raise AssertionError("Model346 baseline class-mean parity failed")
    baseline_scores = scores[0].copy()
    baseline_class_scores = class_scores[0].copy()
    print(f"Model346 exact baseline parity passed: {scores[0].mean() * 100:.6f}%",
          flush=True)

    model343.SETTINGS = settings
    started = time.monotonic()
    if args.workers > 1:
        context = mp.get_context("fork")
        with context.Pool(args.workers, initializer=model346._init_worker,
                          initargs=(rows, settings, base_route, gamma329,
                                    (0.0, 0.0, 0.0, 0.0))) as pool:
            for done, (image_index, image_scores, image_class_scores) in enumerate(
                    pool.imap_unordered(model346._evaluate_image,
                                        range(len(rows)), chunksize=1), 1):
                scores[:, image_index] = image_scores
                class_scores[:, image_index] = image_class_scores
                print(f"completed image {done}/{len(rows)}; elapsed "
                      f"{(time.monotonic() - started) / 60.0:.1f} min", flush=True)
    else:
        model346._init_worker(rows, settings, base_route, gamma329,
                              (0.0, 0.0, 0.0, 0.0))
        for image_index in range(len(rows)):
            _, image_scores, image_class_scores = model346._evaluate_image(image_index)
            scores[:, image_index] = image_scores
            class_scores[:, image_index] = image_class_scores
            print(f"completed image {image_index + 1}/{len(rows)}; elapsed "
                  f"{(time.monotonic() - started) / 60.0:.1f} min", flush=True)
    scores[0] = baseline_scores
    class_scores[0] = baseline_class_scores

    cv = model343.grouped_cv(scores, names, seed=350, bootstrap_draws=args.bootstrap)
    cv["reference"] = "Model346 fixed raw-best route"
    for fold in cv["folds"]:
        fold["selected_gamma336"] = fold.pop("selected_gamma342")
    gamma_counts = Counter(tuple(fold["selected_gamma336"]) for fold in cv["folds"])
    majority_gamma = list(gamma_counts.most_common(1)[0][0])
    majority_index = settings.index(tuple(majority_gamma))
    best = int(np.argmax(scores.mean(axis=1)))
    result = {
        "model": 350,
        "metric": "image-wise four-class mIoU; empty union=0",
        "baseline": "Model346 fixed raw-best route",
        "fixed_model346_gamma336": list(BASELINE_GAMMA336),
        "expert_model": 336,
        "expert": "Model336 HRNet-W48 + U-Net++; team-order probabilities converted to route order",
        "expert_probability_path": str(expert_path),
        "class_order": list(CLASS_ORDER),
        "gamma_search": {
            "Al3Ni": [0.0],
            "Primary Si": [0.0],
            "Al": list(AL_GAMMAS),
            "Eutectic Si": list(EUTECTIC_GAMMAS),
        },
        "candidate_count": len(settings),
        "baseline_valid_image_mean_miou_empty0": float(scores[0].mean()),
        "raw_best_gamma336": list(settings[best]),
        "raw_best_valid_image_mean_miou_empty0": float(scores[best].mean()),
        "raw_best_class_mean_iou_empty0": class_scores[best].mean(axis=0).tolist(),
        "grouped_cv_vs_model346": cv,
        "majority_group_cv_gamma336": majority_gamma,
        "majority_group_cv_gamma_valid_image_mean_miou_empty0": float(
            scores[majority_index].mean()),
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
