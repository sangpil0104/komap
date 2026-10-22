#!/usr/bin/env python3
"""Route HRNet experts into Model341/346/349 anchor predictions.

The four Model336 blend coefficients are selected on the 20-image validation
set only for the raw leaderboard; grouped leave-one-specimen-out route
selection is reported separately. For Model349, ``--expert-id 347
--anchor-model346`` fixes Model346's selected Model336 gamma before blending
Model347. Model353 uses ``--expert-id 351 --anchor-model349`` to add Model351
D4 probabilities to the fixed Model349 route. The official metric uses
empty-union=0.
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
from PIL import Image

HERE = Path(__file__).resolve().parent
for path in (HERE / "historical_scripts", HERE):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import model341_model329_joint_route_search as model341  # noqa: E402
import model343_model342_into_341_route as model343  # noqa: E402
from model1_unet_resnet34 import decode_mask  # noqa: E402
from model323_model270_branch_route_probe import canonical_image_key  # noqa: E402

GAMMAS = (0.0, 0.25, 0.50, 0.75, 1.0)
CLASS_ORDER = ("Al3Ni", "Primary Si", "Al", "Eutectic Si")
_WORKER_ROWS: list[dict] = []
_WORKER_SETTINGS: list[tuple[float, ...]] = []
_WORKER_BASE_ROUTE: dict = {}
_WORKER_GAMMA329: tuple[float, ...] = ()
_WORKER_BASE_GAMMA336: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
_WORKER_BASE_GAMMA347: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)


def _init_worker(rows: list[dict], settings: list[tuple[float, ...]],
                 base_route: dict, gamma329: tuple[float, ...],
                 base_gamma336: tuple[float, ...],
                 base_gamma347: tuple[float, ...]) -> None:
    global _WORKER_ROWS, _WORKER_SETTINGS, _WORKER_BASE_ROUTE
    global _WORKER_GAMMA329, _WORKER_BASE_GAMMA336, _WORKER_BASE_GAMMA347
    _WORKER_ROWS = rows
    _WORKER_SETTINGS = settings
    _WORKER_BASE_ROUTE = base_route
    _WORKER_GAMMA329 = gamma329
    _WORKER_BASE_GAMMA336 = base_gamma336
    _WORKER_BASE_GAMMA347 = base_gamma347


def _candidate_prediction(row: dict, gamma_expert: tuple[float, ...],
                          base_route: dict, gamma329: tuple[float, ...],
                          guide: list[np.ndarray],
                          base_gamma336: tuple[float, ...],
                          base_gamma347: tuple[float, ...]) -> np.ndarray:
    """Apply Model341, optional fixed Model336, then the candidate expert."""
    route = model341.route
    mixed = route.make_mixed(row, base_route)
    for class_id, gamma in enumerate(gamma329):
        if gamma:
            mixed = route.replace_classes(mixed, row["sources"]["329"],
                                          (class_id,), gamma)
    for class_id, gamma in enumerate(base_gamma336):
        if gamma:
            mixed = route.replace_classes(mixed, row["sources"]["336"],
                                          (class_id,), gamma)
    for class_id, gamma in enumerate(base_gamma347):
        if gamma:
            mixed = route.replace_classes(mixed, row["sources"]["347"],
                                          (class_id,), gamma)
    for class_id, gamma in enumerate(gamma_expert):
        if gamma:
            mixed = route.replace_classes(mixed, row["sources"]["342"],
                                          (class_id,), gamma)
    return route.predict(row["item"], mixed, base_route, guide, exact=True)


def _evaluate_image(image_index: int):
    row = _WORKER_ROWS[image_index]
    target = row["item"]["target"]
    guide = model341.route.bilateral_weights(row["item"]["z"])
    image_scores = np.zeros(len(_WORKER_SETTINGS), dtype=np.float64)
    image_class_scores = np.zeros((len(_WORKER_SETTINGS), 4), dtype=np.float64)
    for route_index in _local_progress(range(1, len(_WORKER_SETTINGS)), 'model346_model336_into_341_route'):
        prediction = _candidate_prediction(
            row, _WORKER_SETTINGS[route_index], _WORKER_BASE_ROUTE,
            _WORKER_GAMMA329, guide, _WORKER_BASE_GAMMA336,
            _WORKER_BASE_GAMMA347)
        matrix = np.bincount(target.astype(np.int64).ravel() * 4
                             + prediction.astype(np.int64).ravel(),
                             minlength=16).reshape(4, 4)
        intersection = np.diag(matrix).astype(np.float64)
        union = matrix.sum(0) + matrix.sum(1) - intersection
        iou = np.divide(intersection, union, out=np.zeros(4), where=union > 0)
        image_scores[route_index] = iou.mean()
        image_class_scores[route_index] = iou
        if route_index % 200 == 0:
            print(f"image {row['index'] + 1}: routes {route_index + 1}/"
                  f"{len(_WORKER_SETTINGS)}", flush=True)
    return image_index, image_scores, image_class_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--max-routes", type=int, default=0,
                        help="prefix limit for smoke validation; 0 evaluates all 625")
    parser.add_argument("--fixed-gamma", nargs=4, type=float, default=None,
                        metavar=("AL3NI", "PRIMARY", "AL", "EUTECTIC"),
                        help="score one explicit expert route plus its anchor; skips the 625-way search")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--expert-id", type=int, choices=(336, 347, 351), default=336)
    parser.add_argument("--experiment-id", type=int, default=None,
                        help="optional result/model ID for a repeated route search")
    parser.add_argument("--expert-probabilities", type=Path, default=None)
    parser.add_argument("--anchor-model346", action="store_true",
                        help="fix Model346's selected gamma336 route before adding Model347")
    parser.add_argument("--anchor-model349", action="store_true",
                        help="fix Model349's Model346+347 route before adding Model351 D4")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.anchor_model346 and args.expert_id != 347:
        raise ValueError("--anchor-model346 is only valid with --expert-id 347")
    if args.anchor_model349 and args.expert_id != 351:
        raise ValueError("--anchor-model349 is only valid with --expert-id 351")
    if args.anchor_model346 and args.anchor_model349:
        raise ValueError("only one fixed anchor route may be selected")
    root, outputs = args.root, args.root / "outputs"
    route_id = (353 if args.anchor_model349 else 349 if args.anchor_model346
                else {336: 346, 347: 348, 351: 353}[args.expert_id])
    if args.experiment_id is not None:
        if args.experiment_id < 1:
            raise ValueError("--experiment-id must be positive")
        route_id = args.experiment_id
    expert_paths = {
        336: outputs / "model336_hrnet_best/valid_eval/valid_probabilities.npz",
        347: outputs / "model347_hrnet_inverse_ce_finetune/valid_eval/valid_probabilities.npz",
        351: outputs / "model351_model336_d4_tta/valid_eval_d4/valid_probabilities.npz",
    }
    expert_path = args.expert_probabilities or expert_paths[args.expert_id]
    base_name = "349" if args.anchor_model349 else "346" if args.anchor_model346 else "341"
    output = args.output or outputs / (
        f"model{route_id}_model{args.expert_id}_into_{base_name}_route/results.json")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output}")

    model339_result = json.loads(
        (outputs / "model339_joint_gamma_route_search/results.json").read_text(encoding="utf-8"))
    model341_result = json.loads(
        (outputs / "model341_model329_joint_route_search/results.json").read_text(encoding="utf-8"))
    baseline_record = model341_result["best_image_empty_zero"]
    if baseline_record.get("base_model", 339) == 338:
        base_route = json.loads((outputs / "model338_fine_gamma_route_search/results.json")
                                .read_text(encoding="utf-8"))["best_image_empty_zero"]
    else:
        base_route = model339_result["best_image_empty_zero"]
    gamma329 = tuple(float(value) for value in baseline_record["gamma329"])
    model346_result = None
    model349_result = None
    if args.anchor_model346 or args.anchor_model349:
        model346_result = json.loads(
            (outputs / "model346_model336_into_341_route/results.json")
            .read_text(encoding="utf-8"))
        base_gamma336 = tuple(float(value)
                              for value in model346_result["raw_best_gamma336"])
    else:
        base_gamma336 = (0.0, 0.0, 0.0, 0.0)
    if args.anchor_model349:
        model349_result = json.loads(
            (outputs / "model349_model347_into_346_route/results.json")
            .read_text(encoding="utf-8"))
        base_gamma347 = tuple(float(value)
                              for value in model349_result["raw_best_gamma347"])
    else:
        base_gamma347 = (0.0, 0.0, 0.0, 0.0)

    source_paths = {
        "195": (outputs / "model195_native_resume/valid_probs.npz", False),
        "217": (outputs / "model220_native_tent_extended/valid_probs_06.npz", False),
        "269": (outputs / "model269_fullstats_1x_extended/valid_probs.npz", False),
        "322": (outputs / "model322_team_b4_group_oof/valid_ensemble/probabilities.npz", True),
        "326": (outputs / "model326_hybrid_fullstats_group_oof/valid_ensemble/probabilities.npz", True),
        "329": (outputs / "model329_fullstats_width4000_uniform_group_oof/valid_ensemble/probabilities.npz", True),
    }
    if args.anchor_model346 or args.anchor_model349:
        source_paths["336"] = (
            outputs / "model336_hrnet_best/valid_eval/valid_probabilities.npz", True)
    if args.anchor_model349:
        source_paths["347"] = (
            outputs / "model347_hrnet_inverse_ce_finetune/valid_eval/valid_probabilities.npz", True)
    archives: dict[str, dict[str, np.ndarray]] = {}
    for source, (path, team_order) in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        _, archives[source] = model341.route.load_archive(path, team_order=team_order)

    if not expert_path.is_file():
        raise FileNotFoundError(expert_path)
    expert_key = str(args.expert_id)
    _, archives[expert_key] = model341.route.load_archive(expert_path, team_order=True)
    names = sorted(archives[expert_key])
    if len(names) != 20:
        raise ValueError(f"expected 20 Model{args.expert_id} validation images, got {len(names)}")
    for source, values in archives.items():
        if set(values) != set(names):
            raise ValueError(f"validation image-key mismatch for Model{source}")

    mask_paths = sorted((root / "data/valid/masks").glob("*_mask.png"))
    mask_names = [canonical_image_key(path.name[:-len("_mask.png")]) for path in mask_paths]
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
        # The shared Model343 candidate routine reads its new expert from slot "342".
        sources["342"] = sources[expert_key]
        rows.append({
            "index": index,
            "item": item,
            "base": model341.route.undo_model181_primary_blend(item),
            "sources": {"base": None, "181": item["extra"].astype(np.float32), **sources},
        })

    if args.fixed_gamma is not None:
        if args.max_routes:
            raise ValueError("--fixed-gamma and --max-routes cannot be combined")
        fixed = tuple(float(value) for value in args.fixed_gamma)
        if any(value not in GAMMAS for value in fixed):
            raise ValueError(f"each fixed gamma must be in {GAMMAS}")
        settings = [(0.0, 0.0, 0.0, 0.0), fixed]
    else:
        settings = list(itertools.product(GAMMAS, repeat=4))
    if args.max_routes:
        if not 2 <= args.max_routes <= len(settings):
            raise ValueError("--max-routes must be between 2 and 625")
        settings = settings[:args.max_routes]
    scores = np.empty((len(settings), len(rows)), dtype=np.float64)
    class_scores = np.empty((len(settings), len(rows), 4), dtype=np.float64)
    baseline_scores = None
    if not (args.anchor_model346 or args.anchor_model349):
        baseline_scores = np.asarray(baseline_record["image_scores"], dtype=np.float64)
        if baseline_scores.shape != (len(rows),):
            raise ValueError("Model341 baseline has a different validation image count")
    # Verify the fixed reference before launching the expensive full grid.
    for image_index, row in enumerate(rows):
        guide = model341.route.bilateral_weights(row["item"]["z"])
        prediction = _candidate_prediction(
            row, settings[0], base_route, gamma329, guide, base_gamma336,
            base_gamma347)
        target = row["item"]["target"]
        matrix = np.bincount(target.astype(np.int64).ravel() * 4
                             + prediction.astype(np.int64).ravel(),
                             minlength=16).reshape(4, 4)
        intersection = np.diag(matrix).astype(np.float64)
        union = matrix.sum(0) + matrix.sum(1) - intersection
        iou = np.divide(intersection, union, out=np.zeros(4), where=union > 0)
        scores[0, image_index] = iou.mean()
        class_scores[0, image_index] = iou
    if args.anchor_model346:
        expected = float(model346_result["raw_best_valid_image_mean_miou_empty0"])
        expected_classes = np.asarray(
            model346_result["raw_best_class_mean_iou_empty0"], dtype=np.float64)
        if not np.isclose(scores[0].mean(), expected, atol=1e-10, rtol=0):
            raise AssertionError("Model346 anchor mean score parity failed")
        if not np.allclose(class_scores[0].mean(axis=0), expected_classes,
                           atol=1e-10, rtol=0):
            raise AssertionError("Model346 anchor class-score parity failed")
    elif args.anchor_model349:
        expected = float(model349_result["raw_best_valid_image_mean_miou_empty0"])
        expected_classes = np.asarray(
            model349_result["raw_best_class_mean_iou_empty0"], dtype=np.float64)
        if not np.isclose(scores[0].mean(), expected, atol=1e-10, rtol=0):
            raise AssertionError("Model349 anchor mean score parity failed")
        if not np.allclose(class_scores[0].mean(axis=0), expected_classes,
                           atol=1e-10, rtol=0):
            raise AssertionError("Model349 anchor class-score parity failed")
    elif not np.allclose(scores[0], baseline_scores, atol=1e-10, rtol=0):
        raise AssertionError(
            f"gamma{args.expert_id}=0 failed the Model341 per-image smoke check"
        )
    if baseline_scores is None:
        baseline_scores = scores[0].copy()
    baseline_class_scores = class_scores[0].copy()
    baseline_model = (349 if args.anchor_model349 else 346 if args.anchor_model346 else 341)
    baseline_label = (f"Model{baseline_model} fixed raw-best route"
                      if baseline_model in (346, 349)
                      else "Model341 fixed strict-best route")
    print(f"{baseline_label} baseline parity passed: "
          f"{scores[0].mean() * 100:.6f}%", flush=True)

    model343.SETTINGS = settings
    start = time.monotonic()
    if args.workers > 1:
        context = mp.get_context("fork")
        with context.Pool(args.workers, initializer=_init_worker,
                          initargs=(rows, settings, base_route, gamma329,
                                    base_gamma336, base_gamma347)) as pool:
            for done, (image_index, image_scores, image_class_scores) in enumerate(
                    _local_progress(pool.imap_unordered(_evaluate_image, range(len(rows)), chunksize=1), 'model346_model336_into_341_route', total=len(rows)), 1):
                scores[:, image_index] = image_scores
                class_scores[:, image_index] = image_class_scores
                print(f"completed image {done}/{len(rows)}; elapsed "
                      f"{(time.monotonic() - start) / 60.0:.1f} min", flush=True)
    else:
        _init_worker(rows, settings, base_route, gamma329, base_gamma336,
                     base_gamma347)
        for image_index in _local_progress(range(len(rows)), 'model346_model336_into_341_route'):
            _, image_scores, image_class_scores = _evaluate_image(image_index)
            scores[:, image_index] = image_scores
            class_scores[:, image_index] = image_class_scores
            print(f"completed image {image_index + 1}/{len(rows)}; elapsed "
                  f"{(time.monotonic() - start) / 60.0:.1f} min", flush=True)
    scores[0] = baseline_scores
    class_scores[0] = baseline_class_scores

    cv = model343.grouped_cv(scores, names, seed=route_id, bootstrap_draws=args.bootstrap)
    cv["reference"] = baseline_label
    gamma_field = f"selected_gamma{args.expert_id}"
    for fold in cv["folds"]:
        fold[gamma_field] = fold.pop("selected_gamma342")
    selected = Counter(tuple(fold[gamma_field]) for fold in cv["folds"])
    majority_gamma = list(selected.most_common(1)[0][0])
    majority_index = settings.index(tuple(majority_gamma))
    best = int(np.argmax(scores.mean(axis=1)))
    result = {
        "model": route_id,
        "metric": "image-wise four-class mIoU; empty union=0",
        "baseline": baseline_label,
        "fixed_gamma336": list(base_gamma336),
        "fixed_gamma347": list(base_gamma347),
        "expert_model": args.expert_id,
        "expert": f"Model{args.expert_id} HRNet-W48 + U-Net++; team-order probabilities converted to route order",
        "expert_probability_path": str(expert_path),
        "class_order": list(CLASS_ORDER),
        "gamma_grid": list(GAMMAS),
        "candidate_count": len(settings),
        "baseline_valid_image_mean_miou_empty0": float(scores[0].mean()),
        f"raw_best_gamma{args.expert_id}": list(settings[best]),
        "raw_best_valid_image_mean_miou_empty0": float(scores[best].mean()),
        "raw_best_class_mean_iou_empty0": class_scores[best].mean(axis=0).tolist(),
        f"grouped_cv_vs_model{baseline_model}": cv,
        f"majority_group_cv_gamma{args.expert_id}": majority_gamma,
        "majority_group_cv_gamma_valid_image_mean_miou_empty0": float(scores[majority_index].mean()),
        "settings_valid_image_mean_miou_empty0": scores.mean(axis=1).tolist(),
        "settings_valid_class_mean_iou_empty0": class_scores.mean(axis=1).tolist(),
        "image_names": names,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.anchor_model346:
        from local_result_export import export_final
        def final_prediction(row):
            guide = model341.route.bilateral_weights(row["item"]["z"])
            return _candidate_prediction(row, settings[best], base_route, gamma329, guide,
                                         base_gamma336, base_gamma347)
        export_final(rows, names, final_prediction, root, result)
    print(json.dumps({key: value for key, value in result.items()
                      if not key.startswith("settings_")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
