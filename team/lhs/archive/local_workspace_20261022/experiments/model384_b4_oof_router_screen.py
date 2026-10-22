#!/usr/bin/env python3
"""Train-OOF-only, deliberately conservative screen for a tiny B4 router.

This is not a confirmatory CV estimate: the base models that produced inner
training OOF predictions may have trained on an outer evaluation fold. The
result is only a resource-priority screen; no valid/test files are opened.
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
FEATURE_NAMES = ("predicted_present_count", "eutectic_area_fraction", "normalized_entropy")
QUANTILES = (0.25, 0.50, 0.75)
LAMBDAS = (0.25, 0.50, 1.00)
BOOTSTRAP_DRAWS = 20_000


def image_group(name: str) -> str:
    match = re.match(r"(A\d+)_", Path(name).name)
    if not match:
        raise ValueError(f"cannot parse specimen group from {name}")
    return match.group(1)


def class_iou(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    if prediction.shape != target.shape:
        raise ValueError(f"shape mismatch: {prediction.shape} vs {target.shape}")
    matrix = np.bincount(
        target.astype(np.int64).ravel() * 4 + prediction.astype(np.int64).ravel(),
        minlength=16,
    ).reshape(4, 4)
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(axis=0) + matrix.sum(axis=1) - intersection
    # Competition policy: empty-union class IoU is exactly zero.
    return np.divide(intersection, union, out=np.zeros(4), where=union > 0)


def route_settings() -> list[dict]:
    # Setting zero is the fixed global 50:50 blend and wins exact ties.
    settings = [{"lambda": 0.0, "feature": None, "q": None, "329_when_low": None}]
    for feature, q, b_when_low, shrink in itertools.product(
        range(len(FEATURE_NAMES)), QUANTILES, (True, False), LAMBDAS
    ):
        settings.append({
            "lambda": shrink,
            "feature": feature,
            "q": q,
            "329_when_low": b_when_low,
        })
    return settings


def mode_index(shrink: float, endpoint: str) -> int:
    if shrink not in LAMBDAS or endpoint not in ("322", "329"):
        raise ValueError(f"unsupported route endpoint/lambda: {endpoint}/{shrink}")
    return 1 + LAMBDAS.index(shrink) * 2 + (endpoint == "329")


def choose_inner_setting(train_indices: np.ndarray, groups: list[str],
                         features: np.ndarray, image_scores: np.ndarray,
                         settings: list[dict]) -> tuple[int, np.ndarray]:
    """Select one rule by group-held-out scoring within the outer train set."""
    held_groups = sorted({groups[i] for i in train_indices})
    collected = [[] for _ in settings]
    for held_group in held_groups:
        held = np.asarray([i for i in train_indices if groups[i] == held_group], dtype=int)
        fit = np.asarray([i for i in train_indices if groups[i] != held_group], dtype=int)
        if not len(fit):
            raise ValueError("inner fit split is empty")
        for setting_index, setting in enumerate(settings):
            if setting["lambda"] == 0:
                scores = image_scores[0, held]
            else:
                feature = int(setting["feature"])
                threshold = float(np.quantile(features[fit, feature], setting["q"]))
                to_329 = features[held, feature] <= threshold
                if not setting["329_when_low"]:
                    to_329 = ~to_329
                scores = np.empty(len(held), dtype=np.float64)
                for local_index, use_329 in enumerate(to_329):
                    endpoint = "329" if use_329 else "322"
                    scores[local_index] = image_scores[
                        mode_index(float(setting["lambda"]), endpoint), held[local_index]
                    ]
            collected[setting_index].extend(scores.tolist())
    means = np.asarray([np.mean(values) for values in collected], dtype=np.float64)
    # Baseline is first; np.argmax preserves it on exact ties.
    selected = int(np.argmax(means))
    return selected, means


def route_scores(setting: dict, train_indices: np.ndarray, eval_indices: np.ndarray,
                 features: np.ndarray, image_scores: np.ndarray) -> np.ndarray:
    if setting["lambda"] == 0:
        return image_scores[0, eval_indices]
    threshold = float(np.quantile(
        features[train_indices, int(setting["feature"])], setting["q"]
    ))
    to_329 = features[eval_indices, int(setting["feature"])] <= threshold
    if not setting["329_when_low"]:
        to_329 = ~to_329
    scores = np.empty(len(eval_indices), dtype=np.float64)
    for local_index, use_329 in enumerate(to_329):
        endpoint = "329" if use_329 else "322"
        scores[local_index] = image_scores[
            mode_index(float(setting["lambda"]), endpoint), eval_indices[local_index]
        ]
    return scores


def grouped_bootstrap(delta: np.ndarray, groups: list[str], draws: int) -> list[float]:
    group_to_indices: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        group_to_indices.setdefault(group, []).append(index)
    group_names = np.asarray(sorted(group_to_indices))
    rng = np.random.default_rng(384)
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        picked = rng.choice(group_names, size=len(group_names), replace=True)
        indices = [i for group in picked for i in group_to_indices[str(group)]]
        samples[draw] = delta[indices].mean() * 100.0
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def smoke_test() -> None:
    target = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
    prediction = target.copy()
    ious = class_iou(prediction, target)
    assert ious[2] == 0.0 and ious[3] == 0.0  # both classes have empty union
    assert np.allclose(ious[:2], 1.0)
    settings = route_settings()
    assert settings[0]["lambda"] == 0.0 and len(settings) == 55
    assert mode_index(0.25, "322") == 1
    assert mode_index(1.0, "329") == 6
    synthetic_features = np.asarray([[0.0], [1.0], [0.0], [1.0]])
    synthetic_groups = ["A1", "A2", "A3", "A4"]
    synthetic_scores = np.zeros((7, 4), dtype=np.float64)
    synthetic_scores[0] = 0.5
    synthetic_scores[1:] = 0.7
    selected, means = choose_inner_setting(
        np.arange(4), synthetic_groups, synthetic_features, synthetic_scores,
        [{"lambda": 0.0, "feature": None, "q": None, "329_when_low": None},
         {"lambda": 0.25, "feature": 0, "q": 0.5, "329_when_low": True}],
    )
    assert selected == 1 and means[1] > means[0]
    print("Model384 smoke passed: metric empty=0, global tie priority, route indexing, group fit.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        smoke_test()
        return

    root = args.root.resolve()
    data_root = root / "data_model322_group_cv"
    model322_path = root / "outputs/model322_team_b4_group_oof/train_oof_probs.npz"
    model329_path = root / "outputs/model329_fullstats_width4000_uniform_group_oof/train_oof_probs.npz"
    output = args.output or root / "outputs/model384_b4_oof_router_screen/results.json"
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output}")
    for path in (model322_path, model329_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    code_dir = root / "model326_hybrid_fullstats/KoMaP"
    sys.path.insert(0, str(code_dir))
    from common import list_split, load_mask  # pylint: disable=import-outside-toplevel

    pairs: list[tuple[str, str]] = []
    fold_ids: list[int] = []
    fold_names: dict[int, set[str]] = {}
    for fold in range(1, 5):
        fold_pairs = list_split(str(data_root), f"oof_fold{fold}")
        pairs.extend(fold_pairs)
        fold_ids.extend([fold] * len(fold_pairs))
        fold_names[fold] = {
            Path(image).name.replace("_image.png", "") for image, _ in fold_pairs
        }
    order = sorted(range(len(pairs)), key=lambda i: Path(pairs[i][0]).name)
    pairs = [pairs[i] for i in order]
    fold_ids = [fold_ids[i] for i in order]
    names = [Path(image).name.replace("_image.png", "") for image, _ in pairs]
    if len(names) != 70 or len(set(names)) != 70:
        raise ValueError(f"expected 70 unique train OOF images, found {len(names)}")
    groups = [image_group(name) for name in names]
    if len(set(groups)) != 59:
        raise ValueError(f"expected 59 specimen groups, found {len(set(groups))}")
    group_fold: dict[str, set[int]] = {}
    for group, fold in zip(groups, fold_ids):
        group_fold.setdefault(group, set()).add(fold)
    if any(len(folds) != 1 for folds in group_fold.values()):
        raise ValueError("a specimen group occurs in multiple base OOF folds")
    for model_path in (model322_path, model329_path):
        for fold in range(1, 5):
            fold_path = model_path.parent / f"fold{fold}_oof/probabilities.npz"
            if not fold_path.is_file():
                raise FileNotFoundError(f"missing fold provenance archive: {fold_path}")
            with np.load(fold_path) as fold_archive:
                if set(fold_archive.files) != fold_names[fold]:
                    raise ValueError(
                        f"{fold_path} keys do not exactly match fixed OOF fold {fold}"
                    )

    settings = route_settings()
    features = np.empty((len(names), len(FEATURE_NAMES)), dtype=np.float64)
    image_ious = np.empty((7, len(names), 4), dtype=np.float64)
    start = time.monotonic()
    with np.load(model322_path) as archive322, np.load(model329_path) as archive329:
        expected = set(names)
        if set(archive322.files) != expected or set(archive329.files) != expected:
            raise ValueError("OOF archive keys do not exactly match the 70 train images")
        for index, (name, (_, mask_path)) in enumerate(zip(names, pairs)):
            p322 = archive322[name].astype(np.float32)
            p329 = archive329[name].astype(np.float32)
            if p322.shape != p329.shape or p322.ndim != 3 or p322.shape[0] != 4:
                raise ValueError(f"invalid probability shape for {name}: {p322.shape}/{p329.shape}")
            if not np.isfinite(p322).all() or not np.isfinite(p329).all():
                raise ValueError(f"non-finite probability for {name}")
            if (p322.min() < -0.01 or p329.min() < -0.01 or
                    p322.max() > 1.01 or p329.max() > 1.01):
                raise ValueError(f"probabilities outside [0,1] for {name}")
            target = load_mask(mask_path)
            if target.shape != p322.shape[1:]:
                raise ValueError(f"mask/probability spatial mismatch for {name}")

            global_prob = 0.5 * (p322 + p329)
            global_pred = global_prob.argmax(axis=0).astype(np.uint8)
            areas = np.bincount(global_pred.ravel(), minlength=4) / global_pred.size
            entropy = -np.sum(
                global_prob * np.log(np.clip(global_prob, 1e-7, 1.0)), axis=0
            ).mean() / np.log(4.0)
            features[index] = (float(np.sum(areas >= 0.001)), areas[3], entropy)
            image_ious[0, index] = class_iou(global_pred, target)

            for lambda_index, shrink in enumerate(LAMBDAS):
                for endpoint_index, endpoint_prob in enumerate((p322, p329)):
                    mixed = global_prob + shrink * (endpoint_prob - global_prob)
                    mode = 1 + lambda_index * 2 + endpoint_index
                    image_ious[mode, index] = class_iou(
                        mixed.argmax(axis=0).astype(np.uint8), target
                    )
            if (index + 1) % 10 == 0 or index + 1 == len(names):
                print(f"scored train OOF {index + 1}/70; "
                      f"elapsed {(time.monotonic() - start) / 60.0:.1f} min", flush=True)

    image_scores = image_ious.mean(axis=2)
    outer_fold_ids = sorted(set(fold_ids))
    delta = np.empty(len(names), dtype=np.float64)
    baseline = image_scores[0]
    fold_results = []
    chosen_settings = []
    for outer_fold in outer_fold_ids:
        train_indices = np.asarray([i for i, fold in enumerate(fold_ids) if fold != outer_fold])
        eval_indices = np.asarray([i for i, fold in enumerate(fold_ids) if fold == outer_fold])
        selected_index, inner_means = choose_inner_setting(
            train_indices, groups, features, image_scores, settings
        )
        setting = settings[selected_index]
        selected_scores = route_scores(
            setting, train_indices, eval_indices, features, image_scores
        )
        delta[eval_indices] = selected_scores - baseline[eval_indices]
        chosen_settings.append(setting)
        fold_results.append({
            "outer_base_fold": outer_fold,
            "images": int(len(eval_indices)),
            "groups": int(len({groups[i] for i in eval_indices})),
            "inner_selected_setting": {
                **setting,
                "feature_name": (FEATURE_NAMES[setting["feature"]]
                                 if setting["feature"] is not None else None),
            },
            "inner_selected_train_group_cv_miou": float(inner_means[selected_index] * 100.0),
            "outer_global_blend_miou": float(baseline[eval_indices].mean() * 100.0),
            "outer_routed_miou": float(selected_scores.mean() * 100.0),
            "outer_delta_pp": float((selected_scores.mean() - baseline[eval_indices].mean()) * 100.0),
        })

    total_delta_pp = float(delta.mean() * 100.0)
    ci = grouped_bootstrap(delta, groups, args.bootstrap)
    positive_folds = sum(row["outer_delta_pp"] > 0 for row in fold_results)
    screen_pass = (total_delta_pp >= 0.30 and ci[0] > 0.0 and positive_folds >= 3)
    result = {
        "model": 384,
        "status": "train-only screening; not a confirmatory or valid/test estimate",
        "metric": "image-wise mean of four class IoUs; empty union=0",
        "class_order": list(CLASS_NAMES),
        "data": {
            "images": len(names), "specimen_groups": len(set(groups)),
            "base_oof_folds": outer_fold_ids,
            "model322_oof": str(model322_path), "model329_oof": str(model329_path),
        },
        "global_baseline": "fixed equal-probability mean of Model322 and Model329 (lambda=0)",
        "router": {
            "candidate_experts": ["Model322", "Model329"],
            "feature_names": list(FEATURE_NAMES),
            "threshold_quantiles": list(QUANTILES),
            "shrinkage_lambdas": list(LAMBDAS),
            "max_depth": 1,
        },
        "leakage_warning": (
            "The outer-heldout group can appear in base-model training for inner-router OOF predictions. "
            "Thus the outer score is not an unbiased generalization estimate and must not select a final model. "
            "A pass only prioritizes later confirmation; a fail deprioritizes this small router."
        ),
        "screen_gate": {
            "delta_at_least_pp": 0.30,
            "group_bootstrap_ci_lower_gt_zero": True,
            "positive_outer_folds_at_least": 3,
            "screen_pass": bool(screen_pass),
        },
        "global_oof_image_mean_miou": float(baseline.mean() * 100.0),
        "routed_oof_image_mean_miou": float((baseline + delta).mean() * 100.0),
        "delta_pp": total_delta_pp,
        "delta_group_bootstrap_95ci_pp": ci,
        "positive_images": int((delta > 0).sum()),
        "positive_outer_folds": int(positive_folds),
        "outer_folds": fold_results,
        "seed": 384,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "screen_pass": screen_pass,
        "global_oof_image_mean_miou": result["global_oof_image_mean_miou"],
        "routed_oof_image_mean_miou": result["routed_oof_image_mean_miou"],
        "delta_pp": total_delta_pp,
        "delta_group_bootstrap_95ci_pp": ci,
        "positive_outer_folds": positive_folds,
        "output": str(output),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
