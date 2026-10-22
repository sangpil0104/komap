#!/usr/bin/env python3
"""Train-OOF-only comparison of the new HRNet branch against Model329 B4.

No validation masks are used. A fixed equal blend is the primary paired OOF
gate. Grid-selected and leave-one-group-out routes are exploratory only because
cross-fitted model weights can create indirect dependence across folds.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

import model342_oof_fitted_b4_blend as blend


def group_bootstrap(values: np.ndarray, group_indices: dict[str, list[int]],
                    draws: int, seed: int) -> list[float]:
    names = list(group_indices)
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = rng.choice(names, size=len(names), replace=True)
        indices = [i for group in sampled for i in group_indices[str(group)]]
        estimates[draw] = values[indices].mean() * 100.0
    return [float(v) for v in np.quantile(estimates, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--gammas", type=float, nargs="+", default=(0.0, 0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    gammas = tuple(sorted(set(args.gammas)))
    if gammas != (0.0, 0.25, 0.5, 0.75, 1.0):
        raise ValueError("fair comparison requires the exact Model342 gamma grid: 0,.25,.5,.75,1")
    root = args.root
    output = args.output or root / "outputs/model375_hrnet_b4_oof_blend/results.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    base_path = root / "outputs/model329_fullstats_width4000_uniform_group_oof/train_oof_probs.npz"
    hrnet_path = root / "outputs/model374_hrnet_group_oof/train_oof_probs.npz"
    b4_control_path = root / "outputs/model322_team_b4_group_oof/train_oof_probs.npz"
    if not base_path.is_file() or not hrnet_path.is_file() or not b4_control_path.is_file():
        raise FileNotFoundError(
            f"missing train OOF archives: {base_path}, {hrnet_path}, {b4_control_path}")

    code_dir = root / "model326_hybrid_fullstats/KoMaP"
    sys.path.insert(0, str(code_dir))
    from common import list_split, load_mask  # pylint: disable=import-outside-toplevel

    pairs = []
    for fold in range(1, 5):
        pairs.extend(list_split(str(root / "data_model322_group_cv"), f"oof_fold{fold}"))
    pairs.sort(key=lambda row: Path(row[0]).name)
    names = [Path(image).name.removesuffix("_image.png") for image, _ in pairs]
    groups = [blend.image_group(name) for name in names]
    if len(names) != 70 or len(set(names)) != 70 or len(set(groups)) != 59:
        raise ValueError(f"expected 70 exactly-once images/59 groups, got {len(names)}/{len(set(groups))}")

    settings = list(itertools.product(gammas, repeat=4))
    hrnet_scores = np.empty((len(settings), len(names)), dtype=np.float64)
    hrnet_class_scores = np.empty((len(settings), len(names), 4), dtype=np.float64)
    b4_scores = np.empty_like(hrnet_scores)
    group_indices: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        group_indices.setdefault(group, []).append(index)

    started = time.monotonic()
    with (np.load(base_path) as base_archive,
          np.load(hrnet_path) as hrnet_archive,
          np.load(b4_control_path) as b4_control_archive):
        expected = set(names)
        if (set(base_archive.files) != expected or set(hrnet_archive.files) != expected
                or set(b4_control_archive.files) != expected):
            raise ValueError("Model322/329/374 OOF keys do not match the fixed grouped split")
        for index, (name, (_, mask_path)) in enumerate(zip(names, pairs)):
            model329 = base_archive[name].astype(np.float32)
            hrnet = hrnet_archive[name].astype(np.float32)
            model322 = b4_control_archive[name].astype(np.float32)
            if (model329.shape != hrnet.shape or model329.shape != model322.shape
                    or model329.ndim != 3 or model329.shape[0] != 4):
                raise ValueError(f"probability shape mismatch for {name}: "
                                 f"{model322.shape}/{model329.shape}/{hrnet.shape}")
            if (not np.isfinite(model322).all() or not np.isfinite(model329).all()
                    or not np.isfinite(hrnet).all()):
                raise ValueError(f"nonfinite probabilities for {name}")
            target = load_mask(mask_path)
            hrnet_scores[:, index], hrnet_class_scores[:, index] = blend.score_grid(
                model329, hrnet, target, settings)
            b4_scores[:, index], _ = blend.score_grid(model322, model329, target, settings)
            if (index + 1) % 10 == 0 or index + 1 == len(names):
                print(f"scored OOF images {index + 1}/{len(names)}; "
                      f"elapsed {(time.monotonic()-started)/60:.1f} min", flush=True)

    rng = np.random.default_rng(375)
    group_names = list(group_indices)

    def crossvalidated(matrix: np.ndarray, seed: int) -> dict:
        selected = np.empty(len(names), dtype=np.int64)
        choices = []
        for held_group, held_indices in group_indices.items():
            train_indices = [i for group, indices in group_indices.items()
                             if group != held_group for i in indices]
            choice = int(np.argmax(matrix[:, train_indices].mean(axis=1)))
            selected[held_indices] = choice
            choices.append({"heldout_specimen": held_group,
                            "selected_setting_index": choice,
                            "selected_gamma": list(settings[choice])})
        selected_scores = matrix[selected, np.arange(len(names))]
        baseline = matrix[0]
        delta = selected_scores - baseline
        rng_local = np.random.default_rng(seed)
        delta_draws = np.empty(args.bootstrap, dtype=np.float64)
        absolute_draws = np.empty(args.bootstrap, dtype=np.float64)
        for draw in range(args.bootstrap):
            sampled = rng_local.choice(group_names, size=len(group_names), replace=True)
            indices = [i for group in sampled for i in group_indices[str(group)]]
            delta_draws[draw] = delta[indices].mean() * 100.0
            absolute_draws[draw] = selected_scores[indices].mean() * 100.0
        return {
            "selected_scores": selected_scores,
            "delta": delta,
            "fold_choices": choices,
            "image_mean_delta_pp": float(delta.mean() * 100.0),
            "grouped_cv_image_mean_miou_empty0": float(selected_scores.mean() * 100.0),
            "group_bootstrap_95ci_delta_pp": [
                float(v) for v in np.quantile(delta_draws, [0.025, 0.975])],
            "group_bootstrap_95ci_absolute_miou": [
                float(v) for v in np.quantile(absolute_draws, [0.025, 0.975])],
            "positive_images": int((delta > 0).sum()),
        }

    hrnet_cv = crossvalidated(hrnet_scores, 375)
    b4_cv = crossvalidated(b4_scores, 376)
    model342_path = root / "outputs/model342_oof_fitted_b4_blend/results.json"
    model342 = json.loads(model342_path.read_text(encoding="utf-8"))
    prior_control_delta = float(model342["grouped_cv_vs_model322"]["image_mean_delta_pp"])
    if tuple(model342["gamma_grid"]) != gammas:
        raise AssertionError("Model342 gamma grid differs from the Model375 control grid")
    if not np.isclose(b4_cv["image_mean_delta_pp"], prior_control_delta,
                      atol=1e-6, rtol=0):
        raise AssertionError(
            "reconstructed B4-only control does not reproduce Model342 grouped-CV delta: "
            f"{b4_cv['image_mean_delta_pp']} vs {prior_control_delta}"
        )
    model342_choices = {
        item["heldout_specimen"]: item["selected_setting_index"]
        for item in model342["grouped_cv_vs_model322"]["fold_choices"]
    }
    reconstructed_choices = {
        item["heldout_specimen"]: item["selected_setting_index"]
        for item in b4_cv["fold_choices"]
    }
    if model342_choices != reconstructed_choices:
        raise AssertionError("reconstructed Model342 group-wise setting choices differ")
    if not np.array_equal(hrnet_scores[0], b4_scores[-1]):
        raise AssertionError("Model329 baseline scores differ between the two blend arms")
    if not np.isclose(float(b4_scores[0].mean() * 100.0),
                      float(model342["model322_train_oof_miou_empty0"]),
                      atol=1e-8, rtol=0):
        raise AssertionError("Model322 OOF baseline does not reproduce Model342")
    best_control = int(np.argmax(b4_scores.mean(axis=1)))
    if not np.isclose(float(b4_scores[best_control].mean() * 100.0),
                      float(model342["selected_train_oof_miou_empty0"]),
                      atol=1e-8, rtol=0):
        raise AssertionError("Model342 selected OOF score does not reproduce")
    best = int(np.argmax(hrnet_scores.mean(axis=1)))
    fixed_idx = settings.index((0.5, 0.5, 0.5, 0.5))
    fixed_hrnet_delta = hrnet_scores[fixed_idx] - hrnet_scores[0]
    fixed_b4_delta = b4_scores[fixed_idx] - b4_scores[0]
    fixed_hrnet_vs_b4 = hrnet_scores[fixed_idx] - b4_scores[fixed_idx]
    fixed_ci_hrnet = group_bootstrap(fixed_hrnet_delta, group_indices, args.bootstrap, 377)
    fixed_ci_b4 = group_bootstrap(fixed_b4_delta, group_indices, args.bootstrap, 378)
    fixed_ci_vs_b4 = group_bootstrap(fixed_hrnet_vs_b4, group_indices, args.bootstrap, 379)
    fixed_gate = (
        fixed_hrnet_delta.mean() * 100.0 >= 0.15 and fixed_ci_hrnet[0] > 0
        and fixed_hrnet_vs_b4.mean() * 100.0 >= 0.15 and fixed_ci_vs_b4[0] > 0
    )
    paired_delta = hrnet_cv["selected_scores"] - b4_cv["selected_scores"]
    result = {
        "model": 375,
        "metric": "70-image mean of four class IoUs; empty union=0",
        "data_selection": "Model322/329/374 exactly-once grouped train OOF only; no validation files/masks opened; Model342 JSON contributes OOF control metadata only",
        "class_order": ["Al", "Al3Ni", "Primary Si", "Eutectic Si"],
        "images": len(names),
        "specimen_groups": len(group_indices),
        "gamma_grid": list(gammas),
        "candidate_count": len(settings),
        "model329_oof_image_mean_miou_empty0": float(hrnet_scores[0].mean()),
        "model374_oof_image_mean_miou_empty0": float(hrnet_scores[-1].mean()),
        "full_oof_best_grid_gamma_diagnostic_only": list(settings[best]),
        "full_oof_best_grid_score_selection_biased": float(hrnet_scores[best].mean()),
        "full_oof_best_grid_class_iou_diagnostic_only": hrnet_class_scores[best].mean(axis=0).tolist(),
        "fixed_equal_blend_oof_gate": {
            "gamma_all_classes": 0.5,
            "model329_oof_miou_empty0": float(hrnet_scores[0].mean() * 100.0),
            "model329_plus_hrnet_oof_miou_empty0": float(hrnet_scores[fixed_idx].mean() * 100.0),
            "model329_plus_hrnet_delta_pp": float(fixed_hrnet_delta.mean() * 100.0),
            "model329_plus_hrnet_group_bootstrap_95ci_delta_pp": fixed_ci_hrnet,
            "model322_plus_model329_oof_miou_empty0": float(b4_scores[fixed_idx].mean() * 100.0),
            "model322_plus_model329_delta_pp": float(fixed_b4_delta.mean() * 100.0),
            "model322_plus_model329_group_bootstrap_95ci_delta_pp": fixed_ci_b4,
            "hrnet_blend_minus_b4_control_delta_pp": float(fixed_hrnet_vs_b4.mean() * 100.0),
            "hrnet_blend_minus_b4_control_group_bootstrap_95ci_delta_pp": fixed_ci_vs_b4,
            "predeclared_gate_pass": bool(fixed_gate),
            "note": "Fixed 0.5 weights are the primary paired OOF check; no OOF-selected route is used for this gate.",
        },
        "grouped_cv_hrnet_blend_vs_model329": {
            **{key: value for key, value in hrnet_cv.items() if key != "selected_scores" and key != "delta"},
            "estimate_caveat": "Exploratory only: other-fold base models may have trained on the held-out group.",
        },
        "grouped_cv_b4_blend_control_model342": {
            **{key: value for key, value in b4_cv.items() if key != "selected_scores" and key != "delta"},
            "reported_model342_cv_delta_vs_model322_pp": prior_control_delta,
        },
        "paired_grouped_cv_hrnet_route_minus_b4_control": {
            "image_mean_delta_pp": float(paired_delta.mean() * 100.0),
            "group_bootstrap_95ci_delta_pp": group_bootstrap(
                paired_delta, group_indices, args.bootstrap, 380),
            "estimate_caveat": "Exploratory only: other-fold base models may have trained on the held-out group.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "model329_oof_image_mean_miou_empty0", "model374_oof_image_mean_miou_empty0",
        "full_oof_best_grid_gamma_diagnostic_only", "full_oof_best_grid_score_selection_biased",
        "fixed_equal_blend_oof_gate",
        "grouped_cv_hrnet_blend_vs_model329",
        "grouped_cv_b4_blend_control_model342",
        "paired_grouped_cv_hrnet_route_minus_b4_control")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
