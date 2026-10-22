#!/usr/bin/env python3
"""CPU-only pre-registered geometry diagnostic for Model374/376/377 OOF."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


GROUP_RE = re.compile(r"^(A\d+)")
PALETTE = np.asarray(
    ((153, 127, 76), (25, 76, 153), (76, 178, 76), (204, 204, 204)),
    dtype=np.uint8,
)
CLASS_A, CLASS_B = 1, 3  # Al3Ni, Eutectic Si
MODELS = ("Model374", "Model376", "Model377")


def mask_to_ids(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    result = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        result[np.all(rgb == color, axis=2)] = class_id
    if np.any(result == 255):
        raise ValueError(f"unknown RGB mask color(s) in {path}")
    return result


def distance_to_features(features: np.ndarray) -> np.ndarray:
    """Exact L2 distance to true pixels using OpenCV's precise CPU transform."""
    source = (~features).astype(np.uint8)
    return cv2.distanceTransform(source, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def pair_interface(mask: np.ndarray, a: int, b: int) -> np.ndarray:
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[:-1, :] |= (((mask[:-1, :] == a) & (mask[1:, :] == b)) |
                         ((mask[:-1, :] == b) & (mask[1:, :] == a)))
    boundary[1:, :] |= (((mask[1:, :] == a) & (mask[:-1, :] == b)) |
                        ((mask[1:, :] == b) & (mask[:-1, :] == a)))
    boundary[:, :-1] |= (((mask[:, :-1] == a) & (mask[:, 1:] == b)) |
                         ((mask[:, :-1] == b) & (mask[:, 1:] == a)))
    boundary[:, 1:] |= (((mask[:, 1:] == a) & (mask[:, :-1] == b)) |
                        ((mask[:, 1:] == b) & (mask[:, :-1] == a)))
    return boundary


def neighbor_of(mask: np.ndarray, target: int) -> np.ndarray:
    result = np.zeros(mask.shape, dtype=bool)
    result[:-1, :] |= mask[1:, :] == target
    result[1:, :] |= mask[:-1, :] == target
    result[:, :-1] |= mask[:, 1:] == target
    result[:, 1:] |= mask[:, :-1] == target
    return result


def per_image_miou(prediction: np.ndarray, truth: np.ndarray) -> float:
    matrix = np.bincount(truth.ravel().astype(np.int64) * 4 + prediction.ravel().astype(np.int64),
                         minlength=16).reshape(4, 4)
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(axis=0) + matrix.sum(axis=1) - intersection
    iou = np.divide(intersection, union, out=np.zeros(4, dtype=np.float64), where=union > 0)
    return float(iou.mean())


def group_bootstrap(values: dict[str, float], draws: int, seed: int) -> list[float]:
    array = np.asarray(list(values.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(array), size=(draws, len(array)))
    estimates = np.median(array[sample], axis=1)
    return np.percentile(estimates, [2.5, 97.5]).astype(float).tolist()


def run(cv_root: Path, model_roots: dict[str, Path], draws: int = 20000) -> dict:
    per_image = {model: {} for model in MODELS}
    displacement: dict[str, dict[str, list[float]]] = {
        model: defaultdict(list) for model in MODELS
    }
    error_masks_by_group: dict[str, dict[str, list[np.ndarray]]] = {
        model: defaultdict(list) for model in MODELS
    }
    seen: set[str] = set()
    group_fold: dict[str, int] = {}

    for fold in range(1, 5):
        masks = {path.name.removesuffix("_mask.png"): path
                 for path in (cv_root / f"oof_fold{fold}/masks").glob("*_mask.png")}
        image_dir = cv_root / f"oof_fold{fold}/images"
        if not masks or not image_dir.is_dir():
            raise FileNotFoundError(f"missing OOF data for fold {fold}")
        archives = {
            model: np.load(root / f"fold{fold}_oof/valid_probabilities.npz", allow_pickle=False)
            for model, root in model_roots.items()
        }
        try:
            expected = set(masks)
            for model, archive in archives.items():
                if set(archive.files) != expected:
                    raise ValueError(f"fold {fold}: {model} image IDs do not match GT")
            for image_id, mask_path in masks.items():
                if image_id in seen:
                    raise ValueError(f"duplicate OOF image {image_id}")
                seen.add(image_id)
                match = GROUP_RE.match(image_id)
                if not match:
                    raise ValueError(f"cannot parse specimen group: {image_id}")
                group = match.group(1)
                if group in group_fold and group_fold[group] != fold:
                    raise ValueError(f"specimen group {group} leaks across OOF folds")
                group_fold[group] = fold
                truth = mask_to_ids(mask_path)
                interface = pair_interface(truth, CLASS_A, CLASS_B)
                if not interface.any():
                    raise ValueError(f"no GT Al3Ni/Eutectic interface in {image_id}")
                distance = distance_to_features(interface)
                near = distance <= 2.0
                pair_domain = (truth == CLASS_A) | (truth == CLASS_B)
                distance_to_a = distance_to_features(truth == CLASS_A)
                distance_to_b = distance_to_features(truth == CLASS_B)
                signed_gt = distance_to_a - distance_to_b  # negative in GT Al3Ni, positive in GT Eutectic

                predictions = {}
                for model, archive in archives.items():
                    probabilities = np.asarray(archive[image_id], dtype=np.float32)
                    if probabilities.shape != (4, *truth.shape) or not np.isfinite(probabilities).all():
                        raise ValueError(f"invalid {model} probability shape/values for {image_id}")
                    if np.max(np.abs(probabilities.sum(axis=0) - 1.0)) > 0.03:
                        raise ValueError(f"unnormalized {model} probabilities for {image_id}")
                    prediction = probabilities.argmax(axis=0).astype(np.uint8, copy=False)
                    predictions[model] = prediction
                    swap = (((truth == CLASS_A) & (prediction == CLASS_B)) |
                            ((truth == CLASS_B) & (prediction == CLASS_A)))
                    signed_support = ((prediction == CLASS_A) & neighbor_of(prediction, CLASS_B)
                                      & pair_domain)
                    signed_values = signed_gt[signed_support]
                    if signed_values.size:
                        displacement[model][group].append(float(np.median(signed_values)))
                    corrected = prediction.copy()
                    corrected[swap & near] = truth[swap & near]
                    per_image[model][image_id] = {
                        "group": group,
                        "base_miou": per_image_miou(prediction, truth),
                        "oracle_miou": per_image_miou(corrected, truth),
                        "pair_swap_errors": int(swap.sum()),
                        "pair_swap_errors_within_2px": int((swap & near).sum()),
                        "pair_swap_error_distance_values": distance[swap],
                    }
                model_errors = {
                    model: ((((truth == CLASS_A) & (predictions[model] == CLASS_B)) |
                             ((truth == CLASS_B) & (predictions[model] == CLASS_A))) & near)
                    for model in MODELS
                }
                for model in MODELS:
                    error_masks_by_group[model][group].append(model_errors[model])
        finally:
            for archive in archives.values():
                archive.close()

    if len(seen) != 70 or len(group_fold) != 59:
        raise ValueError(f"expected 70 images/59 groups exactly once, found {len(seen)}/{len(group_fold)}")

    summaries = {}
    groups = sorted({row["group"] for row in per_image["Model374"].values()})
    for model in MODELS:
        entries = list(per_image[model].values())
        total_swaps = sum(row["pair_swap_errors"] for row in entries)
        near_swaps = sum(row["pair_swap_errors_within_2px"] for row in entries)
        base_scores = np.asarray([row["base_miou"] for row in entries], dtype=np.float64)
        oracle_scores = np.asarray([row["oracle_miou"] for row in entries], dtype=np.float64)
        group_deltas = {
            group: float(np.mean([
                row["oracle_miou"] - row["base_miou"]
                for row in entries if row["group"] == group
            ]))
            for group in groups
        }
        rng = np.random.default_rng(381)
        delta_array = np.asarray([row["oracle_miou"] - row["base_miou"] for row in entries])
        group_rows = {group: np.flatnonzero(np.asarray([row["group"] for row in entries]) == group)
                      for group in groups}
        boot = np.empty(draws, dtype=np.float64)
        group_names = list(group_rows)
        for draw in range(draws):
            selected = rng.integers(0, len(group_names), size=len(group_names))
            indices = np.concatenate([group_rows[group_names[index]] for index in selected])
            boot[draw] = delta_array[indices].mean()
        displacement_groups = {
            group: float(np.median(values))
            for group, values in displacement[model].items() if values
        }
        distance_parts = [row["pair_swap_error_distance_values"] for row in entries
                          if row["pair_swap_error_distance_values"].size]
        all_distances = np.concatenate(distance_parts) if distance_parts else np.asarray([], dtype=np.float64)
        summaries[model] = {
            "images": len(entries),
            "groups_with_signed_boundary_support": len(displacement_groups),
            "official_image_mean_miou_empty0_pct": float(base_scores.mean() * 100),
            "pair_swap_error_pixels": int(total_swaps),
            "pair_swap_error_pixels_within_2px": int(near_swaps),
            "pair_swap_errors_within_2px_pct": float(near_swaps / total_swaps * 100) if total_swaps else None,
            "restricted_pair_swap_oracle_miou_pct": float(oracle_scores.mean() * 100),
            "restricted_pair_swap_oracle_delta_pp": float((oracle_scores - base_scores).mean() * 100),
            "restricted_oracle_delta_group_bootstrap_95ci_pp":
                (np.percentile(boot, [2.5, 97.5]) * 100).astype(float).tolist(),
            "error_distance_to_gt_pair_interface_px_quantiles":
                np.percentile(all_distances, [25, 50, 75, 90]).astype(float).tolist()
                if all_distances.size else None,
            "signed_displacement_px_group_median": float(np.median(list(displacement_groups.values())))
                if displacement_groups else None,
            "signed_displacement_px_group_iqr": np.percentile(
                list(displacement_groups.values()), [25, 75]
            ).astype(float).tolist() if displacement_groups else None,
            "signed_displacement_px_group_median_bootstrap_95ci":
                group_bootstrap(displacement_groups, draws, 381) if displacement_groups else None,
            "signed_displacement_groups_same_sign_pct": float(
                np.mean(np.sign(list(displacement_groups.values())) ==
                        np.sign(np.median(list(displacement_groups.values())))) * 100
            ) if displacement_groups else None,
        }

    group_jaccards = {}
    for group in groups:
        masks = [np.concatenate([value.ravel() for value in error_masks_by_group[model][group]])
                 for model in MODELS]
        intersection = np.logical_and.reduce(masks).sum()
        union = np.logical_or.reduce(masks).sum()
        if union:
            group_jaccards[group] = float(intersection / union)
    common_intersection = 0
    common_union = 0
    for group in groups:
        masks = [np.concatenate([value.ravel() for value in error_masks_by_group[model][group]])
                 for model in MODELS]
        common_intersection += int(np.logical_and.reduce(masks).sum())
        common_union += int(np.logical_or.reduce(masks).sum())
    rng = np.random.default_rng(382)
    jaccard_values = np.asarray([group_jaccards[group] for group in groups])
    sample = rng.integers(0, len(jaccard_values), size=(draws, len(jaccard_values)))
    jaccard_ci = np.percentile(jaccard_values[sample].mean(axis=1), [2.5, 97.5]).tolist()

    baseline = summaries["Model374"]
    gate = {
        "at_least_60pct_pair_swap_errors_within_2px":
            (baseline["pair_swap_errors_within_2px_pct"] or 0) >= 60,
        "signed_offset_median_abs_le_0_5px_and_ci_contains_zero": (
            baseline["signed_displacement_px_group_median"] is not None
            and abs(baseline["signed_displacement_px_group_median"]) <= 0.5
            and baseline["signed_displacement_px_group_median_bootstrap_95ci"][0] <= 0
            <= baseline["signed_displacement_px_group_median_bootstrap_95ci"][1]
        ),
        "three_model_group_equal_error_jaccard_at_least_0_50":
            float(np.mean(list(group_jaccards.values()))) >= 0.50,
        "374_restricted_oracle_gain_at_least_0_50pp":
            baseline["restricted_pair_swap_oracle_delta_pp"] >= 0.50,
    }
    return {
        "model": 381,
        "scope": "70 training images, 59 specimen groups; OOF only, valid/test excluded",
        "metric": "official per-image four-class mIoU, empty-union=0",
        "pair": {"class_a": "Al3Ni", "class_a_id": CLASS_A,
                 "class_b": "Eutectic_Si", "class_b_id": CLASS_B},
        "distance": "Euclidean pixel distance to axial 4-neighbor GT pair interface; near=<=2px",
        "signed_offset_convention": "d(to Al3Ni)-d(to Eutectic); sampled on predicted Al3Ni-side boundary",
        "models": summaries,
        "three_model_error_location_jaccard": {
            "pixel_pooled_jaccard": float(common_intersection / common_union) if common_union else 1.0,
            "group_equal_mean_jaccard": float(np.mean(list(group_jaccards.values()))),
            "group_bootstrap_95ci": [float(x) for x in jaccard_ci],
            "groups": len(group_jaccards),
        },
        "model378_diagnostic_go_gate": gate,
        "model378_diagnostic_go": all(gate.values()),
        "interpretation_guard": (
            "This diagnostic is conditional on previously selected training recipes. Signed EDT offsets on irregular "
            "microstructures are label-geometric summaries, not physical phase-boundary measurements. Oracle scores "
            "use ground truth and are not achievable model scores."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--model374-root", type=Path, required=True)
    parser.add_argument("--model376-root", type=Path, required=True)
    parser.add_argument("--model377-root", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.cv_root, {
        "Model374": args.model374_root,
        "Model376": args.model376_root,
        "Model377": args.model377_root,
    }, args.draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "model": result["model"],
        "models": {name: {
            "official_miou_pct": data["official_image_mean_miou_empty0_pct"],
            "pair_swap_errors_near_pct": data["pair_swap_errors_within_2px_pct"],
            "restricted_oracle_delta_pp": data["restricted_pair_swap_oracle_delta_pp"],
            "signed_offset_group_median_px": data["signed_displacement_px_group_median"],
            "signed_offset_ci": data["signed_displacement_px_group_median_bootstrap_95ci"],
        } for name, data in result["models"].items()},
        "error_jaccard": result["three_model_error_location_jaccard"],
        "model378_diagnostic_go_gate": result["model378_diagnostic_go_gate"],
        "model378_diagnostic_go": result["model378_diagnostic_go"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
