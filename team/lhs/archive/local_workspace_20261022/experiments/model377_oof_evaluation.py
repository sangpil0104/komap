#!/usr/bin/env python3
"""Pre-registered train-only OOF evaluation for Model377.

The adoption comparison is Model377 vs Model374. The mechanism comparison is
Model377 vs Model376. Only grouped train OOF predictions and their masks are
read; official valid/test data are not inputs to this program.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_NAMES = ("Al", "Al3Ni", "Primary_Si", "Eutectic_Si")
PALETTE = np.asarray(
    ((153, 127, 76), (25, 76, 153), (76, 178, 76), (204, 204, 204)),
    dtype=np.uint8,
)
GROUP_RE = re.compile(r"^(A\d+)")


def mask_to_ids(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    result = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        result[np.all(rgb == color, axis=2)] = class_id
    if np.any(result == 255):
        raise ValueError(f"unknown RGB mask color(s) in {path}")
    return result


def class_ious(probabilities: np.ndarray, truth: np.ndarray) -> np.ndarray:
    if probabilities.shape != (4, *truth.shape) or not np.isfinite(probabilities).all():
        raise ValueError(f"invalid probability shape/values: {probabilities.shape}, {truth.shape}")
    if np.max(np.abs(probabilities.sum(axis=0) - 1.0)) > 0.03:
        raise ValueError("probability map is not normalized")
    prediction = probabilities.argmax(axis=0)
    matrix = np.bincount(truth.ravel().astype(np.int64) * 4 + prediction.ravel(), minlength=16)
    matrix = matrix.reshape(4, 4)
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(axis=0) + matrix.sum(axis=1) - intersection
    return np.divide(intersection, union, out=np.zeros(4), where=union > 0)


def adjacent_support(truth: np.ndarray, side: int, neighbor_class: int) -> np.ndarray:
    neighbor = np.zeros(truth.shape, dtype=bool)
    neighbor[:-1, :] |= truth[1:, :] == neighbor_class
    neighbor[1:, :] |= truth[:-1, :] == neighbor_class
    neighbor[:, :-1] |= truth[:, 1:] == neighbor_class
    neighbor[:, 1:] |= truth[:, :-1] == neighbor_class
    return (truth == side) & neighbor


def taxicab_distance_to_class(truth: np.ndarray, class_id: int,
                              max_distance: int = 5) -> np.ndarray:
    """Four-neighbor distance to a class, capped at max_distance + 1."""
    sentinel = max_distance + 1
    distance = np.full(truth.shape, sentinel, dtype=np.uint8)
    frontier = truth == class_id
    distance[frontier] = 0
    for step in range(1, max_distance + 1):
        expanded = np.zeros(truth.shape, dtype=bool)
        expanded[:-1, :] |= frontier[1:, :]
        expanded[1:, :] |= frontier[:-1, :]
        expanded[:, :-1] |= frontier[:, 1:]
        expanded[:, 1:] |= frontier[:, :-1]
        newly_reached = expanded & (distance == sentinel)
        distance[newly_reached] = step
        frontier = newly_reached
    return distance


def group_bootstrap(values: np.ndarray, groups: list[str], draws: int,
                    seed: int) -> tuple[float, list[float]]:
    by_group = {group: np.flatnonzero(np.asarray(groups) == group)
                for group in sorted(set(groups))}
    group_names = list(by_group)
    rng = np.random.default_rng(seed)
    boot = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        chosen = rng.integers(0, len(group_names), size=len(group_names))
        indices = np.concatenate([by_group[group_names[i]] for i in chosen])
        boot[draw] = values[indices].mean()
    return float(values.mean()), np.percentile(boot, [2.5, 97.5]).tolist()


def paired_group_delta(reference: dict[str, float], candidate: dict[str, float],
                       draws: int, seed: int) -> dict:
    groups = sorted(set(reference) & set(candidate))
    if not groups:
        raise ValueError("no paired specimen groups for comparison")
    delta = np.asarray([candidate[group] - reference[group] for group in groups], dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = delta[rng.integers(0, len(delta), size=(draws, len(delta)))].mean(axis=1)
    return {
        "groups": len(groups),
        "reference_mean_pct": float(np.mean([reference[group] for group in groups]) * 100),
        "candidate_mean_pct": float(np.mean([candidate[group] for group in groups]) * 100),
        "delta_pp": float(delta.mean() * 100),
        "paired_group_bootstrap_95ci_delta_pp":
            np.percentile(boot * 100, [2.5, 97.5]).tolist(),
    }


def group_class_rates(rows: dict[str, dict], groups: list[str], class_id: int) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for image_id, row in rows.items():
        grouped[row["group"]].append(float(row["class_iou"][class_id]))
    return {group: float(np.mean(values)) for group, values in grouped.items()}


def load_oof(cv_root: Path, model_roots: dict[str, Path]) -> tuple[dict, dict, dict, dict]:
    rows = {model: {} for model in model_roots}
    boundary = {model: defaultdict(lambda: defaultdict(lambda: [0, 0])) for model in model_roots}
    rings = {model: defaultdict(lambda: defaultdict(lambda: [0, 0])) for model in model_roots}
    fold_images: dict[str, list[str]] = {}
    seen: set[str] = set()

    for fold in range(1, 5):
        mask_dir = cv_root / f"oof_fold{fold}" / "masks"
        image_dir = cv_root / f"oof_fold{fold}" / "images"
        masks = {path.name.removesuffix("_mask.png"): path
                 for path in mask_dir.glob("*_mask.png")}
        if not masks or not image_dir.is_dir():
            raise FileNotFoundError(f"missing OOF images or masks in {image_dir} / {mask_dir}")
        archive_paths = {
            model: root / f"fold{fold}_oof" / "valid_probabilities.npz"
            for model, root in model_roots.items()
        }
        if any(not path.is_file() for path in archive_paths.values()):
            raise FileNotFoundError(f"missing OOF probabilities for fold {fold}: {archive_paths}")

        archives = {model: np.load(path, allow_pickle=False)
                    for model, path in archive_paths.items()}
        try:
            expected = set(masks)
            for model, archive in archives.items():
                if set(archive.files) != expected:
                    raise ValueError(f"fold {fold}: {model} prediction IDs do not match masks")
            fold_names = sorted(masks)
            for image_id in fold_names:
                if image_id in seen:
                    raise ValueError(f"duplicate OOF image across folds: {image_id}")
                seen.add(image_id)
                match = GROUP_RE.match(image_id)
                if not match:
                    raise ValueError(f"cannot parse specimen group from {image_id}")
                truth = mask_to_ids(masks[image_id])
                distance_to_al3ni = taxicab_distance_to_class(truth, 1, max_distance=5)
                eutectic_gt = truth == 3
                ring_masks = {
                    "eutectic_gt_distance1_from_al3ni": eutectic_gt & (distance_to_al3ni == 1),
                    "eutectic_gt_distance2_5_from_al3ni":
                        eutectic_gt & (distance_to_al3ni >= 2) & (distance_to_al3ni <= 5),
                    "eutectic_gt_distance_gt5_from_al3ni": eutectic_gt & (distance_to_al3ni > 5),
                }
                image_path = image_dir / f"{image_id}_image.png"
                with Image.open(image_path) as image:
                    resolution = f"{image.width}x{image.height}"
                fold_predictions = {}
                for model, archive in archives.items():
                    probabilities = np.asarray(archive[image_id], dtype=np.float32)
                    ious = class_ious(probabilities, truth)
                    prediction = probabilities.argmax(axis=0)
                    rows[model][image_id] = {
                        "fold": fold,
                        "group": match.group(1),
                        "resolution": resolution,
                        "class_iou": ious,
                        "true_class_fraction": np.asarray(
                            [(truth == class_id).mean() for class_id in range(4)],
                            dtype=np.float64,
                        ),
                        "predicted_class_fraction": np.asarray(
                            [(prediction == class_id).mean() for class_id in range(4)],
                            dtype=np.float64,
                        ),
                    }
                    fold_predictions[model] = prediction
                    for ring_name, region in ring_masks.items():
                        support_count = int(region.sum())
                        for predicted_class, metric in (
                            (3, "correct_Eutectic"),
                            (1, "Eutectic_to_Al3Ni"),
                            (0, "Eutectic_to_Al"),
                            (2, "Eutectic_to_Primary_Si"),
                        ):
                            key = (match.group(1), ring_name, metric)
                            rings[model][key]["support"][0] += support_count
                            rings[model][key]["count"][0] += int(
                                np.count_nonzero(prediction[region] == predicted_class)
                            )
                    for side, neighbor, metric in (
                        (3, 1, "Eutectic_interface_to_Al3Ni"),
                        (1, 3, "Al3Ni_interface_to_Eutectic"),
                        (3, 0, "Eutectic_interface_to_Al"),
                        (3, 2, "Eutectic_interface_to_Primary_Si"),
                        (0, 1, "Al_interface_to_Al3Ni"),
                        (1, 0, "Al3Ni_interface_to_Al"),
                        (0, 2, "Al_interface_to_Primary_Si"),
                        (2, 0, "Primary_Si_interface_to_Al"),
                    ):
                        support = adjacent_support(truth, side, neighbor)
                        key = (match.group(1), metric)
                        boundary[model][key]["support"][0] += int(support.sum())
                        boundary[model][key]["error"][0] += int(
                            np.count_nonzero(prediction[support] == neighbor)
                        )
                fold_images[image_id] = [fold, match.group(1)]
        finally:
            for archive in archives.values():
                archive.close()

    if len(seen) != 70:
        raise ValueError(f"expected exactly 70 once-only train OOF images, found {len(seen)}")
    return rows, boundary, rings, fold_images


def summarize_boundary_rates(boundary: dict, model: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for (group, metric), counts in boundary[model].items():
        grouped[metric][group][0] += counts["error"][0]
        grouped[metric][group][1] += counts["support"][0]
    result = {}
    for metric, groups in grouped.items():
        result[metric] = {
            group: error / support
            for group, (error, support) in groups.items() if support > 0
        }
    return result


def summarize_ring_rates(rings: dict, model: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for (group, ring_name, metric), counts in rings[model].items():
        name = f"{ring_name}:{metric}"
        grouped[name][group][0] += counts["count"][0]
        grouped[name][group][1] += counts["support"][0]
    return {
        metric: {group: error / support for group, (error, support) in groups.items() if support > 0}
        for metric, groups in grouped.items()
    }


def summarize_miou(rows: dict[str, dict], model: str, draws: int) -> dict:
    names = sorted(rows[model])
    groups = [rows[model][name]["group"] for name in names]
    scores = np.asarray([np.mean(rows[model][name]["class_iou"]) for name in names])
    mean, ci = group_bootstrap(scores, groups, draws, 377)
    class_iou = np.asarray([rows[model][name]["class_iou"] for name in names])
    folds = {}
    for fold in range(1, 5):
        indices = [i for i, name in enumerate(names) if rows[model][name]["fold"] == fold]
        folds[str(fold)] = float(scores[indices].mean() * 100)
    return {
        "images": len(names),
        "specimen_groups": len(set(groups)),
        "image_mean_miou_empty0_pct": float(mean * 100),
        "group_bootstrap_95ci_pct": [float(x * 100) for x in ci],
        "class_mean_iou_pct": {
            CLASS_NAMES[i]: float(class_iou[:, i].mean() * 100) for i in range(4)
        },
        "fold_image_mean_miou_pct": folds,
    }


def compare_miou(reference: dict[str, dict], candidate: dict[str, dict],
                 draws: int, seed: int) -> dict:
    names = sorted(set(reference) & set(candidate))
    if len(names) != 70:
        raise ValueError(f"paired score coverage is {len(names)}, expected 70")
    groups = [reference[name]["group"] for name in names]
    base = np.asarray([np.mean(reference[name]["class_iou"]) for name in names])
    cand = np.asarray([np.mean(candidate[name]["class_iou"]) for name in names])
    delta = cand - base
    mean, ci = group_bootstrap(delta, groups, draws, seed)
    folds = {}
    for fold in range(1, 5):
        indices = [i for i, name in enumerate(names) if candidate[name]["fold"] == fold]
        folds[str(fold)] = float(delta[indices].mean() * 100)
    base_cls = np.asarray([reference[name]["class_iou"] for name in names])
    cand_cls = np.asarray([candidate[name]["class_iou"] for name in names])
    cls_delta = (cand_cls - base_cls).mean(axis=0) * 100
    base_predicted_fraction = np.asarray([
        reference[name]["predicted_class_fraction"] for name in names
    ])
    cand_predicted_fraction = np.asarray([
        candidate[name]["predicted_class_fraction"] for name in names
    ])
    truth_fraction = np.asarray([reference[name]["true_class_fraction"] for name in names])
    predicted_fraction = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        fraction_delta = cand_predicted_fraction[:, class_id] - base_predicted_fraction[:, class_id]
        fraction_mean, fraction_ci = group_bootstrap(fraction_delta, groups, draws, seed + 100 + class_id)
        predicted_fraction[class_name] = {
            "reference_predicted_fraction_pct": float(base_predicted_fraction[:, class_id].mean() * 100),
            "candidate_predicted_fraction_pct": float(cand_predicted_fraction[:, class_id].mean() * 100),
            "ground_truth_fraction_pct": float(truth_fraction[:, class_id].mean() * 100),
            "candidate_minus_reference_pp": float(fraction_mean * 100),
            "paired_group_bootstrap_95ci_delta_pp": [float(x * 100) for x in fraction_ci],
        }
    group_delta = {
        group: float(np.mean([delta[i] for i, name in enumerate(names)
                              if reference[name]["group"] == group]) * 100)
        for group in sorted(set(groups))
    }
    resolution_strata = {}
    for resolution in sorted({reference[name]["resolution"] for name in names}):
        selected = [i for i, name in enumerate(names)
                    if reference[name]["resolution"] == resolution]
        stratum_groups = sorted({groups[i] for i in selected})
        by_group = {
            group: np.asarray([i for i in selected if groups[i] == group], dtype=np.int64)
            for group in stratum_groups
        }
        rng = np.random.default_rng(seed + len(resolution_strata) + 1)
        boot = np.empty(draws, dtype=np.float64)
        for draw in range(draws):
            chosen = rng.integers(0, len(stratum_groups), size=len(stratum_groups))
            sample = np.concatenate([by_group[stratum_groups[index]] for index in chosen])
            boot[draw] = delta[sample].mean()
        resolution_strata[resolution] = {
            "images": len(selected),
            "specimen_groups": len(stratum_groups),
            "reference_image_mean_miou_pct": float(base[selected].mean() * 100),
            "candidate_image_mean_miou_pct": float(cand[selected].mean() * 100),
            "delta_pp": float(delta[selected].mean() * 100),
            "paired_group_bootstrap_95ci_delta_pp":
                np.percentile(boot * 100, [2.5, 97.5]).tolist(),
        }
    return {
        "reference_image_mean_miou_empty0_pct": float(base.mean() * 100),
        "candidate_image_mean_miou_empty0_pct": float(cand.mean() * 100),
        "delta_pp": float(mean * 100),
        "paired_group_bootstrap_95ci_delta_pp": [float(x * 100) for x in ci],
        "class_mean_iou_delta_pp": {
            CLASS_NAMES[i]: float(cls_delta[i]) for i in range(4)
        },
        "fold_delta_pp": folds,
        "resolution_stratified": resolution_strata,
        "specimen_group_delta_summary_pp": {
            "mean": float(np.mean(list(group_delta.values()))),
            "median": float(np.median(list(group_delta.values()))),
            "minimum": float(np.min(list(group_delta.values()))),
            "maximum": float(np.max(list(group_delta.values()))),
            "positive_groups": int(sum(value > 0 for value in group_delta.values())),
            "total_groups": len(group_delta),
        },
        "per_specimen_group_delta_pp": group_delta,
        "predicted_area_fraction_comparison": predicted_fraction,
        "improved_images": int((delta > 0).sum()),
        "improved_specimen_groups": int(sum(
            np.mean(delta[np.asarray(groups) == group]) > 0 for group in set(groups)
        )),
    }


def run(cv_root: Path, model374_root: Path, model376_root: Path,
        model377_root: Path, draws: int) -> dict:
    roots = {"Model374": model374_root, "Model376": model376_root, "Model377": model377_root}
    rows, boundary, rings, fold_images = load_oof(cv_root, roots)
    scores = {name: summarize_miou(rows, name, draws) for name in roots}
    adoption = compare_miou(rows["Model374"], rows["Model377"], draws, 377)
    incremental = compare_miou(rows["Model376"], rows["Model377"], draws, 378)
    folds = adoption["fold_delta_pp"]
    gate = {
        "vs_model374_gain_at_least_0_5pp": adoption["delta_pp"] >= 0.5,
        "vs_model374_at_least_3_of_4_folds_positive":
            sum(value > 0 for value in folds.values()) >= 3,
        "vs_model374_worst_fold_at_least_minus_0_3pp": min(folds.values()) >= -0.3,
        "vs_model374_group_bootstrap_ci_lower_above_zero":
            adoption["paired_group_bootstrap_95ci_delta_pp"][0] > 0,
    }

    rates = {name: summarize_boundary_rates(boundary, name) for name in roots}
    ring_rates = {name: summarize_ring_rates(rings, name) for name in roots}
    mechanism = {}
    metric_names = sorted(set(rates["Model376"]) & set(rates["Model377"]))
    for index, metric in enumerate(metric_names):
        mechanism[metric] = paired_group_delta(
            rates["Model376"][metric], rates["Model377"][metric], draws, 379 + index
        )
    eutectic_ring_comparison = {}
    for index, metric in enumerate(sorted(set(ring_rates["Model376"]) & set(ring_rates["Model377"]))):
        if not (set(ring_rates["Model376"][metric]) & set(ring_rates["Model377"][metric])):
            continue
        eutectic_ring_comparison[metric] = paired_group_delta(
            ring_rates["Model376"][metric], ring_rates["Model377"][metric],
            draws, 500 + index,
        )
    target = rates["Model376"]["Eutectic_interface_to_Al3Ni"]
    reverse = rates["Model376"]["Al3Ni_interface_to_Eutectic"]
    target_new = rates["Model377"]["Eutectic_interface_to_Al3Ni"]
    reverse_new = rates["Model377"]["Al3Ni_interface_to_Eutectic"]
    both_groups = sorted(set(target) & set(reverse) & set(target_new) & set(reverse_new))
    swap_delta = np.asarray([
        (target_new[group] + reverse_new[group]) - (target[group] + reverse[group])
        for group in both_groups
    ])
    rng = np.random.default_rng(400)
    swap_boot = swap_delta[rng.integers(0, len(swap_delta), size=(draws, len(swap_delta)))].mean(axis=1)
    swap_sum = {
        "groups": len(both_groups),
        "model376_two_direction_rate_sum_pct": float(np.mean([
            target[group] + reverse[group] for group in both_groups
        ]) * 100),
        "model377_two_direction_rate_sum_pct": float(np.mean([
            target_new[group] + reverse_new[group] for group in both_groups
        ]) * 100),
        "paired_delta_pp": float(swap_delta.mean() * 100),
        "paired_group_bootstrap_95ci_delta_pp":
            np.percentile(swap_boot * 100, [2.5, 97.5]).tolist(),
    }
    target_delta = mechanism["Eutectic_interface_to_Al3Ni"]
    reverse_delta = mechanism["Al3Ni_interface_to_Eutectic"]
    eutectic_delta = incremental["class_mean_iou_delta_pp"]["Eutectic_Si"]
    off_target = (
        mechanism["Eutectic_interface_to_Al"]["delta_pp"] <= 0
        and mechanism["Eutectic_interface_to_Primary_Si"]["delta_pp"] <= 0
    )
    no_primary_boundary_worse = all(
        mechanism[key]["delta_pp"] <= 0 for key in (
            "Al_interface_to_Primary_Si", "Primary_Si_interface_to_Al"
        )
    )
    no_al_al3ni_worse = all(
        mechanism[key]["delta_pp"] <= 0 for key in (
            "Al_interface_to_Al3Ni", "Al3Ni_interface_to_Al"
        )
    )
    mechanism_gate = {
        "two_direction_swap_sum_reduced_at_least_2pp": swap_sum["paired_delta_pp"] <= -2.0,
        "Eutectic_to_Al3Ni_paired_group_ci_upper_below_zero":
            target_delta["paired_group_bootstrap_95ci_delta_pp"][1] < 0,
        "reverse_Al3Ni_to_Eutectic_rate_increase_at_most_1_5pp":
            reverse_delta["delta_pp"] <= 1.5,
        "Eutectic_mean_IoU_not_lower_than_Model376": eutectic_delta >= 0,
        "no_unintended_Eutectic_to_Al_or_Primary_leakage_increase": off_target,
        "no_Al_or_Primary_boundary_swap_increase": no_primary_boundary_worse,
        "no_Al_Al3Ni_boundary_swap_increase": no_al_al3ni_worse,
    }
    return {
        "scope": "70 training images, exactly-once specimen-grouped OOF; official valid/test inputs, masks, and scores are excluded",
        "metric": "per-image arithmetic mean of four class IoUs; empty union=0",
        "boundary_metric": "unique GT pixels with an axial 4-neighbor of the paired class, grouped ratio then equal group mean",
        "bootstrap": {"unit": "specimen group", "draws": draws, "seed_base": 377},
        "images": len(fold_images),
        "specimen_groups": len({value[1] for value in fold_images.values()}),
        "model_scores": scores,
        "adoption_comparison_model377_vs_model374": adoption,
        "adoption_gate": gate,
        "adoption_gate_passed": all(gate.values()),
        "incremental_comparison_model377_vs_model376": incremental,
        "directional_mechanism_model377_vs_model376": {
            "rates": mechanism,
            "Eutectic_error_by_taxicab_distance_from_Al3Ni": eutectic_ring_comparison,
            "two_direction_swap_sum": swap_sum,
            "gate": mechanism_gate,
            "gate_passed": all(mechanism_gate.values()),
        },
        "interpretation_guard": (
            "This is a single selection-conditional experiment: the direction was motivated by prior Model374/376 OOF. "
            "Passing grouped OOF gates does not establish official validation/test performance. "
            "Boundary rates count unique support pixels and are not physical phase widths. Distance rings are diagnostics, not gate inputs."
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
    result = run(args.cv_root, args.model374_root, args.model376_root,
                 args.model377_root, args.draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in (
        "directional_mechanism_model377_vs_model376", "interpretation_guard"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
