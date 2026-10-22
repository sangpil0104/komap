#!/usr/bin/env python3
"""Exploratory directional class-confusion audit on partial Model376 train OOF.

Reads only train masks and matching grouped-OOF probabilities. The current
partial run covers folds 1-3; this is descriptive evidence, not a final model
selection result or an unbiased confirmation of a pair chosen from Model374
OOF.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from model374_train_oof_error_diagnostic import mask_to_ids


CLASS_NAMES = ("Al", "Al3Ni", "Primary_Si", "Eutectic_Si")
GROUP_RE = re.compile(r"^(A\d+)")
PAIRS = ((0, 1), (1, 3))


def empty_slots(pair: tuple[int, int]) -> dict[int, list[int]]:
    return {side: [0, 0] for side in pair}  # interface pixels, predicted as partner


def add_interface_counts(slot: dict[int, list[int]], truth: np.ndarray,
                         prediction: np.ndarray, pair: tuple[int, int]) -> None:
    """Count both labeled sides of each horizontal/vertical GT pair interface."""
    left, right = pair
    for gt_a, gt_b, pred_a, pred_b in (
        (truth[:, :-1], truth[:, 1:], prediction[:, :-1], prediction[:, 1:]),
        (truth[:-1, :], truth[1:, :], prediction[:-1, :], prediction[1:, :]),
    ):
        left_right = (gt_a == left) & (gt_b == right)
        right_left = (gt_a == right) & (gt_b == left)
        for side, interface, predicted in (
            (left, left_right, pred_a),
            (right, left_right, pred_b),
            (right, right_left, pred_a),
            (left, right_left, pred_b),
        ):
            n = int(interface.sum())
            if n:
                slot[side][0] += n
                partner = right if side == left else left
                slot[side][1] += int(np.count_nonzero(predicted[interface] == partner))


def add_group_counts(groups: dict, group: str, pair: tuple[int, int],
                     truth: np.ndarray, prediction: np.ndarray) -> None:
    slots = groups.setdefault(group, {}).setdefault(pair, empty_slots(pair))
    add_interface_counts(slots, truth, prediction, pair)


def summarize_scope(groups: dict, pair: tuple[int, int], draws: int,
                    rng: np.random.Generator) -> dict:
    left, right = pair
    used = []
    for group, pair_map in groups.items():
        slots = pair_map.get(pair)
        if slots and slots[left][0] and slots[right][0]:
            used.append((group, slots))
    if not used:
        return {"groups_with_both_sides": 0, "left_minus_right_delta_pp": None}

    side_rates = {
        side: np.asarray([slots[side][1] / slots[side][0] for _, slots in used])
        for side in pair
    }
    delta = side_rates[left] - side_rates[right]
    indices = rng.integers(0, len(delta), size=(draws, len(delta)))
    boot = delta[indices].mean(axis=1) * 100.0
    return {
        "groups_with_both_sides": len(used),
        "interface_pixels": {
            CLASS_NAMES[side]: int(sum(slots[side][0] for _, slots in used))
            for side in pair
        },
        "group_mean_partner_swap_rate_pct": {
            CLASS_NAMES[side]: float(side_rates[side].mean() * 100.0)
            for side in pair
        },
        "left_minus_right_delta_pp": float(delta.mean() * 100.0),
        "group_bootstrap_95ci_delta_pp": np.percentile(boot, [2.5, 97.5]).tolist(),
        "fraction_groups_delta_negative": float(np.mean(delta < 0)),
    }


def run(cv_root: Path, baseline_root: Path, candidate_root: Path,
        folds: tuple[int, ...], draws: int) -> dict:
    if not folds or len(set(folds)) != len(folds) or any(f not in (1, 2, 3, 4) for f in folds):
        raise ValueError("folds must be unique values from 1 through 4")
    if draws < 100:
        raise ValueError("use at least 100 bootstrap draws")

    roots = {"Model374": baseline_root, "Model376": candidate_root}
    groups = {model: {} for model in roots}
    fold_groups = {model: {} for model in roots}
    seen: set[str] = set()
    fold_image_counts = {}

    for fold in folds:
        mask_dir = cv_root / f"oof_fold{fold}" / "masks"
        masks = {p.name.removesuffix("_mask.png"): p for p in mask_dir.glob("*_mask.png")}
        archives = {
            model: root / f"fold{fold}_oof" / "valid_probabilities.npz"
            for model, root in roots.items()
        }
        if not masks or any(not path.is_file() for path in archives.values()):
            raise FileNotFoundError(f"missing masks or OOF probabilities for fold {fold}")
        with np.load(archives["Model374"], allow_pickle=False) as base, \
             np.load(archives["Model376"], allow_pickle=False) as candidate:
            if set(base.files) != set(masks) or set(candidate.files) != set(masks):
                raise ValueError(f"fold {fold}: OOF image IDs do not exactly match masks")
            if set(base.files) != set(candidate.files):
                raise ValueError(f"fold {fold}: baseline/candidate OOF IDs differ")
            per_fold = {model: {} for model in roots}
            for image_id in sorted(masks):
                if image_id in seen:
                    raise ValueError(f"duplicate OOF image across folds: {image_id}")
                seen.add(image_id)
                match = GROUP_RE.match(image_id)
                if not match:
                    raise ValueError(f"cannot parse specimen group from {image_id}")
                group = match.group(1)
                truth = mask_to_ids(masks[image_id])
                for model, archive in (("Model374", base), ("Model376", candidate)):
                    probabilities = np.asarray(archive[image_id], dtype=np.float32)
                    if probabilities.shape != (4, *truth.shape) or not np.isfinite(probabilities).all():
                        raise ValueError(f"invalid probability map: {model} {image_id}")
                    if float(np.max(np.abs(probabilities.sum(axis=0) - 1.0))) > 0.03:
                        raise ValueError(f"unnormalized probability map: {model} {image_id}")
                    prediction = probabilities.argmax(axis=0).astype(np.uint8)
                    for pair in PAIRS:
                        add_group_counts(groups[model], group, pair, truth, prediction)
                        add_group_counts(per_fold[model], group, pair, truth, prediction)
            for model in roots:
                fold_groups[model][fold] = per_fold[model]
        fold_image_counts[str(fold)] = len(masks)

    model_results = {}
    for model in roots:
        pair_results = {}
        for pair in PAIRS:
            label = f"{CLASS_NAMES[pair[0]]}_vs_{CLASS_NAMES[pair[1]]}"
            rng = np.random.default_rng(37600 + pair[0] * 10 + pair[1])
            pair_results[label] = {
                "combined_folds": summarize_scope(groups[model], pair, draws, rng),
                "per_fold": {
                    str(fold): summarize_scope(fold_groups[model][fold], pair, draws, rng)
                    for fold in folds
                },
            }
        model_results[model] = pair_results

    complete = set(folds) == {1, 2, 3, 4}
    return {
        "scope": (
            "complete 70-image train-only grouped OOF; no official valid/test images or masks"
            if complete else "partial train-only grouped OOF; no official valid/test images or masks"
        ),
        "folds": list(folds),
        "images_by_fold": fold_image_counts,
        "images": len(seen),
        "specimen_groups": len({GROUP_RE.match(name).group(1) for name in seen}),
        "metric": "per-specimen-group mean directed partner-swap rate on axial GT interfaces",
        "bootstrap": {"unit": "specimen group", "draws": draws, "seed_base": 37600},
        "class_pairs": [list(pair) for pair in PAIRS],
        "results": model_results,
        "interpretation_guard": (
            "The pair/direction was motivated by prior Model374 OOF, so this same-data comparison is "
            "selection-conditional, not an unbiased confirmation or evidence to tune loss weight. "
            "Pixel-run interface counts are not physical widths."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.cv_root, args.baseline_root, args.candidate_root,
                 tuple(args.folds), args.draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "images": result["images"], "specimen_groups": result["specimen_groups"],
        "folds": result["folds"], "results": result["results"],
        "interpretation_guard": result["interpretation_guard"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
