#!/usr/bin/env python3
"""Check whether pairwise boundary errors are explained by thin GT structures.

Uses Model374 exactly-once grouped train OOF only. Interface side thickness is
estimated as contiguous GT run length along the local boundary normal; it is a
pixel-space proxy, not physical metallographic thickness.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from model374_train_oof_error_diagnostic import mask_to_ids


N_CLASSES = 4
CLASS_NAMES = ["Al", "Al3Ni", "Primary_Si", "Eutectic_Si"]
GROUP_RE = re.compile(r"^(A\d+)")
PAIR_BINS = ("thin_le_2px", "medium_3_to_5px", "thick_ge_6px")


def run_lengths(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Same-label run sizes in the four axial directions."""
    height, width = mask.shape
    left = np.ones((height, width), dtype=np.uint16)
    right = np.ones_like(left)
    up = np.ones_like(left)
    down = np.ones_like(left)
    for col in range(1, width):
        left[:, col] = np.where(mask[:, col] == mask[:, col - 1], left[:, col - 1] + 1, 1)
    for col in range(width - 2, -1, -1):
        right[:, col] = np.where(mask[:, col] == mask[:, col + 1], right[:, col + 1] + 1, 1)
    for row in range(1, height):
        up[row, :] = np.where(mask[row, :] == mask[row - 1, :], up[row - 1, :] + 1, 1)
    for row in range(height - 2, -1, -1):
        down[row, :] = np.where(mask[row, :] == mask[row + 1, :], down[row + 1, :] + 1, 1)
    return left, right, up, down


def width_bin(widths: np.ndarray) -> np.ndarray:
    return np.where(widths <= 2, 0, np.where(widths <= 5, 1, 2)).astype(np.int8)


def add_samples(slot: np.ndarray, bins: np.ndarray, pred_side: np.ndarray,
                side_class: int, partner_class: int) -> None:
    direct = (pred_side == partner_class).astype(np.int64)
    any_wrong = (pred_side != side_class).astype(np.int64)
    slot[:, 0] += np.bincount(bins, minlength=3).astype(np.int64)
    slot[:, 1] += np.bincount(bins, weights=direct, minlength=3).astype(np.int64)
    slot[:, 2] += np.bincount(bins, weights=any_wrong, minlength=3).astype(np.int64)


def collect_image(mask: np.ndarray, pred: np.ndarray) -> dict[tuple[int, int], dict[int, np.ndarray]]:
    left_run, right_run, up_run, down_run = run_lengths(mask)
    result = {(left, right): {
        left: np.zeros((3, 3), dtype=np.int64),
        right: np.zeros((3, 3), dtype=np.int64),
    } for left in range(N_CLASSES) for right in range(left + 1, N_CLASSES)}

    for axis in ("horizontal", "vertical"):
        if axis == "horizontal":
            a, b = mask[:, :-1], mask[:, 1:]
            pa, pb = pred[:, :-1], pred[:, 1:]
            a_left_run, b_right_run = left_run[:, :-1], right_run[:, 1:]
        else:
            a, b = mask[:-1, :], mask[1:, :]
            pa, pb = pred[:-1, :], pred[1:, :]
            a_left_run, b_right_run = up_run[:-1, :], down_run[1:, :]

        different = a != b
        for left, right in result:
            is_pair = different & (np.minimum(a, b) == left) & (np.maximum(a, b) == right)
            a_is_left, b_is_left = is_pair & (a == left), is_pair & (b == left)
            a_is_right, b_is_right = is_pair & (a == right), is_pair & (b == right)

            for side, partner, at_a, at_b in (
                (left, right, a_is_left, b_is_left),
                (right, left, a_is_right, b_is_right),
            ):
                widths = np.concatenate((a_left_run[at_a], b_right_run[at_b])).astype(np.int32)
                predictions = np.concatenate((pa[at_a], pb[at_b]))
                add_samples(result[left, right][side], width_bin(widths), predictions, side, partner)
    return result


def summarize(groups: dict[str, dict], draws: int, seed: int) -> dict:
    pair_results = {}
    rng = np.random.default_rng(seed)
    for left in range(N_CLASSES):
        for right in range(left + 1, N_CLASSES):
            group_rates = {left: [], right: []}
            aggregate = {left: np.zeros((3, 3), dtype=np.int64),
                         right: np.zeros((3, 3), dtype=np.int64)}
            relevant_groups = []
            for group, pair_map in groups.items():
                slots = pair_map.get((left, right))
                if slots is None:
                    continue
                left_n = int(slots[left][:, 0].sum())
                right_n = int(slots[right][:, 0].sum())
                if left_n == 0 or right_n == 0:
                    continue
                relevant_groups.append(group)
                for side in (left, right):
                    aggregate[side] += slots[side]
                    total = float(slots[side][:, 0].sum())
                    group_rates[side].append(float(slots[side][:, 1].sum() / total))

            bins = {}
            for side, partner in ((left, right), (right, left)):
                bin_rows = {}
                for bin_index, bin_name in enumerate(PAIR_BINS):
                    n, direct, any_wrong = map(int, aggregate[side][bin_index])
                    bin_rows[bin_name] = {
                        "interface_samples": n,
                        "predicted_as_partner": direct,
                        "any_wrong_class": any_wrong,
                        "partner_swap_rate": direct / n if n else None,
                        "any_error_rate": any_wrong / n if n else None,
                    }
                total_n = int(aggregate[side][:, 0].sum())
                total_direct = int(aggregate[side][:, 1].sum())
                bin_rows["all_thicknesses"] = {
                    "interface_samples": total_n,
                    "predicted_as_partner": total_direct,
                    "partner_swap_rate": total_direct / total_n if total_n else None,
                    "mean_group_partner_swap_rate": float(np.mean(group_rates[side]))
                    if group_rates[side] else None,
                }
                bins[CLASS_NAMES[side]] = bin_rows

            thickness_group_deltas = {}
            for bin_index, bin_name in enumerate(PAIR_BINS):
                left_rates, right_rates = [], []
                for group, pair_map in groups.items():
                    slots = pair_map.get((left, right))
                    if slots is None:
                        continue
                    left_n = int(slots[left][bin_index, 0])
                    right_n = int(slots[right][bin_index, 0])
                    if left_n == 0 or right_n == 0:
                        continue
                    left_rates.append(float(slots[left][bin_index, 1] / left_n))
                    right_rates.append(float(slots[right][bin_index, 1] / right_n))
                if left_rates:
                    delta_by_group = np.asarray(left_rates) - np.asarray(right_rates)
                    boot_bin = np.empty(draws, dtype=np.float64)
                    for draw in range(draws):
                        sample = rng.integers(0, len(delta_by_group), size=len(delta_by_group))
                        boot_bin[draw] = delta_by_group[sample].mean() * 100.0
                    thickness_group_deltas[bin_name] = {
                        "groups_with_both_sides_in_bin": len(delta_by_group),
                        "mean_group_left_minus_right_partner_swap_rate_percentage_points":
                            float(delta_by_group.mean() * 100.0),
                        "group_bootstrap_95ci_percentage_points":
                            np.percentile(boot_bin, [2.5, 97.5]).tolist(),
                        "fraction_groups_left_side_rate_higher":
                            float(np.mean(delta_by_group > 0)),
                    }
                else:
                    thickness_group_deltas[bin_name] = {
                        "groups_with_both_sides_in_bin": 0,
                        "mean_group_left_minus_right_partner_swap_rate_percentage_points": None,
                        "group_bootstrap_95ci_percentage_points": None,
                        "fraction_groups_left_side_rate_higher": None,
                    }

            if relevant_groups:
                group_delta = np.asarray(group_rates[left]) - np.asarray(group_rates[right])
                boot = np.empty(draws, dtype=np.float64)
                for draw in range(draws):
                    sample = rng.integers(0, len(group_delta), size=len(group_delta))
                    boot[draw] = group_delta[sample].mean() * 100.0
                ci = np.percentile(boot, [2.5, 97.5]).tolist()
                sign_fraction = float(np.mean(group_delta > 0))
                average_delta = float(group_delta.mean() * 100.0)
            else:
                ci, sign_fraction, average_delta = None, None, None
            pair_results[f"{CLASS_NAMES[left]}_vs_{CLASS_NAMES[right]}"] = {
                "groups_with_this_interface": len(relevant_groups),
                "group_mean_left_minus_right_partner_swap_rate_percentage_points": average_delta,
                "group_bootstrap_95ci_percentage_points": ci,
                f"fraction_groups_{CLASS_NAMES[left]}_side_rate_higher": sign_fraction,
                "thickness_stratified_side_rates": bins,
                "thickness_stratified_group_contrasts": thickness_group_deltas,
            }
    return {
        "groups": len(groups),
        "bootstrap_draws": draws,
        "pairs": pair_results,
        "interpretation_guard": (
            "Thickness is an axial pixel-run proxy, not physical width; bins pool interfaces "
            "across resolutions and groups. Pairwise rates are descriptive and do not prove "
            "annotation bias or geometric displacement."
        ),
    }


def run_audit(cv_root: Path, oof_root: Path, draws: int, model_id: int = 374) -> dict:
    grouped_counts: dict[str, dict] = {}
    seen = set()
    folds = {}
    for fold in range(1, 5):
        mask_dir = cv_root / f"oof_fold{fold}" / "masks"
        archive_path = oof_root / f"fold{fold}_oof" / "valid_probabilities.npz"
        if not mask_dir.is_dir() or not archive_path.is_file():
            raise FileNotFoundError(f"missing fold {fold} OOF input")
        masks = {p.name.removesuffix("_mask.png"): p for p in mask_dir.glob("*_mask.png")}
        with np.load(archive_path, allow_pickle=False) as archive:
            if set(archive.files) != set(masks):
                raise ValueError(f"fold {fold}: probability/mask IDs differ")
            for image_id in sorted(masks):
                if image_id in seen:
                    raise ValueError(f"duplicate OOF image {image_id}")
                seen.add(image_id)
                match = GROUP_RE.match(image_id)
                if not match:
                    raise ValueError(f"cannot parse specimen group from {image_id}")
                truth = mask_to_ids(masks[image_id])
                probabilities = np.asarray(archive[image_id], dtype=np.float32)
                if probabilities.shape != (N_CLASSES, *truth.shape):
                    raise ValueError(f"probability/mask shape mismatch for {image_id}")
                if not np.isfinite(probabilities).all():
                    raise ValueError(f"nonfinite probabilities for {image_id}")
                if float(np.max(np.abs(probabilities.sum(axis=0) - 1.0))) > 0.03:
                    raise ValueError(f"unnormalized probabilities for {image_id}")
                pred = probabilities.argmax(axis=0).astype(np.uint8)
                image_counts = collect_image(truth, pred)
                group = match.group(1)
                if group not in grouped_counts:
                    grouped_counts[group] = {}
                for pair, side_map in image_counts.items():
                    if pair not in grouped_counts[group]:
                        grouped_counts[group][pair] = {
                            side: np.zeros((3, 3), dtype=np.int64) for side in side_map
                        }
                    for side, data in side_map.items():
                        grouped_counts[group][pair][side] += data
            folds[str(fold)] = len(masks)
    if len(seen) != 70 or len(grouped_counts) != 59:
        raise ValueError(f"expected 70 images/59 groups, found {len(seen)}/{len(grouped_counts)}")
    return {
        "model": model_id,
        "scope": f"Model{model_id} training data only; exactly-once specimen-grouped OOF",
        "metric_context": "official image-wise four-class mean IoU, empty union=0",
        "images": len(seen),
        "fold_image_counts": folds,
        "thickness_definition": "contiguous same-class GT run length along the interface normal",
        "thickness_bins_pixels": list(PAIR_BINS),
        "grouped_pair_interface_results": summarize(grouped_counts, draws, 374),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--model-id", type=int, default=374)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(args.cv_root, args.oof_root, args.draws, args.model_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
