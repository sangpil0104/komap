#!/usr/bin/env python3
"""Stratify Model374 train-only grouped OOF errors by capture conditions.

This is descriptive and CPU-only. It reads only exactly-once grouped OOF
probabilities, their training masks, and matching training images; validation
and test data are not loaded.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from model374_train_oof_error_diagnostic import boundary_band, image_ious, mask_to_ids


N_CLASSES = 4
CLASS_NAMES = ["Al", "Al3Ni", "Primary_Si", "Eutectic_Si"]
STEP_RE = re.compile(r"_step(\d+)_")
GROUP_RE = re.compile(r"^(A\d+)")


def corr(x: list[float], y: list[float]) -> float | None:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def pair_interface_stats(truth: np.ndarray, pred: np.ndarray) -> dict[tuple[int, int], list[int]]:
    """Count prediction swaps on each side of each GT class-pair interface.

    Each tuple is [GT-left-side pixels, left->right mistakes,
                   GT-right-side pixels, right->left mistakes].
    Horizontal and vertical interface edges are both counted.
    """
    stats = {(left, right): [0, 0, 0, 0]
             for left in range(N_CLASSES) for right in range(left + 1, N_CLASSES)}
    for a, b, pa, pb in (
        (truth[:, :-1], truth[:, 1:], pred[:, :-1], pred[:, 1:]),
        (truth[:-1, :], truth[1:, :], pred[:-1, :], pred[1:, :]),
    ):
        changed = a != b
        for left, right in stats:
            is_pair = changed & (np.minimum(a, b) == left) & (np.maximum(a, b) == right)
            a_is_left = is_pair & (a == left)
            a_is_right = is_pair & (a == right)
            left_den = int(a_is_left.sum() + a_is_right.sum())
            left_errors = int(((a_is_left & (pa == right)).sum()
                               + (a_is_right & (pb == right)).sum()))
            right_den = left_den
            right_errors = int(((a_is_right & (pa == left)).sum()
                                + (a_is_left & (pb == left)).sum()))
            # Re-index the four totals by GT class, not by edge orientation.
            stats[left, right][0] += left_den
            stats[left, right][1] += left_errors
            stats[left, right][2] += right_den
            stats[left, right][3] += right_errors
    return stats


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"images": 0}
    ious = np.asarray([row["class_iou_empty0"] for row in rows], dtype=np.float64)
    return {
        "images": len(rows),
        "image_mean_miou_empty0": float(ious.mean()),
        "class_mean_iou_empty0": {
            name: float(ious[:, index].mean()) for index, name in enumerate(CLASS_NAMES)
        },
        "mean_gt_boundary_band_fraction": float(np.mean([
            row["gt_boundary_band_fraction"] for row in rows
        ])),
        "mean_error_fraction_in_gt_boundary_band": float(np.mean([
            row["wrong_fraction_in_gt_boundary_band"] for row in rows
        ])),
        "mean_gray_mean": float(np.mean([row["gray_mean"] for row in rows])),
        "mean_gray_std": float(np.mean([row["gray_std"] for row in rows])),
    }


def group_bootstrap_contrast(rows: list[dict], selected: np.ndarray,
                            draws: int = 20000, seed: int = 374) -> dict:
    """Bootstrap selected-vs-other image means by resampling specimen groups."""
    group_names = sorted({row["specimen_group"] for row in rows})
    group_lookup = {name: index for index, name in enumerate(group_names)}
    scores = np.asarray([row["class_iou_empty0"] for row in rows], dtype=np.float64)
    group_index = np.asarray([group_lookup[row["specimen_group"]] for row in rows], dtype=np.int64)
    if selected.shape != (len(rows),) or selected.all() or (~selected).all():
        raise ValueError("bootstrap contrast must select a nonempty proper subset of images")
    selected_count = np.bincount(group_index[selected], minlength=len(group_names)).astype(np.float64)
    other_count = np.bincount(group_index[~selected], minlength=len(group_names)).astype(np.float64)
    selected_sum = np.zeros((len(group_names), scores.shape[1]), dtype=np.float64)
    other_sum = np.zeros_like(selected_sum)
    np.add.at(selected_sum, group_index[selected], scores[selected])
    np.add.at(other_sum, group_index[~selected], scores[~selected])
    observed_sel = scores[selected].mean(axis=0)
    observed_other = scores[~selected].mean(axis=0)

    rng = np.random.default_rng(seed)
    deltas = np.empty((draws, scores.shape[1]), dtype=np.float64)
    for draw in range(draws):
        sampled = rng.integers(0, len(group_names), size=len(group_names))
        multiplicity = np.bincount(sampled, minlength=len(group_names)).astype(np.float64)
        sel_n = float(np.dot(multiplicity, selected_count))
        other_n = float(np.dot(multiplicity, other_count))
        if sel_n == 0 or other_n == 0:
            deltas[draw] = np.nan
            continue
        sel = multiplicity @ selected_sum / sel_n
        other = multiplicity @ other_sum / other_n
        deltas[draw] = (sel - other) * 100.0

    valid = deltas[np.isfinite(deltas).all(axis=1)]
    ci = np.percentile(valid, [2.5, 97.5], axis=0)
    return {
        "images_selected": int(selected.sum()),
        "images_other": int((~selected).sum()),
        "observed_delta_selected_minus_other_percentage_points": {
            "image_mean_miou_empty0": float((observed_sel.mean() - observed_other.mean()) * 100),
            **{name: float((observed_sel[index] - observed_other[index]) * 100)
               for index, name in enumerate(CLASS_NAMES)},
        },
        "specimen_group_bootstrap_95ci_percentage_points": {
            "image_mean_miou_empty0": ci[:, :].mean(axis=1).tolist()
            if scores.shape[1] == 1 else np.percentile(valid.mean(axis=1), [2.5, 97.5]).tolist(),
            **{name: ci[:, index].tolist() for index, name in enumerate(CLASS_NAMES)},
        },
        "bootstrap_draws": int(len(valid)),
        "interpretation_guard": "descriptive subgroup contrast; not a test-set score or causal effect",
    }


def run(cv_root: Path, oof_root: Path, images_root: Path, boundary_radius: int) -> dict:
    rows: list[dict] = []
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    boundary_confusion = np.zeros_like(confusion)
    pair_stats = {(left, right): [0, 0, 0, 0]
                  for left in range(N_CLASSES) for right in range(left + 1, N_CLASSES)}
    seen: set[str] = set()

    for fold in range(1, 5):
        mask_dir = cv_root / f"oof_fold{fold}" / "masks"
        archive_path = oof_root / f"fold{fold}_oof" / "valid_probabilities.npz"
        if not mask_dir.is_dir() or not archive_path.is_file():
            raise FileNotFoundError(f"missing OOF fold {fold} input: {mask_dir} or {archive_path}")
        masks = {p.name.removesuffix("_mask.png"): p for p in mask_dir.glob("*_mask.png")}
        with np.load(archive_path, allow_pickle=False) as archive:
            if set(archive.files) != set(masks):
                raise ValueError(f"fold {fold}: probability/mask IDs differ")
            for image_id in sorted(masks):
                if image_id in seen:
                    raise ValueError(f"duplicate OOF image {image_id}")
                seen.add(image_id)
                image_path = images_root / f"{image_id}_image.png"
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                match = STEP_RE.search(image_id)
                group = GROUP_RE.match(image_id)
                if not match or not group:
                    raise ValueError(f"cannot parse step/specimen from {image_id}")

                probabilities = np.asarray(archive[image_id], dtype=np.float32)
                truth = mask_to_ids(masks[image_id])
                if probabilities.shape != (N_CLASSES, *truth.shape):
                    raise ValueError(f"shape mismatch for {image_id}: {probabilities.shape}, {truth.shape}")
                if not np.isfinite(probabilities).all():
                    raise ValueError(f"nonfinite probability values for {image_id}")
                if float(np.max(np.abs(probabilities.sum(axis=0) - 1.0))) > 0.03:
                    raise ValueError(f"unnormalized probabilities for {image_id}")

                pred = probabilities.argmax(axis=0).astype(np.uint8)
                band = boundary_band(truth, boundary_radius)
                wrong = pred != truth
                matrix = np.bincount(
                    truth.reshape(-1).astype(np.int64) * N_CLASSES + pred.reshape(-1).astype(np.int64),
                    minlength=N_CLASSES * N_CLASSES,
                ).reshape(N_CLASSES, N_CLASSES)
                band_matrix = np.bincount(
                    truth[band].astype(np.int64) * N_CLASSES + pred[band].astype(np.int64),
                    minlength=N_CLASSES * N_CLASSES,
                ).reshape(N_CLASSES, N_CLASSES)
                confusion += matrix
                boundary_confusion += band_matrix
                per_image_pair_stats = pair_interface_stats(truth, pred)
                for pair, counts in per_image_pair_stats.items():
                    for index, count in enumerate(counts):
                        pair_stats[pair][index] += count

                with Image.open(image_path) as im:
                    width, height = im.size
                    gray = np.asarray(im.convert("L"), dtype=np.float32)
                if gray.shape != truth.shape:
                    raise ValueError(f"image/mask resolution mismatch for {image_id}")
                rows.append({
                    "image_id": image_id,
                    "specimen_group": group.group(1),
                    "fold": fold,
                    "step": f"step{match.group(1)}",
                    "resolution": f"{width}x{height}",
                    "class_iou_empty0": image_ious(pred, truth),
                    "image_miou_empty0": float(np.mean(image_ious(pred, truth))),
                    "gt_boundary_band_fraction": float(band.mean()),
                    "wrong_fraction_in_gt_boundary_band": float((wrong & band).sum() / max(1, wrong.sum())),
                    "gray_mean": float(gray.mean()),
                    "gray_std": float(gray.std()),
                })

    if len(rows) != 70 or len({row["specimen_group"] for row in rows}) != 59:
        raise ValueError(f"expected 70 images / 59 groups, found {len(rows)} / "
                         f"{len({row['specimen_group'] for row in rows})}")

    def grouped(key: str) -> dict:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            buckets[str(row[key])].append(row)
        return {value: summarize(buckets[value]) for value in sorted(buckets)}

    resolution_step: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        resolution_step[f"{row['resolution']}__{row['step']}"].append(row)
    boundary_threshold = float(np.quantile(
        [row["gt_boundary_band_fraction"] for row in rows], 2 / 3
    ))
    group_bootstrap = {
        "675x480_vs_other_resolutions": group_bootstrap_contrast(
            rows, np.asarray([row["resolution"] == "675x480" for row in rows])
        ),
        "step4_vs_other_steps": group_bootstrap_contrast(
            rows, np.asarray([row["step"] == "step4" for row in rows]), seed=375
        ),
        "top_boundary_density_tertile_vs_lower_two_thirds": {
            "boundary_fraction_threshold": boundary_threshold,
            **group_bootstrap_contrast(
                rows,
                np.asarray([row["gt_boundary_band_fraction"] >= boundary_threshold for row in rows]),
                seed=376,
            ),
        },
    }

    pairwise_boundary = {}
    for (left, right), counts in pair_stats.items():
        left_den, left_errors, right_den, right_errors = counts
        left_rate = left_errors / left_den if left_den else 0.0
        right_rate = right_errors / right_den if right_den else 0.0
        pairwise_boundary[f"{CLASS_NAMES[left]}_vs_{CLASS_NAMES[right]}"] = {
            f"{CLASS_NAMES[left]}_gt_interface_pixels": left_den,
            f"{CLASS_NAMES[left]}_interface_pred_as_{CLASS_NAMES[right]}": left_errors,
            f"{CLASS_NAMES[right]}_gt_interface_pixels": right_den,
            f"{CLASS_NAMES[right]}_interface_pred_as_{CLASS_NAMES[left]}": right_errors,
            f"{CLASS_NAMES[left]}_to_{CLASS_NAMES[right]}_rate": left_rate,
            f"{CLASS_NAMES[right]}_to_{CLASS_NAMES[left]}_rate": right_rate,
            "directional_rate_difference": left_rate - right_rate,
        }

    result = {
        "model": 374,
        "scope": "70 train images, exactly-once specimen-grouped OOF; valid/test excluded",
        "metric": "image-wise mean of four class IoUs; empty union=0",
        "boundary_radius_pixels": boundary_radius,
        "images": len(rows),
        "specimen_groups": len({row["specimen_group"] for row in rows}),
        "overall": summarize(rows),
        "by_resolution": grouped("resolution"),
        "by_step": grouped("step"),
        "by_resolution_and_step": {
            value: summarize(group_rows) for value, group_rows in sorted(resolution_step.items())
        },
        "group_bootstrap_subgroup_contrasts": group_bootstrap,
        "correlation_across_images": {
            "image_miou_vs_gt_boundary_fraction_pearson": corr(
                [row["gt_boundary_band_fraction"] for row in rows],
                [row["image_miou_empty0"] for row in rows],
            ),
            "image_miou_vs_gray_std_pearson": corr(
                [row["gray_std"] for row in rows], [row["image_miou_empty0"] for row in rows]
            ),
            "image_miou_vs_gray_mean_pearson": corr(
                [row["gray_mean"] for row in rows], [row["image_miou_empty0"] for row in rows]
            ),
        },
        "boundary_pair_confusion_pixels": boundary_confusion.tolist(),
        "boundary_pair_directional_errors": pairwise_boundary,
        "full_confusion_pixels": confusion.tolist(),
        "per_image": rows,
        "interpretation_guard": (
            "Stratum means are descriptive; sample counts are small and are not independent "
            "of specimen groups. Pairwise rates use GT interface pixels; asymmetry may reflect "
            "class-direction bias but is not a signed geometric boundary displacement."
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--boundary-radius", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.cv_root, args.oof_root, args.images_root, args.boundary_radius)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    brief = {key: value for key, value in result.items() if key not in ("per_image", "full_confusion_pixels")}
    print(json.dumps(brief, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
