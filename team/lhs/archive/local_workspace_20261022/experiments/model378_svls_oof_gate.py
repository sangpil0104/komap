#!/usr/bin/env python3
"""Paired train-OOF gates for Model378 SVLS; never reads valid/test data."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from model374_train_oof_error_diagnostic import boundary_band, image_ious, mask_to_ids


GROUP_RE = re.compile(r"^(A\d+)")
STEP_RE = re.compile(r"_step(\d+)_")
CLASS_NAMES = ("Al", "Al3Ni", "Primary_Si", "Eutectic_Si")


def _check_probs(probabilities: np.ndarray, expected_shape: tuple[int, int]) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.shape != (4, *expected_shape):
        raise ValueError(f"probability/mask shape mismatch: {probabilities.shape} != {(4, *expected_shape)}")
    if not np.isfinite(probabilities).all():
        raise ValueError("nonfinite OOF probability")
    if float(np.max(np.abs(probabilities.sum(axis=0) - 1.0))) > 0.03:
        raise ValueError("OOF probabilities are not normalized")
    return probabilities


def _interface_support(truth: np.ndarray, class_id: int = 3) -> np.ndarray:
    """GT pixels of class_id adjacent axially to any unlike class."""
    support = np.zeros(truth.shape, dtype=bool)
    same = truth == class_id
    support[1:, :] |= same[1:, :] & (truth[:-1, :] != class_id)
    support[:-1, :] |= same[:-1, :] & (truth[1:, :] != class_id)
    support[:, 1:] |= same[:, 1:] & (truth[:, :-1] != class_id)
    support[:, :-1] |= same[:, :-1] & (truth[:, 1:] != class_id)
    return support


def _load_rows(cv_root: Path, model_roots: dict[str, Path], expected_images: int = 70) -> tuple[dict, dict]:
    rows: dict[str, dict[str, dict]] = {name: {} for name in model_roots}
    folds: dict[str, dict[str, list[str]]] = {name: {} for name in model_roots}
    seen_by_fold: dict[int, set[str]] = defaultdict(set)
    group_fold: dict[str, int] = {}

    for fold in range(1, 5):
        mask_dir = cv_root / f"oof_fold{fold}" / "masks"
        masks = {path.name.removesuffix("_mask.png"): path
                 for path in mask_dir.glob("*_mask.png")}
        if not masks:
            raise FileNotFoundError(f"no masks in {mask_dir}")
        archives = {}
        try:
            for name, root in model_roots.items():
                path = root / f"fold{fold}_oof" / "valid_probabilities.npz"
                if not path.is_file():
                    raise FileNotFoundError(path)
                archives[name] = np.load(path, allow_pickle=False)
                if set(archives[name].files) != set(masks):
                    raise ValueError(f"{name} fold{fold}: image IDs differ from OOF masks")

            for image_id, mask_path in masks.items():
                if image_id in seen_by_fold[fold]:
                    raise ValueError(f"duplicate image {image_id} in fold {fold}")
                seen_by_fold[fold].add(image_id)
                group_match, step_match = GROUP_RE.match(image_id), STEP_RE.search(image_id)
                if not group_match or not step_match:
                    raise ValueError(f"cannot parse specimen group/step from {image_id}")
                truth = mask_to_ids(mask_path)
                group, step = group_match.group(1), int(step_match.group(1))
                if group in group_fold and group_fold[group] != fold:
                    raise ValueError(f"specimen group {group} is split across OOF folds")
                group_fold[group] = fold
                interface = _interface_support(truth)
                band = boundary_band(truth, radius=1)
                if not band.any() or not interface.any():
                    raise ValueError(f"empty boundary/interface support in {image_id}")

                for name, archive in archives.items():
                    probabilities = _check_probs(archive[image_id], truth.shape)
                    prediction = probabilities.argmax(axis=0)
                    class_iou = np.asarray(image_ious(prediction, truth), dtype=np.float64)
                    rows[name][image_id] = {
                        "fold": fold,
                        "group": group,
                        "step": step,
                        "image_miou": float(class_iou.mean()),
                        "class_iou": class_iou,
                        "boundary_pixels": int(band.sum()),
                        "boundary_errors": int(np.count_nonzero((prediction != truth) & band)),
                        "eutectic_interface_pixels": int(interface.sum()),
                        "eutectic_interface_true_positive": int(
                            np.count_nonzero((prediction == 3) & interface)
                        ),
                    }
        finally:
            for archive in archives.values():
                archive.close()

    ids = set(rows[next(iter(model_roots))])
    if len(ids) != expected_images:
        raise ValueError(f"expected {expected_images} once-only OOF images, found {len(ids)}")
    if any(set(value) != ids for value in rows.values()):
        raise ValueError("models do not have the same exactly-once OOF image set")
    groups = {row["group"] for row in rows[next(iter(model_roots))].values()}
    if len(groups) != 59:
        raise ValueError(f"expected 59 specimen groups, found {len(groups)}")
    for name, model_rows in rows.items():
        fold_groups: dict[str, list[str]] = {}
        for image_id, row in model_rows.items():
            fold_groups.setdefault(str(row["fold"]), []).append(image_id)
        folds[name] = {fold: sorted(items) for fold, items in fold_groups.items()}
    return rows, folds


def compare(parent: dict[str, dict], candidate: dict[str, dict], draws: int,
            seed: int, parent_name: str, candidate_name: str) -> dict:
    ids = sorted(parent)
    if set(ids) != set(candidate):
        raise ValueError("paired image IDs do not match")
    delta = {image_id: candidate[image_id]["image_miou"] - parent[image_id]["image_miou"]
             for image_id in ids}
    by_fold: dict[int, list[str]] = defaultdict(list)
    by_group: dict[str, list[str]] = defaultdict(list)
    by_step: dict[int, list[str]] = defaultdict(list)
    for image_id in ids:
        for source in (parent[image_id], candidate[image_id]):
            if any(source[key] != parent[image_id][key] for key in ("fold", "group", "step")):
                raise ValueError(f"metadata differs for {image_id}")
        row = parent[image_id]
        by_fold[row["fold"]].append(image_id)
        by_group[row["group"]].append(image_id)
        by_step[row["step"]].append(image_id)

    group_names = sorted(by_group)
    group_delta_sum = np.asarray([
        np.sum([delta[image_id] for image_id in by_group[group]]) for group in group_names
    ], dtype=np.float64)
    group_sizes = np.asarray([len(by_group[group]) for group in group_names], dtype=np.int64)
    rng = np.random.default_rng(seed)
    group_draws = rng.integers(0, len(group_names), size=(draws, len(group_names)))
    # Resample specimen groups, keeping all images within each sampled group,
    # then average image deltas. This matches the established Model376 OOF CI.
    bootstrap = (group_delta_sum[group_draws].sum(axis=1)
                 / group_sizes[group_draws].sum(axis=1))
    ci = np.percentile(bootstrap * 100.0, [2.5, 97.5]).tolist()

    parent_mean = float(np.mean([parent[image_id]["image_miou"] for image_id in ids]))
    candidate_mean = float(np.mean([candidate[image_id]["image_miou"] for image_id in ids]))
    delta_pp = (candidate_mean - parent_mean) * 100.0
    fold_rows = {}
    fold_deltas = []
    band_gate_folds = 0
    interface_rows = {}
    eutectic_guard_folds = 0
    for fold, fold_ids in sorted(by_fold.items()):
        pmean = float(np.mean([parent[i]["image_miou"] for i in fold_ids]))
        cmean = float(np.mean([candidate[i]["image_miou"] for i in fold_ids]))
        fold_delta = (cmean - pmean) * 100.0
        fold_deltas.append(fold_delta)
        p_band_n = sum(parent[i]["boundary_pixels"] for i in fold_ids)
        c_band_n = sum(candidate[i]["boundary_pixels"] for i in fold_ids)
        p_band_err = sum(parent[i]["boundary_errors"] for i in fold_ids)
        c_band_err = sum(candidate[i]["boundary_errors"] for i in fold_ids)
        p_band_rate, c_band_rate = p_band_err / p_band_n, c_band_err / c_band_n
        relative_reduction = ((p_band_rate - c_band_rate) / p_band_rate
                              if p_band_rate > 0 else None)
        if relative_reduction is not None and relative_reduction >= 0.03:
            band_gate_folds += 1

        p_interface_n = sum(parent[i]["eutectic_interface_pixels"] for i in fold_ids)
        c_interface_n = sum(candidate[i]["eutectic_interface_pixels"] for i in fold_ids)
        p_interface_tp = sum(parent[i]["eutectic_interface_true_positive"] for i in fold_ids)
        c_interface_tp = sum(candidate[i]["eutectic_interface_true_positive"] for i in fold_ids)
        p_interface_recall = p_interface_tp / p_interface_n
        c_interface_recall = c_interface_tp / c_interface_n
        interface_delta_pp = (c_interface_recall - p_interface_recall) * 100.0
        if interface_delta_pp >= -0.50:
            eutectic_guard_folds += 1
        fold_rows[str(fold)] = {
            "images": len(fold_ids),
            "parent_image_mean_miou_pct": pmean * 100.0,
            "candidate_image_mean_miou_pct": cmean * 100.0,
            "delta_percentage_points": fold_delta,
            "boundary_band_pixels": p_band_n,
            "parent_boundary_error_rate": p_band_rate,
            "candidate_boundary_error_rate": c_band_rate,
            "boundary_error_relative_reduction": relative_reduction,
            "eutectic_interface_pixels": p_interface_n,
            "parent_eutectic_interface_recall": p_interface_recall,
            "candidate_eutectic_interface_recall": c_interface_recall,
            "eutectic_interface_recall_delta_percentage_points": interface_delta_pp,
        }
        interface_rows[str(fold)] = interface_delta_pp

    class_deltas = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        p = float(np.mean([parent[i]["class_iou"][class_id] for i in ids]))
        c = float(np.mean([candidate[i]["class_iou"][class_id] for i in ids]))
        class_deltas[class_name] = (c - p) * 100.0
    step4_ids = by_step.get(4, [])
    if not step4_ids:
        raise ValueError("no Step 4 images")
    step4_delta = float(np.mean([delta[i] for i in step4_ids]) * 100.0)

    mIoU_gate = {
        "mean_delta_at_least_0_50pp": delta_pp >= 0.50,
        "at_least_3_of_4_folds_positive": sum(x > 0 for x in fold_deltas) >= 3,
        "worst_fold_no_worse_than_minus_0_30pp": min(fold_deltas) >= -0.30,
        "group_bootstrap_95ci_lower_above_zero": ci[0] > 0,
        "no_class_mean_drop_over_0_50pp": min(class_deltas.values()) >= -0.50,
        "step4_no_drop_over_0_30pp": step4_delta >= -0.30,
    }
    mechanism_gate = {
        "boundary_error_relative_reduction_at_least_3pct_in_3_folds": band_gate_folds >= 3,
        "eutectic_interface_recall_drop_within_0_50pp_in_every_fold": eutectic_guard_folds == 4,
    }
    return {
        "metric": "official image-wise mean of four class IoUs; empty union=0",
        "scope": "train-only exactly-once specimen-grouped OOF; no valid/test input",
        "parent": parent_name,
        "candidate": candidate_name,
        "images": len(ids),
        "specimen_groups": len(group_names),
        "parent_image_mean_miou_pct": parent_mean * 100.0,
        "candidate_image_mean_miou_pct": candidate_mean * 100.0,
        "delta_percentage_points": delta_pp,
        "group_bootstrap_draws": draws,
        "group_bootstrap_95ci_delta_percentage_points": ci,
        "folds": fold_rows,
        "class_mean_iou_delta_percentage_points": class_deltas,
        "step4_images": len(step4_ids),
        "step4_delta_percentage_points": step4_delta,
        "mIoU_gate": mIoU_gate,
        "mechanism_gate": mechanism_gate,
        "adoption_gate_passed": all(mIoU_gate.values()) and all(mechanism_gate.values()),
        "interpretation_guard": (
            "OOF comparison is selection-conditional because this dataset has been used for prior "
            "experiments; it is not test performance."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--parent-name", required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=378)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, _ = _load_rows(args.cv_root, {
        "parent": args.parent_root,
        "candidate": args.candidate_root,
    })
    result = compare(rows["parent"], rows["candidate"], args.draws, args.seed,
                     args.parent_name, "Model378 SVLS")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
