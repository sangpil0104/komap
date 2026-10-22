#!/usr/bin/env python3
"""Score the fixed Model397 probability blend on the 70 train-only OOF images."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from write_oof_manifest import verify_manifest


ROOT = Path(__file__).resolve().parents[1]
FOLD_FILE = ROOT / "config/model397_group_folds.json"
CLASS_NAMES = ["Al", "Al3Ni", "Primary Si", "Eutectic Si"]
PALETTE = np.asarray(
    [(153, 127, 76), (25, 76, 153), (76, 178, 76), (204, 204, 204)],
    dtype=np.uint8,
)
GROUP_RE = re.compile(r"^(A[0-9]+)_")
STEP_RE = re.compile(r"_step([0-9]+)_")


def group_and_step(image_id: str) -> tuple[str, int | None]:
    group_match = GROUP_RE.match(image_id + "_")
    if group_match is None:
        raise ValueError(f"cannot parse specimen group from {image_id!r}")
    step_match = STEP_RE.search(image_id + "_")
    return group_match.group(1), int(step_match.group(1)) if step_match else None


def read_mask(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    ids = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        ids[np.all(rgb == color, axis=2)] = class_id
    if np.any(ids == 255):
        raise ValueError(f"unknown RGB class color in {path}")
    return ids


def class_ious(probabilities: np.ndarray, truth: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.shape != (4, *truth.shape) or not np.isfinite(probabilities).all():
        raise ValueError(f"invalid probability shape/values {probabilities.shape}, GT={truth.shape}")
    if probabilities.min() < -1e-4 or probabilities.max() > 1.0001:
        raise ValueError("probabilities outside [0,1]")
    if np.max(np.abs(probabilities.sum(axis=0) - 1.0)) > 0.03:
        raise ValueError("probability channels are not normalized")
    prediction = probabilities.argmax(axis=0)
    confusion = np.bincount(
        truth.astype(np.int64).ravel() * 4 + prediction.astype(np.int64).ravel(),
        minlength=16,
    ).reshape(4, 4)
    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(axis=0) + confusion.sum(axis=1) - intersection
    return np.divide(intersection, union, out=np.zeros(4), where=union > 0)


def summarize(class_scores: np.ndarray) -> dict:
    return {
        "image_mean_miou_percent": float(class_scores.mean(axis=1).mean() * 100.0),
        "class_image_mean_iou_percent": {
            name: float(class_scores[:, index].mean() * 100.0)
            for index, name in enumerate(CLASS_NAMES)
        },
    }


def bootstrap_indices(rows: list[dict], draws: int = 20_000, seed: int = 397) -> list[np.ndarray]:
    by_fold_group: dict[int, dict[str, list[int]]] = {fold: {} for fold in range(1, 5)}
    for index, row in enumerate(rows):
        by_fold_group[row["fold"]].setdefault(row["group"], []).append(index)
    if any(not groups for groups in by_fold_group.values()):
        raise ValueError("cannot bootstrap: at least one fold has no groups")
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        selected: list[int] = []
        for fold in range(1, 5):
            groups = sorted(by_fold_group[fold])
            group_indices = rng.integers(0, len(groups), size=len(groups))
            for group_index in group_indices:
                selected.extend(by_fold_group[fold][groups[int(group_index)]])
        samples.append(np.asarray(selected, dtype=np.int32))
    return samples


def compare(candidate: np.ndarray, reference: np.ndarray, rows: list[dict],
            samples: list[np.ndarray]) -> dict:
    delta_class = (candidate - reference) * 100.0
    delta_image = delta_class.mean(axis=1)
    sampled = np.asarray([delta_image[index].mean() for index in samples])
    fold_delta = {
        str(fold): float(delta_image[np.asarray([row["fold"] == fold for row in rows])].mean())
        for fold in range(1, 5)
    }
    group_delta: dict[str, float] = defaultdict(float)
    for row, value in zip(rows, delta_image):
        group_delta[row["group"]] += float(value)
    step4 = np.asarray([row["step"] == 4 for row in rows])
    net = float(delta_image.sum())
    largest_group_share = max(0.0, max(group_delta.values())) / net if net > 0 else None
    return {
        "delta_image_mean_pp": float(delta_image.mean()),
        "paired_group_bootstrap_95ci_pp": [float(x) for x in np.quantile(sampled, [0.025, 0.975])],
        "positive_images": int(np.count_nonzero(delta_image > 0)),
        "positive_folds": int(sum(value > 0 for value in fold_delta.values())),
        "fold_delta_pp": fold_delta,
        "class_delta_pp": {
            name: float(delta_class[:, index].mean()) for index, name in enumerate(CLASS_NAMES)
        },
        "step4_delta_pp": float(delta_image[step4].mean()) if step4.any() else None,
        "largest_group_positive_contribution_share": largest_group_share,
    }


def score(data_root: Path, output_root: Path) -> dict:
    data_root, output_root = data_root.resolve(), output_root.resolve()
    image_dir = data_root / "train/images"
    mask_dir = data_root / "train/masks"
    image_paths = {
        path.name.removesuffix("_image.png"): path
        for path in sorted(image_dir.glob("*_image.png"))
    }
    if len(image_paths) != 70:
        raise ValueError(f"expected exactly 70 train images, found {len(image_paths)}")
    mask_paths = {image_id: mask_dir / f"{image_id}_mask.png" for image_id in image_paths}
    if any(not path.is_file() for path in mask_paths.values()):
        raise FileNotFoundError("one or more train masks are missing")

    groups = {image_id: group_and_step(image_id) for image_id in image_paths}
    if len({value[0] for value in groups.values()}) != 59:
        raise ValueError("expected 59 specimen groups in the 70-image train split")
    fold_config = json.loads(FOLD_FILE.read_text(encoding="utf-8"))
    fold_for_group: dict[str, int] = {}
    for entry in fold_config["folds"]:
        fold = int(entry["fold"])
        for group in entry["heldout_groups"]:
            if group in fold_for_group:
                raise ValueError(f"specimen group appears in multiple folds: {group}")
            fold_for_group[group] = fold
    if set(fold_for_group) != {value[0] for value in groups.values()}:
        raise ValueError("fold file and train specimen groups differ")

    rows: list[dict] = []
    model329_root = output_root / "model329"
    model392_root = output_root / "model392"
    for fold in range(1, 5):
        expected = {
            image_id for image_id, (group, _) in groups.items()
            if fold_for_group[group] == fold
        }
        archives = {
            "B4": model329_root / f"fold{fold}_oof/probabilities.npz",
            "seed1": model392_root / "seed1" / f"fold{fold}_oof/valid_probabilities.npz",
            "seed2": model392_root / "seed2" / f"fold{fold}_oof/valid_probabilities.npz",
        }
        verify_manifest(output_root, "model329", fold, None)
        verify_manifest(output_root, "model392", fold, 1)
        verify_manifest(output_root, "model392", fold, 2)
        loaded = {name: np.load(path, allow_pickle=False) for name, path in archives.items()}
        try:
            for name, archive in loaded.items():
                if set(archive.files) != expected:
                    raise ValueError(
                        f"{name} fold{fold} OOF ID mismatch: expected {len(expected)}, got {len(archive.files)}"
                    )
            for image_id in sorted(expected):
                truth = read_mask(mask_paths[image_id])
                p_b4 = np.asarray(loaded["B4"][image_id], dtype=np.float32)
                p_seed1 = np.asarray(loaded["seed1"][image_id], dtype=np.float32)
                p_seed2 = np.asarray(loaded["seed2"][image_id], dtype=np.float32)
                p_hrnet = 0.5 * p_seed1 + 0.5 * p_seed2
                p_blend = 0.50 * p_b4 + 0.25 * p_seed1 + 0.25 * p_seed2
                group, step = groups[image_id]
                rows.append({
                    "fold": fold,
                    "image_id": image_id,
                    "group": group,
                    "step": step,
                    "B4": class_ious(p_b4, truth),
                    "H": class_ious(p_hrnet, truth),
                    "C": class_ious(p_blend, truth),
                })
        finally:
            for archive in loaded.values():
                archive.close()

    if len(rows) != 70 or len({row["image_id"] for row in rows}) != 70:
        raise ValueError("OOF scoring did not cover every train image exactly once")
    samples = bootstrap_indices(rows)
    arm_scores = {
        arm: np.asarray([row[arm] for row in rows], dtype=np.float64)
        for arm in ("B4", "H", "C")
    }
    comparisons = {
        "C_vs_H": compare(arm_scores["C"], arm_scores["H"], rows, samples),
        "C_vs_B4": compare(arm_scores["C"], arm_scores["B4"], rows, samples),
    }
    h = comparisons["C_vs_H"]
    b4 = comparisons["C_vs_B4"]
    gate_checks = {
        "C_vs_H_gain_ge_0_50pp": h["delta_image_mean_pp"] >= 0.50,
        "C_vs_H_group_ci_lower_gt_0": h["paired_group_bootstrap_95ci_pp"][0] > 0.0,
        "C_vs_H_positive_at_least_3_folds": h["positive_folds"] >= 3,
        "C_vs_H_worst_fold_ge_minus_0_30pp": min(h["fold_delta_pp"].values()) >= -0.30,
        "C_vs_B4_gain_ge_0_30pp": b4["delta_image_mean_pp"] >= 0.30,
        "C_vs_B4_group_ci_lower_gt_0": b4["paired_group_bootstrap_95ci_pp"][0] > 0.0,
        "C_vs_B4_positive_at_least_3_folds": b4["positive_folds"] >= 3,
        "classes_and_step4_safe_vs_H": (
            min(h["class_delta_pp"].values()) >= -0.50
            and h["step4_delta_pp"] is not None
            and h["step4_delta_pp"] >= -0.50
        ),
        "largest_group_gain_share_le_25_percent": (
            h["largest_group_positive_contribution_share"] is not None
            and h["largest_group_positive_contribution_share"] <= 0.25
        ),
    }
    return {
        "experiment": "Model397 fixed B4/HRNet grouped train-OOF reproduction",
        "metric": "image-wise 4-class mIoU; empty-union=0",
        "class_order": CLASS_NAMES,
        "images": len(rows),
        "specimen_groups": len({row["group"] for row in rows}),
        "folds": 4,
        "bootstrap_draws": 20_000,
        "bootstrap_seed": 397,
        "fixed_blend": "0.50 * Model329_B4 + 0.25 * Model392_seed1 + 0.25 * Model392_seed2",
        "arms": {arm: summarize(scores) for arm, scores in arm_scores.items()},
        "comparisons": comparisons,
        "oof_screen": {"checks": gate_checks, "pass": all(gate_checks.values())},
        "official_valid_read": False,
        "official_valid_evaluation_authorized": False,
        "test_read": False,
        "interpretation": (
            "Train-only grouped OOF development result; selection-conditional, not official-valid or test evidence."
        ),
        "per_image": [
            {key: (value.tolist() if isinstance(value, np.ndarray) else value)
             for key, value in row.items()}
            for row in rows
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None,
                        help="Result JSON path; defaults to OUTPUT_ROOT/results/model397_oof.json")
    args = parser.parse_args()
    result = score(args.data_root, args.output_root)
    out_path = args.out or args.output_root / "results/model397_oof.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "result": str(out_path),
        "model329_b4_oof_percent": result["arms"]["B4"]["image_mean_miou_percent"],
        "model392_seedmean_oof_percent": result["arms"]["H"]["image_mean_miou_percent"],
        "model397_blend_oof_percent": result["arms"]["C"]["image_mean_miou_percent"],
        "oof_screen_pass": result["oof_screen"]["pass"],
        "official_valid_read": False,
        "test_read": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
