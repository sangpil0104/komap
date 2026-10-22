#!/usr/bin/env python3
"""Descriptive train-OOF decomposition of predicted-component FP pixels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import model404_predicted_component_fp_audit as component_audit
import model407_component_fp_lock as model407_lock


ROOT = Path(__file__).resolve().parents[1]
TARGET_CLASSES = component_audit.TARGET_CLASSES
AREA_BINS = component_audit.AREA_BINS
ARMS = ("candidate", "control", "model392_parent")
BIN_METRICS = (
    "pred_components",
    "zero_overlap_components",
    "overlap_components",
    "zero_overlap_fp_pixels",
    "overlap_fp_pixels",
    "all_component_fp_pixels",
)
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 40721


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decompose_image(probability: np.ndarray, truth: np.ndarray) -> dict:
    """Split each class's FP pixels by component area and GT overlap status."""
    raw = component_audit.image_metrics(probability, truth)
    result: dict[str, dict] = {}
    for class_name, source in raw.items():
        class_id = next(
            class_id for class_id, name in TARGET_CLASSES.items()
            if name == class_name
        )
        by_bin: dict[str, dict[str, float]] = {}
        all_bin_fp = 0.0
        for bin_name in AREA_BINS:
            pred_count = float(source[f"pred_components_{bin_name}"])
            zero_count = float(source[f"unmatched_components_{bin_name}"])
            all_fp = float(source[f"component_fp_pixels_{bin_name}"])
            zero_fp = float(source[f"unmatched_component_area_{bin_name}"])
            overlap_count = pred_count - zero_count
            overlap_fp = all_fp - zero_fp
            if (overlap_count < -1e-6 or overlap_fp < -1e-6
                    or zero_fp < -1e-6):
                raise ValueError("component overlap decomposition became negative")
            by_bin[bin_name] = {
                "pred_components": pred_count,
                "zero_overlap_components": zero_count,
                "overlap_components": overlap_count,
                "zero_overlap_fp_pixels": zero_fp,
                "overlap_fp_pixels": overlap_fp,
                "all_component_fp_pixels": all_fp,
            }
            all_bin_fp += all_fp
        total_fp = float(source["false_positive_pixels"])
        if not np.isclose(all_bin_fp, total_fp, rtol=0.0, atol=0.5):
            raise ValueError(
                f"component bins account for {all_bin_fp} FP pixels, expected {total_fp}"
            )
        result[class_name] = {
            "gt_present": bool(np.any(truth == class_id)),
            "total_target_fp_pixels": total_fp,
            "area_bins": by_bin,
        }
    return result


def _empty_records() -> dict:
    return {
        arm: {
            class_name: {
                bin_name: {metric: [] for metric in BIN_METRICS}
                for bin_name in AREA_BINS
            }
            for class_name in TARGET_CLASSES.values()
        }
        for arm in ARMS
    }


def _paired_summary(candidate: np.ndarray, reference: np.ndarray,
                    samples: list[np.ndarray], mask: np.ndarray,
                    rows: list[dict]) -> dict:
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        return {
            "support_images": 0,
            "support_groups": 0,
            "candidate_minus_reference_per_image_mean": None,
            "paired_group_bootstrap_95ci": None,
        }
    bootstrap_values = []
    for sample in samples:
        selected_sample = sample[mask[sample]]
        if selected_sample.size:
            bootstrap_values.append(float(delta[selected_sample].mean()))
    if len(bootstrap_values) < 100:
        interval = None
    else:
        low, high = np.quantile(
            np.asarray(bootstrap_values, dtype=np.float64), [0.025, 0.975]
        )
        interval = [float(low), float(high)]
    return {
        "support_images": int(selected.size),
        "support_groups": len({rows[int(index)]["group"] for index in selected}),
        "candidate_minus_reference_per_image_mean": float(delta[selected].mean()),
        "paired_group_bootstrap_95ci": interval,
    }


def audit(root: Path, data_root: Path, candidate_root: Path,
          control_root: Path, baseline_root: Path,
          draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> dict:
    root = root.resolve()
    data_root = data_root.resolve()
    lock_record = model407_lock.verify_oof_lock(root)

    image_paths = sorted((data_root / "train/images").glob("*_image.png"))
    image_ids = [path.name.removesuffix("_image.png") for path in image_paths]
    if len(image_ids) != 70:
        raise ValueError(f"expected 70 train images, found {len(image_ids)}")
    metadata = {
        image_id: component_audit.group_and_step(image_id)
        for image_id in image_ids
    }
    groups = {group for group, _ in metadata.values()}
    if len(groups) != 59:
        raise ValueError(f"expected 59 specimen groups, found {len(groups)}")

    fold_manifest = json.loads(
        component_audit.FOLD_FILE.read_text(encoding="utf-8")
    )
    fold_for_group: dict[str, int] = {}
    for fold_data in fold_manifest["folds"]:
        fold = int(fold_data["fold"])
        for group in fold_data["heldout_groups"]:
            if group in fold_for_group:
                raise ValueError(f"specimen group assigned to multiple folds: {group}")
            fold_for_group[group] = fold
    if set(fold_for_group) != groups:
        raise ValueError("fixed fold manifest does not cover all train groups")

    roots = {
        "candidate": candidate_root.resolve(),
        "control": control_root.resolve(),
        "model392_parent": baseline_root.resolve() / "seed1",
    }
    records = _empty_records()
    rows: list[dict] = []
    gt_presence = {class_name: [] for class_name in TARGET_CLASSES.values()}
    for fold in range(1, 5):
        expected = {
            image_id for image_id in image_ids
            if fold_for_group[metadata[image_id][0]] == fold
        }
        archives = {
            arm: component_audit.shared.load_archive(
                arm_root / f"fold{fold}_oof/valid_probabilities.npz", expected
            )
            for arm, arm_root in roots.items()
        }
        for image_id in sorted(expected):
            truth = component_audit.shared.read_mask(
                data_root / "train/masks" / f"{image_id}_mask.png"
            )
            group, step = metadata[image_id]
            presence = {
                class_name: bool(np.any(truth == class_id))
                for class_id, class_name in TARGET_CLASSES.items()
            }
            for class_name, present in presence.items():
                gt_presence[class_name].append(present)
            row = {
                "image_id": image_id,
                "group": group,
                "step": step,
                "fold": fold,
                "gt_presence": presence,
            }
            for arm in ARMS:
                metrics = decompose_image(archives[arm][image_id], truth)
                row[arm] = metrics
                for class_name, class_values in metrics.items():
                    for bin_name, bin_values in class_values["area_bins"].items():
                        for metric, value in bin_values.items():
                            records[arm][class_name][bin_name][metric].append(value)
            rows.append(row)

    if (len(rows) != 70 or len({row["image_id"] for row in rows}) != 70
            or len({row["group"] for row in rows}) != 59):
        raise ValueError("audit must cover exactly 70 train images and 59 groups")
    samples = component_audit.bootstrap_indices(
        rows, draws=draws, seed=seed
    )

    strata = ("all", "gt_present", "gt_absent")
    presence_masks = {
        class_name: {
            "all": np.ones(len(rows), dtype=bool),
            "gt_present": np.asarray(gt_presence[class_name], dtype=bool),
            "gt_absent": ~np.asarray(gt_presence[class_name], dtype=bool),
        }
        for class_name in TARGET_CLASSES.values()
    }
    per_arm_mean: dict[str, dict] = {}
    for arm in ARMS:
        per_arm_mean[arm] = {}
        for class_name in TARGET_CLASSES.values():
            per_arm_mean[arm][class_name] = {}
            for stratum in strata:
                mask = presence_masks[class_name][stratum]
                per_arm_mean[arm][class_name][stratum] = {
                    "support_images": int(mask.sum()),
                    "support_groups": len({
                        rows[int(index)]["group"]
                        for index in np.flatnonzero(mask)
                    }),
                    "area_bins": {
                        bin_name: {
                            metric: (
                                float(np.mean(np.asarray(
                                    records[arm][class_name][bin_name][metric],
                                    dtype=np.float64,
                                )[mask]))
                                if mask.any() else None
                            )
                            for metric in BIN_METRICS
                        }
                        for bin_name in AREA_BINS
                    },
                }

    paired: dict[str, dict] = {}
    for reference in ("control", "model392_parent"):
        paired[f"candidate_vs_{reference}"] = {}
        for class_name in TARGET_CLASSES.values():
            paired[f"candidate_vs_{reference}"][class_name] = {}
            for stratum in strata:
                mask = presence_masks[class_name][stratum]
                paired[f"candidate_vs_{reference}"][class_name][stratum] = {
                    bin_name: {
                        metric: _paired_summary(
                            np.asarray(records["candidate"][class_name][bin_name][metric]),
                            np.asarray(records[reference][class_name][bin_name][metric]),
                            samples,
                            mask,
                            rows,
                        )
                        for metric in BIN_METRICS
                    }
                    for bin_name in AREA_BINS
                }

    return {
        "experiment": "Model407 predicted-component FP pixel-area decomposition",
        "scope": "fixed grouped train-OOF only: 70 images / 59 specimen groups",
        "metric": "native-resolution argmax; 8-connected target-class components",
        "area_bins_native_pixels": AREA_BINS,
        "fp_decomposition": {
            "zero_overlap_fp_pixels": "FP pixels in components with zero same-class GT overlap",
            "overlap_fp_pixels": "FP pixels in components with nonzero same-class GT overlap",
            "all_component_fp_pixels": "sum of the preceding two categories",
        },
        "gt_presence_strata": {
            "gt_present": "image has at least one GT pixel of the target class",
            "gt_absent": "image has zero GT pixels of the target class",
            "metric_note": (
                "With official empty-union=0, target-class IoU is 0 in either absent-class case; "
                "FP changes can still affect other classes through argmax competition."
            ),
            "support_images_per_class": {
                class_name: {
                    stratum: int(mask.sum())
                    for stratum, mask in presence_masks[class_name].items()
                }
                for class_name in TARGET_CLASSES.values()
            },
        },
        "per_arm_image_mean": per_arm_mean,
        "paired_group_bootstrap": paired,
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "model407_source_lock_sha256": sha256(root / model407_lock.SOURCE_LOCK),
        "model407_oof_lock_sha256": sha256(root / model407_lock.OOF_LOCK),
        "model407_oof_lock_record": lock_record,
        "audit_script_sha256": sha256(Path(__file__).resolve()),
        "official_valid_read": False,
        "test_read": False,
        "adoption_gate": False,
        "threshold_or_route_search": False,
        "interpretation": "Descriptive only; does not alter Model407's preregistered score or GO/NO-GO gate.",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def self_test() -> None:
    truth = np.zeros((24, 24), dtype=np.uint8)
    truth[4:6, 4:6] = 1
    prediction = np.zeros_like(truth)
    prediction[4:6, 4:6] = 1
    prediction[4, 6] = 1  # one FP pixel in a GT-overlapping component
    prediction[10, 10] = 1  # one FP pixel in a zero-overlap component
    probability = np.full((4, *truth.shape), 0.001, dtype=np.float32)
    probability[0] = 0.997
    probability[1] = 0.001
    probability[2] = 0.001
    probability[3] = 0.001
    for class_id in range(4):
        selected = prediction == class_id
        probability[:, selected] = 0.001
        probability[class_id, selected] = 0.997
    result = decompose_image(probability, truth)["Al3Ni"]
    assert result["gt_present"] is True
    tiny = result["area_bins"]["tiny_1_to_lt64"]
    assert result["total_target_fp_pixels"] == 2.0
    assert tiny["pred_components"] == 2.0
    assert tiny["zero_overlap_components"] == 1.0
    assert tiny["overlap_components"] == 1.0
    assert tiny["zero_overlap_fp_pixels"] == 1.0
    assert tiny["overlap_fp_pixels"] == 1.0
    assert tiny["all_component_fp_pixels"] == 2.0
    eutectic = decompose_image(probability, truth)["Eutectic Si"]
    assert eutectic["gt_present"] is False
    print("Model407 predicted-component FP area audit self-test passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--control-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    required = {
        "--data-root": args.data_root,
        "--candidate-root": args.candidate_root,
        "--control-root": args.control_root,
        "--baseline-root": args.baseline_root,
        "--out": args.out,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("required unless --self-test: " + ", ".join(missing))
    if args.draws < 100:
        parser.error("--draws must be at least 100")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {args.out}")
    result = audit(
        args.root, args.data_root, args.candidate_root, args.control_root,
        args.baseline_root, draws=args.draws, seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({
        "out": str(args.out),
        "scope": result["scope"],
        "official_valid_read": False,
        "test_read": False,
        "adoption_gate": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
