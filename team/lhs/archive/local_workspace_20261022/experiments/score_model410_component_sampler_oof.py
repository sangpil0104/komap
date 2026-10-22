#!/usr/bin/env python3
"""One-shot official-metric scoring for Model410's paired train OOF."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import model396_fixed_architecture_oof as m396
import model405_small_component_fixed_blend_oof as m405
import model410_component_sampler_lock as lock
import score_model406_component_hardnegative_oof as scorer406


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments/results/model410_component_sampler_oof.json"
SPLITS = ROOT / "outputs/model403_local_supcon/splits/model392_cv"
WIDTH_AREA_BINS = ((1, 4), (4, 16), (16, 64))
BIN_NAMES = ("1-<4", "4-<16", "16-<64")
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 410


def _width_normalized_component_stats(
    truth: np.ndarray,
    prediction: np.ndarray,
    class_id: int,
    area_factor: float,
) -> list[dict[str, int]]:
    """Count/hit GT and unmatched predicted components in modal-width bins."""
    if truth.ndim != 2 or prediction.shape != truth.shape:
        raise ValueError("truth and prediction must be matching HxW class maps")
    if not np.isfinite(area_factor) or area_factor <= 0:
        raise ValueError("area_factor must be finite and positive")
    gt_count, gt_labels, gt_stats, _ = cv2.connectedComponentsWithStats(
        np.ascontiguousarray(truth == class_id, dtype=np.uint8),
        connectivity=8,
        ltype=cv2.CV_32S,
    )
    gt_hit_pixels = np.bincount(
        gt_labels[prediction == class_id].ravel(), minlength=gt_count
    )
    pred_count, pred_labels, pred_stats, _ = cv2.connectedComponentsWithStats(
        np.ascontiguousarray(prediction == class_id, dtype=np.uint8),
        connectivity=8,
        ltype=cv2.CV_32S,
    )
    pred_gt_overlap = np.bincount(
        pred_labels[truth == class_id].ravel(), minlength=pred_count
    )
    result = [
        {"gt_components": 0, "gt_hits": 0, "unmatched_pred_components": 0}
        for _ in WIDTH_AREA_BINS
    ]
    for component_id in range(1, gt_count):
        area = int(gt_stats[component_id, cv2.CC_STAT_AREA]) * area_factor
        bin_id = next(
            (i for i, (low, high) in enumerate(WIDTH_AREA_BINS) if low <= area < high),
            None,
        )
        if bin_id is None:
            continue
        raw_area = int(gt_stats[component_id, cv2.CC_STAT_AREA])
        result[bin_id]["gt_components"] += 1
        result[bin_id]["gt_hits"] += int(gt_hit_pixels[component_id] / raw_area >= 0.50)
    for component_id in range(1, pred_count):
        area = int(pred_stats[component_id, cv2.CC_STAT_AREA]) * area_factor
        bin_id = next(
            (i for i, (low, high) in enumerate(WIDTH_AREA_BINS) if low <= area < high),
            None,
        )
        if bin_id is not None and pred_gt_overlap[component_id] == 0:
            result[bin_id]["unmatched_pred_components"] += 1
    return result


def width_normalized_component_diagnostics(
    candidate_root: Path, control_root: Path, source_lock: dict
) -> dict:
    """Descriptive train-OOF diagnostics in the same size coordinates as sampling."""
    masks_by_fold = m396.mask_files(ROOT)
    rows: list[dict] = []
    values = {
        arm: {
            class_name: [[{"gt_components": 0, "gt_hits": 0,
                           "unmatched_pred_components": 0} for _ in range(3)]
                         for _ in range(70)]
            for class_name in m405.TARGETS.values()
        }
        for arm in ("candidate", "control")
    }
    image_index = 0
    for fold, mask_map in masks_by_fold.items():
        target_width = int(
            source_lock["model392_parent_model_and_preprocessing_metadata"]
            [str(fold)]["preprocessing_config"]["target_width"]
        )
        image_dir = SPLITS / f"seed1_fold{fold}/valid/images"
        candidate_path = candidate_root / f"fold{fold}_oof/valid_probabilities.npz"
        control_path = control_root / f"fold{fold}_oof/valid_probabilities.npz"
        with np.load(candidate_path, allow_pickle=False) as candidate, \
             np.load(control_path, allow_pickle=False) as control:
            expected = set(mask_map)
            if set(candidate.files) != expected or set(control.files) != expected:
                raise ValueError(f"Model410 OOF IDs differ from fixed fold {fold}")
            for image_id in sorted(expected):
                image_path = image_dir / f"{image_id}_image.png"
                with Image.open(image_path) as image:
                    image_width = image.width
                area_factor = (target_width / image_width) ** 2
                truth = m396.mask_to_ids(mask_map[image_id])
                group_match = m405.GROUP_RE.match(image_id)
                step_match = m405.STEP_RE.search(image_id)
                if group_match is None or step_match is None:
                    raise ValueError(f"cannot parse held-out train image ID: {image_id}")
                rows.append({
                    "image": image_id,
                    "group": group_match.group(1),
                    "step": int(step_match.group(1)),
                    "fold": fold,
                })
                for arm, archive in (("candidate", candidate), ("control", control)):
                    prediction = np.asarray(archive[image_id], dtype=np.float32).argmax(axis=0)
                    if prediction.shape != truth.shape:
                        raise ValueError(f"prediction shape mismatch for {arm}/{image_id}")
                    for class_id, class_name in m405.TARGETS.items():
                        values[arm][class_name][image_index] = _width_normalized_component_stats(
                            truth, prediction, class_id, area_factor
                        )
                image_index += 1

    if image_index != 70 or len({row["group"] for row in rows}) != 59:
        raise ValueError("width-normalized component audit did not cover 70 images/59 groups")
    bootstrap_indices = m396.group_bootstrap_indices(
        rows, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED
    )
    result = {
        "definition": (
            "8-connected components binned by raw_area*(fold_modal_width/image_width)^2; "
            "GT hit means at least 50% of component pixels are predicted as the true class; "
            "unmatched prediction means zero same-class GT overlap"
        ),
        "scope": "70 held-out train images/59 specimen groups; descriptive only, not an added gate",
        "area_bins_modal_width_equivalent_pixels": [list(item) for item in WIDTH_AREA_BINS],
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "classes": {},
        "valid_read": False,
        "test_read": False,
    }
    for class_name in m405.TARGETS.values():
        result["classes"][class_name] = {}
        for bin_id, bin_name in enumerate(BIN_NAMES):
            by_arm = {}
            for arm in ("control", "candidate"):
                records = [item[bin_id] for item in values[arm][class_name]]
                total_gt = sum(item["gt_components"] for item in records)
                total_hits = sum(item["gt_hits"] for item in records)
                by_arm[arm] = {
                    "gt_components": total_gt,
                    "gt_hits": total_hits,
                    "gt_hit_rate_percent": 100.0 * total_hits / total_gt if total_gt else None,
                    "unmatched_pred_components_per_image": float(np.mean([
                        item["unmatched_pred_components"] for item in records
                    ])),
                }
            control_rows = [item[bin_id] for item in values["control"][class_name]]
            candidate_rows = [item[bin_id] for item in values["candidate"][class_name]]
            hit_delta_draws = []
            unmatched_delta = np.asarray([
                cand["unmatched_pred_components"] - ctrl["unmatched_pred_components"]
                for ctrl, cand in zip(control_rows, candidate_rows)
            ], dtype=np.float64)
            unmatched_draws = []
            for indices in bootstrap_indices:
                control_gt = sum(control_rows[i]["gt_components"] for i in indices)
                candidate_gt = sum(candidate_rows[i]["gt_components"] for i in indices)
                control_hit = sum(control_rows[i]["gt_hits"] for i in indices)
                candidate_hit = sum(candidate_rows[i]["gt_hits"] for i in indices)
                control_rate = control_hit / control_gt if control_gt else 0.0
                candidate_rate = candidate_hit / candidate_gt if candidate_gt else 0.0
                hit_delta_draws.append((candidate_rate - control_rate) * 100.0)
                unmatched_draws.append(float(unmatched_delta[indices].mean()))
            result["classes"][class_name][bin_name] = {
                "control": by_arm["control"],
                "candidate": by_arm["candidate"],
                "candidate_vs_control": {
                    "gt_hit_rate_delta_pp": (
                        by_arm["candidate"]["gt_hit_rate_percent"]
                        - by_arm["control"]["gt_hit_rate_percent"]
                        if by_arm["candidate"]["gt_hit_rate_percent"] is not None
                        and by_arm["control"]["gt_hit_rate_percent"] is not None else None
                    ),
                    "gt_hit_rate_group_bootstrap_95ci_pp": [
                        float(x) for x in np.quantile(hit_delta_draws, [0.025, 0.975])
                    ],
                    "unmatched_pred_components_per_image_delta": float(unmatched_delta.mean()),
                    "unmatched_pred_components_group_bootstrap_95ci": [
                        float(x) for x in np.quantile(unmatched_draws, [0.025, 0.975])
                    ],
                },
            }
    return result


def _self_test() -> None:
    truth = np.zeros((32, 32), dtype=np.uint8)
    prediction = np.zeros_like(truth)
    truth[8, 8:10] = 1
    prediction[8, 8:10] = 1
    prediction[20, 20:22] = 3
    class_one = _width_normalized_component_stats(
        truth, prediction, 1, (1278 / 675) ** 2
    )
    class_three = _width_normalized_component_stats(
        truth, prediction, 3, (1278 / 1440) ** 2
    )
    assert class_one[1]["gt_components"] == 1 and class_one[1]["gt_hits"] == 1
    assert class_three[0]["unmatched_pred_components"] == 1
    assert sum(item["gt_components"] for item in class_three) == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.root.resolve() != ROOT.resolve():
        raise ValueError("Model410 scorer is tied to the locked repository root")
    if args.self_test:
        _self_test()
        print("Model410 width-normalized OOF component diagnostic self-test passed")
        return
    claim_path = Path(str(RESULT) + ".claim")
    if RESULT.exists() or claim_path.exists():
        raise FileExistsError(f"refusing to repeat one-shot score: {RESULT}")
    oof_lock = lock.verify_oof_lock()
    source_lock = lock.verify_source_lock()
    claim = {
        "experiment": "Model410 paired small-component-balanced crop sampler",
        "claimed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "oof_lock_sha256": lock.sha256(lock.OOF_LOCK),
        "policy": "one-shot; claim retained after interruption",
    }
    descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(claim, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())

    result = scorer406.score(
        ROOT / "data",
        ROOT / "outputs/model410_component_sampler/candidate",
        ROOT / "outputs/model410_component_sampler/control",
        ROOT / "outputs/model392_widthnorm",
    )
    result["width_normalized_component_diagnostics"] = width_normalized_component_diagnostics(
        ROOT / "outputs/model410_component_sampler/candidate",
        ROOT / "outputs/model410_component_sampler/control",
        source_lock,
    )
    checks = result["preregistered_gate"]["checks"]
    paired = result["paired_candidate_vs_control"]
    checks["all_class_iou_deltas_ge_minus_0_30pp"] = (
        min(paired["class_delta_pp"].values()) >= -0.30
    )
    checks["step4_delta_ge_minus_0_30pp"] = (
        paired["step4_delta_pp"] is not None and paired["step4_delta_pp"] >= -0.30
    )
    result["preregistered_gate"]["pass"] = all(checks.values())
    result["experiment"] = "Model410 small-component-balanced crop sampling"
    result["objective"] = (
        "50% positive crop anchors sample target class, modal-width-equivalent size bin "
        "[1,4)/[4,16)/[16,64), "
        "using raw_area*(fold_modal_width/image_width)^2, "
        "then 8-connected component and component pixel uniformly; unchanged 50% uniform crops"
    )
    result["status"] = "OOF_SCREEN_PASS" if result["preregistered_gate"]["pass"] else "NO_GO"
    result["valid_read"] = False
    result["test_read"] = False
    result["official_valid_stage_allowed"] = False
    result["evidence_note"] = (
        "Repeated fixed-train OOF development evidence only; no independent test claim. "
        "A pass authorizes only a separately locked seed-2 train-OOF confirmation."
    )
    result["source_lock_sha256"] = lock.sha256(lock.SOURCE_LOCK)
    result["oof_lock_sha256"] = lock.sha256(lock.OOF_LOCK)
    result["claim_sha256"] = lock.sha256(claim_path)
    result["model410_oof_lock_scope"] = oof_lock["training_images"]
    with RESULT.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=True)
        stream.write("\n")
    print(json.dumps({
        "result": str(RESULT),
        "status": result["status"],
        "candidate_oof_percent": result["candidate_scores"]["image_mean_miou_percent"],
        "control_oof_percent": result["control_scores"]["image_mean_miou_percent"],
        "candidate_vs_control_delta_pp": result["paired_candidate_vs_control"]["delta_image_mean_pp"],
        "group_bootstrap_95ci_pp": result["paired_candidate_vs_control"]["paired_group_bootstrap_95ci_pp"],
        "registered_gate_pass": result["preregistered_gate"]["pass"],
        "official_valid_used": False,
        "test_used": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
