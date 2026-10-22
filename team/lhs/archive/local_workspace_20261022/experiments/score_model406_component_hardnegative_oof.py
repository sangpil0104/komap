#!/usr/bin/env python3
"""Paired Model406 score with a preregistered small-island false-positive guard."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import numpy as np

import model396_fixed_architecture_oof as m396
import model405_small_component_fixed_blend_oof as m405
import model406_component_hardnegative_lock as m406lock
import score_model404_component_recall_oof_v6 as v6


TARGETS = m405.TARGETS
ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 406


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _claim(path: Path, oof_lock_hash: str) -> str:
    record = {
        "experiment": "Model406 component-recall plus hard-negative island suppression",
        "claimed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "oof_lock_sha256": oof_lock_hash,
        "policy": "one-shot; claim retained after interruption",
    }
    payload = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha256(path)


def component_fp_diagnostics(candidate_root: Path, control_root: Path) -> dict:
    masks_by_fold = m396.mask_files(ROOT)
    rows: list[dict] = []
    records = {
        arm: {name: [] for name in TARGETS.values()}
        for arm in ("candidate", "control")
    }
    for fold, mask_map in masks_by_fold.items():
        candidate_path = candidate_root / f"fold{fold}_oof/valid_probabilities.npz"
        control_path = control_root / f"fold{fold}_oof/valid_probabilities.npz"
        with np.load(candidate_path, allow_pickle=False) as cand, \
             np.load(control_path, allow_pickle=False) as ctrl:
            expected = set(mask_map)
            if set(cand.files) != expected or set(ctrl.files) != expected:
                raise ValueError(f"Model406 OOF IDs differ from fixed fold {fold}")
            for image_id in sorted(expected):
                truth = m396.mask_to_ids(mask_map[image_id])
                group = m405.GROUP_RE.match(image_id).group(1)
                step = int(m405.STEP_RE.search(image_id).group(1))
                rows.append({"image": image_id, "group": group, "step": step, "fold": fold})
                for arm, archive in (("candidate", cand), ("control", ctrl)):
                    probability = np.asarray(archive[image_id], dtype=np.float32)
                    if probability.shape != (4, *truth.shape) or not np.isfinite(probability).all():
                        raise ValueError(f"invalid {arm} probability map: {image_id}")
                    prediction = probability.argmax(axis=0)
                    for class_id, name in TARGETS.items():
                        records[arm][name].append(
                            m405._component_metrics(truth, prediction, class_id)
                        )
    if len(rows) != 70 or len({row["group"] for row in rows}) != 59:
        raise ValueError("Model406 component audit did not cover 70 images/59 groups")
    draws = m396.group_bootstrap_indices(rows, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    result = {}
    for name in TARGETS.values():
        result[name] = {}
        for arm in ("control", "candidate"):
            arm_rows = records[arm][name]
            result[name][arm] = {
                "unmatched_tiny_pred_components_per_image": float(np.mean([
                    item["unmatched_tiny_pred_components"] for item in arm_rows
                ])),
                "tiny_pred_components_per_image": float(np.mean([
                    item["tiny_pred_components"] for item in arm_rows
                ])),
                "target_fp_pixels_per_image": float(np.mean([
                    item["target_fp_pixels"] for item in arm_rows
                ])),
            }
        control = records["control"][name]
        candidate = records["candidate"][name]
        delta = np.asarray([
            cand["unmatched_tiny_pred_components"]
            - ctrl["unmatched_tiny_pred_components"]
            for ctrl, cand in zip(control, candidate)
        ], dtype=np.float64)
        boot = np.asarray([delta[indices].mean() for indices in draws])
        result[name]["candidate_vs_control"] = {
            "unmatched_tiny_pred_components_per_image_delta": float(delta.mean()),
            "paired_group_bootstrap_95ci": [
                float(x) for x in np.quantile(boot, [0.025, 0.975])
            ],
            "target_fp_pixels_per_image_delta": float(np.mean([
                cand["target_fp_pixels"] - ctrl["target_fp_pixels"]
                for ctrl, cand in zip(control, candidate)
            ])),
        }
    return {
        "definition": (
            "8-connected target-class predicted components of 1–63 native pixels "
            "with zero overlap with same-class GT"
        ),
        "images": 70,
        "specimen_groups": 59,
        "valid_read": False,
        "test_read": False,
        "classes": result,
    }


def score(data_root: Path, candidate_root: Path, control_root: Path,
          baseline_root: Path) -> dict:
    result = v6.score(data_root, candidate_root, control_root, baseline_root)
    audit = component_fp_diagnostics(candidate_root, control_root)
    checks = result["preregistered_gate"]["checks"]
    no_island_increase = all(
        audit["classes"][name]["candidate_vs_control"][
            "unmatched_tiny_pred_components_per_image_delta"
        ] <= 0.0
        for name in TARGETS.values()
    )
    checks["unmatched_tiny_pred_components_non_increasing_both_targets"] = no_island_increase
    result["preregistered_gate"]["pass"] = all(checks.values())
    result["status"] = "OOF_SCREEN_PASS" if result["preregistered_gate"]["pass"] else "NO_GO"
    result["experiment"] = (
        "Model406 component-recall plus hard-negative island-suppression fine-tuning"
    )
    result["objective"] = (
        "Model404 v4 balanced tiny-component recall + top-0.1% target-class hard negatives "
        "outside a one-pixel GT halo; combined auxiliary gradient calibrated to 10%"
    )
    result["small_island_false_positive_diagnostics"] = audit
    result["valid_stage_allowed"] = False
    result["decision_note"] = (
        "No valid/test look is authorized. A passing result only permits a separately "
        "locked independent-seed grouped-OOF confirmation."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--control-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    claim_path = Path(str(args.out) + ".claim")
    if args.out.exists() or claim_path.exists():
        raise FileExistsError(f"refusing to overwrite prior Model406 score/claim: {args.out}")
    m406lock.verify_oof_lock(ROOT)
    claim_hash = _claim(claim_path, _sha256(ROOT / m406lock.OOF_LOCK))
    result = score(args.data_root, args.candidate_root, args.control_root, args.baseline_root)
    result["claim_sha256"] = claim_hash
    result["source_lock_sha256"] = _sha256(ROOT / m406lock.SOURCE_LOCK)
    result["calibration_lock_sha256"] = _sha256(ROOT / m406lock.CALIBRATION_LOCK)
    result["oof_lock_sha256"] = _sha256(ROOT / m406lock.OOF_LOCK)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "result": str(args.out),
        "candidate_oof_percent": result["candidate_scores"]["image_mean_miou_percent"],
        "matched_control_oof_percent": result["control_scores"]["image_mean_miou_percent"],
        "delta_pp": result["paired_candidate_vs_control"]["delta_image_mean_pp"],
        "group_bootstrap_95ci_pp":
            result["paired_candidate_vs_control"]["paired_group_bootstrap_95ci_pp"],
        "small_island_diagnostics": result["small_island_false_positive_diagnostics"]["classes"],
        "gate_pass": result["preregistered_gate"]["pass"],
        "valid_read": False,
        "test_read": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
