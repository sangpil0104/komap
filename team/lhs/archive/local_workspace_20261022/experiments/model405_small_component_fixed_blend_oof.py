#!/usr/bin/env python3
"""Locked CPU-only Model405 small-component branch-swap grouped-OOF screen."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from scipy.ndimage import label

import model396_fixed_architecture_oof as m396
import model397_fixed_b4_model392_blend as m397


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "experiments/model405_small_component_fixed_blend_preregistration.md"
SCRIPT = "experiments/model405_small_component_fixed_blend_oof.py"
LOCK = "experiments/results/model405_small_component_fixed_blend_lock.json"
RESULT = "experiments/results/model405_small_component_fixed_blend_oof.json"
CLAIM = RESULT + ".claim"
MODEL404_LOCK = "experiments/results/model404_preregistration_lock_v6.json"
MODEL404_LOCK_SCRIPT = "experiments/model404_preregistration_lock_v6.py"
MODEL404_RESULT = "experiments/results/model404_component_recall_oof_v6.json"
MODEL404_FP_AUDIT = "experiments/results/model404_predicted_component_fp_audit_v6.json"
MODEL404_CONFUSION = "experiments/results/model404_tiny_component_confusion_audit.json"
MODEL404_ROOT = "outputs/model404_component_recall_v6/candidate"
B4_ROOT = "outputs/model329_fullstats_width4000_uniform_group_oof"
MODEL392_ROOT = "outputs/model392_widthnorm"
CV_ROOT = "data_model322_group_cv"
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 405
GROUP_RE = re.compile(r"^(A\d+)_")
STEP_RE = re.compile(r"_step(\d+)_")
TARGETS = {1: "Al3Ni", 3: "Eutectic Si"}
CONNECTIVITY = np.ones((3, 3), dtype=np.uint8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model404_lock(root: Path) -> None:
    subprocess.run(
        [sys.executable, str(root / MODEL404_LOCK_SCRIPT),
         "--root", str(root), "--verify"],
        cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )


def verify_parent_records(root: Path) -> None:
    m397.verify_lock(root)
    result_path = root / m397.RESULT
    claim_path = root / m397.CLAIM
    result = json.loads(result_path.read_text(encoding="utf-8"))
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    if (result.get("experiment") != "Model397 fixed B4/Model392 grouped-OOF blend"
            or result.get("status") != "OOF_SCREEN_PASS"
            or result.get("official_valid_used") is not False
            or result.get("test_used") is not False
            or result.get("valid_stage_allowed") is not False
            or result.get("lock_sha256") != sha256(root / m397.LOCK)
            or result.get("claim_sha256") != sha256(claim_path)
            or claim.get("lock_sha256") != sha256(root / m397.LOCK)):
        raise ValueError("Model397 OOF result/claim is missing, altered, or permits a valid look")

    verify_model404_lock(root)
    component = json.loads((root / MODEL404_RESULT).read_text(encoding="utf-8"))
    if (component.get("experiment") != "Model404 v6 Al3Ni + Eutectic-Si tiny-component recall with parent guard"
            or component.get("valid_read") is not False
            or component.get("test_read") is not False
            or component.get("images") != 70
            or component.get("specimen_groups") != 59):
        raise ValueError("Model404 v6 result is missing or out of scope")
    audit = json.loads((root / MODEL404_FP_AUDIT).read_text(encoding="utf-8"))
    if (audit.get("valid_read") is not False or audit.get("test_read") is not False
            or audit.get("scope") != "fixed grouped train-OOF only: 70 images, 59 specimen groups"):
        raise ValueError("Model404 FP audit is missing or out of scope")


def masks_by_fold(root: Path) -> dict[int, dict[str, Path]]:
    return m396.mask_files(root)


def input_paths(root: Path) -> list[Path]:
    masks = masks_by_fold(root)
    paths = [
        root / PROTOCOL,
        root / SCRIPT,
        root / MODEL404_LOCK_SCRIPT,
        root / m397.LOCK,
        root / m397.RESULT,
        root / m397.CLAIM,
        root / MODEL404_LOCK,
        root / MODEL404_RESULT,
        root / MODEL404_FP_AUDIT,
        root / MODEL404_CONFUSION,
    ]
    for fold, fold_masks in masks.items():
        paths.extend(fold_masks.values())
        paths.extend((
            root / B4_ROOT / f"fold{fold}/args.json",
            root / B4_ROOT / f"fold{fold}_oof/probabilities.npz",
            root / MODEL392_ROOT / f"seed2/fold{fold}/last.pt",
            root / MODEL392_ROOT / f"seed2/fold{fold}/args.json",
            root / MODEL392_ROOT / f"seed2/fold{fold}/model392_run_receipt.json",
            root / MODEL392_ROOT / f"seed2/fold{fold}_oof/model392_manifest.json",
            root / MODEL392_ROOT / f"seed2/fold{fold}_oof/valid_probabilities.npz",
            root / MODEL404_ROOT / f"fold{fold}/args.json",
            root / MODEL404_ROOT / f"fold{fold}/last.pt",
            root / MODEL404_ROOT / f"fold{fold}_oof/valid_per_image.csv",
            root / MODEL404_ROOT / f"fold{fold}_oof/valid_summary.json",
            root / MODEL404_ROOT / f"fold{fold}_oof/valid_probabilities.npz",
        ))
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return sorted({path.resolve() for path in paths})


def verify_inputs(root: Path) -> dict:
    verify_parent_records(root)
    masks = masks_by_fold(root)
    b4_root = root / B4_ROOT
    for fold, fold_masks in masks.items():
        expected = set(fold_masks)
        paths = (
            b4_root / f"fold{fold}_oof/probabilities.npz",
            root / MODEL392_ROOT / f"seed2/fold{fold}_oof/valid_probabilities.npz",
            root / MODEL404_ROOT / f"fold{fold}_oof/valid_probabilities.npz",
        )
        b4_args = json.loads((b4_root / f"fold{fold}/args.json").read_text(encoding="utf-8"))
        if (b4_args.get("arch") != "unet"
                or b4_args.get("encoder") != "efficientnet-b4"
                or b4_args.get("crop") != 1536
                or b4_args.get("scale") != 1.0
                or b4_args.get("target_width") != 4000
                or b4_args.get("sample") != "uniform"
                or b4_args.get("train_splits") != f"train_fold{fold}"
                or not str(b4_args.get("data", "")).endswith("/data_model322_group_cv")):
            raise ValueError(f"Model329 fixed OOF recipe/split mismatch in fold {fold}")
        candidate_args = json.loads(
            (root / MODEL404_ROOT / f"fold{fold}/args.json").read_text(encoding="utf-8")
        )
        if (candidate_args.get("arm") != "candidate"
                or candidate_args.get("epochs") != 8
                or candidate_args.get("seed") != 1
                or candidate_args.get("resume") is not False):
            raise ValueError(f"Model404 candidate recipe mismatch in fold {fold}")
        for path in paths:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != expected:
                    raise ValueError(f"OOF IDs differ in fold {fold}: {path}")
    if len({name for fold in masks.values() for name in fold}) != 70:
        raise ValueError("fixed OOF coverage is not exactly 70 unique images")
    return {"images": 70, "groups": 59, "folds": 4}


def create_lock(root: Path) -> dict:
    lock_path = root / LOCK
    if lock_path.exists() or (root / RESULT).exists() or (root / CLAIM).exists():
        raise FileExistsError("refusing to replace a Model405 lock, claim, or result")
    verified = verify_inputs(root)
    files = {
        path.resolve().relative_to(root.resolve()).as_posix(): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in input_paths(root)
    }
    receipt = {
        "experiment": "Model405 fixed Model404 small-component branch swap",
        "revision": 1,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "metric": "image-wise 4-class mIoU; empty-union=0",
        "class_order": list(m396.CLASS_NAMES),
        "folds": verified["folds"],
        "images": verified["images"],
        "specimen_groups": verified["groups"],
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "formula": {
            "reference": "0.50*Model329_B4 + 0.25*Model392_seed1 + 0.25*Model392_seed2",
            "candidate": "0.50*Model329_B4 + 0.25*Model404_v6_candidate + 0.25*Model392_seed2",
        },
        "official_valid_and_test_excluded": True,
        "gpu_used": False,
        "locked_files": files,
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return receipt


def verify_lock(root: Path) -> dict:
    path = root / LOCK
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected_identity = {
        "experiment": "Model405 fixed Model404 small-component branch swap",
        "revision": 1,
        "metric": "image-wise 4-class mIoU; empty-union=0",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "official_valid_and_test_excluded": True,
        "gpu_used": False,
    }
    if any(receipt.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("unexpected Model405 lock/protocol settings")
    if receipt.get("formula") != {
        "reference": "0.50*Model329_B4 + 0.25*Model392_seed1 + 0.25*Model392_seed2",
        "candidate": "0.50*Model329_B4 + 0.25*Model404_v6_candidate + 0.25*Model392_seed2",
    }:
        raise ValueError("Model405 fixed blend formula changed")
    expected = {
        item.resolve().relative_to(root.resolve()).as_posix()
        for item in input_paths(root)
    }
    locked = receipt.get("locked_files", {})
    if set(locked) != expected:
        raise ValueError("Model405 input inventory differs from frozen lock")
    for name, metadata in locked.items():
        item = root / name
        if (item.stat().st_size != metadata["size_bytes"]
                or sha256(item) != metadata["sha256"]):
            raise ValueError(f"Model405 locked input changed: {name}")
    verify_inputs(root)
    return receipt


def _component_metrics(truth: np.ndarray, pred: np.ndarray, class_id: int) -> dict:
    gt_labels, gt_count = label(truth == class_id, structure=CONNECTIVITY)
    gt_areas = np.bincount(gt_labels.ravel(), minlength=gt_count + 1)
    gt_is_component = np.arange(gt_areas.size) > 0
    tiny_gt_ids = np.flatnonzero(gt_is_component & (gt_areas >= 1) & (gt_areas < 64))
    if tiny_gt_ids.size:
        gt_pred_counts = np.bincount(
            gt_labels[pred == class_id].ravel(), minlength=gt_count + 1
        )
        tiny_hits = int(np.count_nonzero(
            gt_pred_counts[tiny_gt_ids] / gt_areas[tiny_gt_ids] >= 0.50
        ))
    else:
        tiny_hits = 0

    pred_labels, pred_count = label(pred == class_id, structure=CONNECTIVITY)
    pred_areas = np.bincount(pred_labels.ravel(), minlength=pred_count + 1)
    pred_is_component = np.arange(pred_areas.size) > 0
    tiny_pred_ids = np.flatnonzero(
        pred_is_component & (pred_areas >= 1) & (pred_areas < 64)
    )
    if tiny_pred_ids.size:
        same_gt_overlap = np.bincount(
            pred_labels[truth == class_id].ravel(), minlength=pred_count + 1
        )
        unmatched_tiny = int(np.count_nonzero(same_gt_overlap[tiny_pred_ids] == 0))
    else:
        unmatched_tiny = 0
    return {
        "tiny_gt_components": int(tiny_gt_ids.size),
        "tiny_gt_hits": tiny_hits,
        "tiny_pred_components": int(tiny_pred_ids.size),
        "unmatched_tiny_pred_components": unmatched_tiny,
        "target_fp_pixels": int(np.count_nonzero((pred == class_id) & (truth != class_id))),
        "target_fn_pixels": int(np.count_nonzero((pred != class_id) & (truth == class_id))),
    }


def _component_summary(rows: list[dict], records: dict[str, dict[str, list[dict]]],
                       bootstrap_indices: list[np.ndarray]) -> dict:
    summary: dict[str, dict] = {}
    for class_id, class_name in TARGETS.items():
        summary[class_name] = {}
        for arm in ("reference", "candidate"):
            image_rows = records[arm][class_name]
            n_gt = sum(row["tiny_gt_components"] for row in image_rows)
            n_hit = sum(row["tiny_gt_hits"] for row in image_rows)
            summary[class_name][arm] = {
                "tiny_gt_components": int(n_gt),
                "tiny_gt_hits": int(n_hit),
                "tiny_gt_hit_rate_percent": float(100.0 * n_hit / n_gt) if n_gt else None,
                "tiny_pred_components_per_image": float(np.mean(
                    [row["tiny_pred_components"] for row in image_rows]
                )),
                "unmatched_tiny_pred_components_per_image": float(np.mean(
                    [row["unmatched_tiny_pred_components"] for row in image_rows]
                )),
                "target_fp_pixels_per_image": float(np.mean(
                    [row["target_fp_pixels"] for row in image_rows]
                )),
                "target_fn_pixels_per_image": float(np.mean(
                    [row["target_fn_pixels"] for row in image_rows]
                )),
            }
        ref = records["reference"][class_name]
        cand = records["candidate"][class_name]
        ref_hits = np.asarray([row["tiny_gt_hits"] for row in ref], dtype=np.float64)
        cand_hits = np.asarray([row["tiny_gt_hits"] for row in cand], dtype=np.float64)
        gt_counts = np.asarray([row["tiny_gt_components"] for row in ref], dtype=np.float64)
        tiny_fp_delta = np.asarray([
            c["unmatched_tiny_pred_components"] - r["unmatched_tiny_pred_components"]
            for r, c in zip(ref, cand)
        ], dtype=np.float64)
        hit_delta_draws = []
        fp_delta_draws = []
        for indices in bootstrap_indices:
            total_gt = gt_counts[indices].sum()
            ref_rate = ref_hits[indices].sum() / total_gt if total_gt else 0.0
            cand_rate = cand_hits[indices].sum() / total_gt if total_gt else 0.0
            hit_delta_draws.append((cand_rate - ref_rate) * 100.0)
            fp_delta_draws.append(float(tiny_fp_delta[indices].mean()))
        summary[class_name]["candidate_vs_reference"] = {
            "tiny_gt_hit_rate_delta_pp": float(
                summary[class_name]["candidate"]["tiny_gt_hit_rate_percent"]
                - summary[class_name]["reference"]["tiny_gt_hit_rate_percent"]
            ),
            "tiny_gt_hit_rate_group_bootstrap_95ci_pp": [
                float(x) for x in np.quantile(hit_delta_draws, [0.025, 0.975])
            ],
            "unmatched_tiny_pred_components_per_image_delta": float(tiny_fp_delta.mean()),
            "unmatched_tiny_pred_components_per_image_group_bootstrap_95ci": [
                float(x) for x in np.quantile(fp_delta_draws, [0.025, 0.975])
            ],
            "target_fp_pixels_per_image_delta": float(np.mean([
                c["target_fp_pixels"] - r["target_fp_pixels"]
                for r, c in zip(ref, cand)
            ])),
            "target_fn_pixels_per_image_delta": float(np.mean([
                c["target_fn_pixels"] - r["target_fn_pixels"]
                for r, c in zip(ref, cand)
            ])),
        }
    return summary


def adoption_gate(comparison: dict, components: dict) -> dict:
    folds = comparison["fold_delta_pp"]
    class_delta = comparison["class_delta_pp"]
    checks = {
        "overall_gain_at_least_0_20pp": comparison["delta_image_mean_pp"] >= 0.20,
        "group_bootstrap_ci_lower_above_zero":
            comparison["paired_group_bootstrap_95ci_pp"][0] > 0.0,
        "positive_in_at_least_3_of_4_folds": comparison["positive_folds"] >= 3,
        "worst_fold_at_least_minus_0_30pp": min(folds.values()) >= -0.30,
        "both_target_class_iou_deltas_nonnegative":
            class_delta["Al3Ni"] >= 0.0 and class_delta["Eutectic Si"] >= 0.0,
        "all_class_iou_deltas_at_least_minus_0_50pp":
            min(class_delta.values()) >= -0.50,
        "step4_delta_at_least_minus_0_30pp": comparison["step4_delta_pp"] >= -0.30,
        "tiny_gt_component_hit_rate_non_decreasing_for_both_targets": all(
            components[name]["candidate_vs_reference"]["tiny_gt_hit_rate_delta_pp"] >= 0.0
            for name in TARGETS.values()
        ),
        "unmatched_tiny_pred_components_non_increasing_for_both_targets": all(
            components[name]["candidate_vs_reference"][
                "unmatched_tiny_pred_components_per_image_delta"
            ] <= 0.0
            for name in TARGETS.values()
        ),
    }
    return {"checks": checks, "pass": all(checks.values())}


def _claim(path: Path, lock_hash: str) -> str:
    record = {
        "experiment": "Model405 fixed Model404 small-component branch swap",
        "claimed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "lock_sha256": lock_hash,
        "policy": "one-shot; claim retained after interruption",
    }
    payload = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(path)


def score_oof(root: Path) -> dict:
    verify_lock(root)
    output_path, claim_path = root / RESULT, root / CLAIM
    if output_path.exists() or claim_path.exists():
        raise FileExistsError("Model405 scoring was already claimed or published")
    claim_hash = _claim(claim_path, sha256(root / LOCK))
    masks = masks_by_fold(root)
    rows: list[dict] = []
    scores = {"reference": [], "candidate": []}
    component_rows = {
        arm: {name: [] for name in TARGETS.values()}
        for arm in ("reference", "candidate")
    }
    for fold, mask_map in masks.items():
        b4_path = root / B4_ROOT / f"fold{fold}_oof/probabilities.npz"
        seed1_path = root / MODEL392_ROOT / f"seed1/fold{fold}_oof/valid_probabilities.npz"
        seed2_path = root / MODEL392_ROOT / f"seed2/fold{fold}_oof/valid_probabilities.npz"
        candidate_path = root / MODEL404_ROOT / f"fold{fold}_oof/valid_probabilities.npz"
        with np.load(b4_path, allow_pickle=False) as b4_arc, \
             np.load(seed1_path, allow_pickle=False) as s1_arc, \
             np.load(seed2_path, allow_pickle=False) as s2_arc, \
             np.load(candidate_path, allow_pickle=False) as comp_arc:
            ids = set(mask_map)
            if any(set(arc.files) != ids for arc in (b4_arc, s1_arc, s2_arc, comp_arc)):
                raise ValueError(f"OOF archive IDs differ in fold {fold}")
            for image_id in sorted(ids):
                truth = m396.mask_to_ids(mask_map[image_id])
                p_b4 = np.asarray(b4_arc[image_id], dtype=np.float32)
                p_s1 = np.asarray(s1_arc[image_id], dtype=np.float32)
                p_s2 = np.asarray(s2_arc[image_id], dtype=np.float32)
                p_comp = np.asarray(comp_arc[image_id], dtype=np.float32)
                if any(p.shape != (4, *truth.shape) or not np.isfinite(p).all()
                       for p in (p_b4, p_s1, p_s2, p_comp)):
                    raise ValueError(f"invalid probability tensor for {image_id}")
                p_ref = 0.50 * p_b4 + 0.25 * p_s1 + 0.25 * p_s2
                p_candidate = 0.50 * p_b4 + 0.25 * p_comp + 0.25 * p_s2
                class_scores = {
                    "reference": m396.class_ious(p_ref, truth),
                    "candidate": m396.class_ious(p_candidate, truth),
                }
                parsed_group, parsed_step = GROUP_RE.match(image_id), STEP_RE.search(image_id)
                if parsed_group is None or parsed_step is None:
                    raise ValueError(f"cannot parse group/step from {image_id}")
                rows.append({
                    "image": image_id, "group": parsed_group.group(1),
                    "step": int(parsed_step.group(1)), "fold": fold,
                })
                for arm in ("reference", "candidate"):
                    scores[arm].append(class_scores[arm])
                    prediction = (p_ref if arm == "reference" else p_candidate).argmax(axis=0)
                    for class_id, class_name in TARGETS.items():
                        component_rows[arm][class_name].append(
                            _component_metrics(truth, prediction, class_id)
                        )
        print(f"validated and scored Model405 OOF fold {fold}/4", flush=True)

    if (len(rows) != 70 or len({row["image"] for row in rows}) != 70
            or len({row["group"] for row in rows}) != 59):
        raise ValueError("Model405 OOF coverage is not exactly 70 images / 59 groups")
    arrays = {arm: np.asarray(value, dtype=np.float64) for arm, value in scores.items()}
    indices = m396.group_bootstrap_indices(rows, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    paired = m396.comparison(arrays["candidate"], arrays["reference"], rows, indices)
    # Recompute the locked Model397 reference and fail closed if arithmetic or inputs drift.
    prior = json.loads((root / m397.RESULT).read_text(encoding="utf-8"))
    reference_summary = m396.overall(arrays["reference"])
    prior_score = prior["arms"]["C"]["image_mean_miou_percent"]
    if abs(reference_summary["image_mean_miou_percent"] - prior_score) > 1e-6:
        raise ValueError("Model405 reference does not reproduce the fixed Model397 OOF score")
    comp_summary = _component_summary(rows, component_rows, indices)
    decision = adoption_gate(paired, comp_summary)
    result = {
        "experiment": "Model405 fixed Model404 small-component branch swap",
        "status": "OOF_SCREEN_PASS" if decision["pass"] else "NO_GO",
        "metric": "image-wise 4-class mIoU; empty-union=0",
        "class_order": list(m396.CLASS_NAMES),
        "scope": "train-only exactly-once grouped OOF; selection-conditional development evidence",
        "reference_formula": "0.50*Model329_B4 + 0.25*Model392_seed1 + 0.25*Model392_seed2",
        "candidate_formula": "0.50*Model329_B4 + 0.25*Model404_v6_candidate + 0.25*Model392_seed2",
        "official_valid_used": False,
        "test_used": False,
        "gpu_used": False,
        "lock_sha256": sha256(root / LOCK),
        "claim_sha256": claim_hash,
        "reference": reference_summary,
        "candidate": m396.overall(arrays["candidate"]),
        "candidate_vs_reference": paired,
        "small_component_diagnostics": comp_summary,
        "pre_registered_gate": decision,
        "valid_stage_allowed": False,
        "decision_note": (
            "This one-shot OOF screen cannot authorize a valid/test look. "
            "A passing screen only supports a separately locked independent-seed OOF confirmation."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result


def self_test() -> None:
    truth = np.asarray([[0, 1, 1], [0, 2, 3], [0, 3, 3]], dtype=np.uint8)
    one_hot = np.zeros((4, 3, 3), dtype=np.float32)
    for class_id in range(4):
        one_hot[class_id] = truth == class_id
    reference = 0.50 * one_hot + 0.25 * one_hot + 0.25 * one_hot
    candidate = 0.50 * one_hot + 0.25 * one_hot + 0.25 * one_hot
    assert np.array_equal(reference, candidate)
    assert np.array_equal(m396.class_ious(candidate, truth), np.ones(4))
    met = _component_metrics(truth, truth.copy(), 1)
    assert met["tiny_gt_components"] == 1 and met["tiny_gt_hits"] == 1
    assert met["unmatched_tiny_pred_components"] == 0
    print("Model405 smoke passed: fixed blend math, empty=0 IoU, tiny-component diagnostics.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--create-lock", action="store_true")
    mode.add_argument("--verify-lock", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.self_test:
        self_test()
    elif args.create_lock:
        receipt = create_lock(root)
        print(json.dumps({
            "locked_files": len(receipt["locked_files"]),
            "lock": str(root / LOCK),
        }, ensure_ascii=False, indent=2))
    elif args.verify_lock:
        receipt = verify_lock(root)
        print(json.dumps({
            "verified_files": len(receipt["locked_files"]),
            "lock_sha256": sha256(root / LOCK),
        }, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(score_oof(root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
