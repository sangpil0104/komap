#!/usr/bin/env python3
"""Finish Model400's interrupted fixed score after locking a narrow recovery."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image

import model400_model396_d2_valid_once as base


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "experiments/model400_score_recovery.py"
SMOKE = "experiments/model400_score_recovery_smoke.py"
ADDENDUM = "experiments/model400_score_recovery_addendum.md"
RECOVERY_LOCK = "experiments/results/model400_score_recovery_lock.json"
RECOVERY_CLAIM = base.RESULT + ".recovery.claim"
RECOVERY_RECEIPT = "experiments/results/model400_score_recovery_receipt.json"
FAILED_IMAGE_ID = "A117_step4_sample2"


def _canonical_hash(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _json_sha(path: Path) -> str:
    return base.sha256(path)


def _current_preflight(root: Path) -> tuple[dict[str, Any], dict[str, Path], dict[str, Path], list[str]]:
    original_lock = base.verify_lock(root)
    result_path = root / base.RESULT
    if result_path.exists():
        raise FileExistsError("Model400 result already exists; no score recovery is allowed")
    original_claim_path = root / base.CLAIM
    if not original_claim_path.is_file():
        raise FileNotFoundError("interrupted original Model400 score claim is missing")
    original_claim = json.loads(original_claim_path.read_text(encoding="utf-8"))
    if (original_claim.get("stage") != "official_valid_scoring"
            or original_claim.get("lock_sha256") != original_lock["lock_sha256"]):
        raise ValueError("original score claim is not tied to the frozen Model400 lock")
    images, masks = base.valid_pairs(root)
    expected_ids = sorted(images)
    if not expected_ids or expected_ids[0] != FAILED_IMAGE_ID:
        raise ValueError("valid image order differs from the observed failed scorer row")
    manifest_path = root / base.INFER_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "INFERENCE_COMPLETE_NOT_SCORED"
            or manifest.get("lock_sha256") != original_lock["lock_sha256"]
            or manifest.get("image_ids") != expected_ids
            or manifest.get("official_valid_score_computed") is not False
            or manifest.get("test_inputs_excluded") is not True):
        raise ValueError("completed inference manifest is not eligible for score recovery")
    map_items = manifest.get("hrnet_seed2_mean_probability_maps", [])
    map_receipts = {item["image_id"]: item for item in map_items}
    if len(map_items) != len(expected_ids) or set(map_receipts) != set(expected_ids):
        raise ValueError("recovery requires exactly 20 locked HRNet probability maps")
    for image_id in expected_ids:
        item = map_receipts[image_id]
        expected_path = (root / base.HRNET_MEAN_DIR / f"{image_id}.npz").resolve()
        map_path = (root / item["path"]).resolve()
        if map_path != expected_path or not map_path.is_file():
            raise ValueError(f"unexpected HRNet map path for {image_id}")
        if base.sha256(map_path) != item.get("sha256"):
            raise ValueError(f"HRNet map hash mismatch for {image_id}")
        with Image.open(images[image_id]) as image:
            shape = (image.height, image.width)
        if tuple(item.get("shape", ())) != shape:
            raise ValueError(f"HRNet map shape receipt mismatch for {image_id}")
        with np.load(map_path, allow_pickle=False) as archive:
            if set(archive.files) != {image_id}:
                raise ValueError(f"unexpected map IDs for {image_id}")
            probs = np.asarray(archive[image_id], dtype=np.float32)
        base._check_probs(probs, shape, f"score-recovery/{image_id}")
    b4_path = root / base.B4_ROOT / "valid_ensemble/probabilities.npz"
    if manifest.get("b4_valid_probabilities_sha256") != base.sha256(b4_path):
        raise ValueError("B4 probability archive hash differs from inference manifest")
    with np.load(b4_path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected_ids):
            raise ValueError("B4 archive IDs differ from the locked valid IDs")
        for image_id in expected_ids:
            with Image.open(images[image_id]) as image:
                shape = (image.height, image.width)
            base._check_probs(np.asarray(archive[image_id]), shape, f"B4/{image_id}")
    if (root / RECOVERY_CLAIM).exists() or (root / RECOVERY_RECEIPT).exists():
        raise FileExistsError("Model400 score recovery already claimed or recorded")
    return original_lock, images, masks, expected_ids


def _recovery_input_paths(root: Path, images: dict[str, Path], masks: dict[str, Path],
                          expected_ids: list[str]) -> list[Path]:
    paths = [
        root / SCRIPT,
        root / SMOKE,
        root / ADDENDUM,
        root / base.SCRIPT,
        root / base.LOCK,
        root / base.INFER_MANIFEST,
        root / base.INFER_CLAIM,
        root / base.CLAIM,
        root / base.B4_ROOT / "valid_ensemble/probabilities.npz",
    ]
    paths.extend(images.values())
    paths.extend(masks.values())
    paths.extend(root / base.HRNET_MEAN_DIR / f"{image_id}.npz" for image_id in expected_ids)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return sorted({path.resolve() for path in paths})


def create_lock(root: Path) -> dict[str, Any]:
    lock_path = root / RECOVERY_LOCK
    if lock_path.exists():
        raise FileExistsError("refusing to overwrite the Model400 score-recovery lock")
    original_lock, images, masks, expected_ids = _current_preflight(root)
    files = {
        path.resolve().relative_to(root.resolve()).as_posix(): {
            "size_bytes": path.stat().st_size,
            "sha256": _json_sha(path),
        }
        for path in _recovery_input_paths(root, images, masks, expected_ids)
    }
    record = {
        "experiment": "Model400 score-execution recovery",
        "revision": 1,
        "decision": "complete same frozen metric after logger-only interruption; no tuning",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "base_model400_lock_sha256": original_lock["lock_sha256"],
        "failed_first_image": FAILED_IMAGE_ID,
        "original_score_claim_sha256": base.sha256(root / base.CLAIM),
        "expected_valid_ids": expected_ids,
        "metric": "image-wise 4-class mIoU; empty-union=0",
        "recovery_files": files,
        "runtime": base._runtime_record(),
        "original_attempt_result_written": False,
        "one_image_metric_computed_in_memory": True,
        "additional_candidate_selection": False,
        "test_inputs_excluded": True,
    }
    record["lock_sha256"] = _canonical_hash(record)
    base.write_exclusive(lock_path, record)
    return record


def verify_lock(root: Path) -> dict[str, Any]:
    original_lock, images, masks, expected_ids = _current_preflight(root)
    path = root / RECOVERY_LOCK
    record = json.loads(path.read_text(encoding="utf-8"))
    digest = record.get("lock_sha256")
    body = {key: value for key, value in record.items() if key != "lock_sha256"}
    if digest != _canonical_hash(body):
        raise ValueError("Model400 score-recovery lock hash is invalid")
    if (record.get("experiment") != "Model400 score-execution recovery"
            or record.get("base_model400_lock_sha256") != original_lock["lock_sha256"]
            or record.get("original_score_claim_sha256") != base.sha256(root / base.CLAIM)
            or record.get("expected_valid_ids") != expected_ids
            or record.get("metric") != "image-wise 4-class mIoU; empty-union=0"
            or record.get("runtime") != base._runtime_record()
            or record.get("original_attempt_result_written") is not False
            or record.get("one_image_metric_computed_in_memory") is not True
            or record.get("additional_candidate_selection") is not False
            or record.get("test_inputs_excluded") is not True):
        raise ValueError("score-recovery lock differs from the reviewed recovery scope")
    expected_paths = {
        item.resolve().relative_to(root.resolve()).as_posix()
        for item in _recovery_input_paths(root, images, masks, expected_ids)
    }
    locked = record.get("recovery_files", {})
    if set(locked) != expected_paths:
        raise ValueError("score-recovery locked input inventory changed")
    for relative, metadata in locked.items():
        source = root / relative
        if (source.stat().st_size != metadata["size_bytes"]
                or base.sha256(source) != metadata["sha256"]):
            raise ValueError(f"score-recovery input changed: {relative}")
    if (root / RECOVERY_CLAIM).exists() or (root / RECOVERY_RECEIPT).exists():
        raise FileExistsError("score-recovery attempt already claimed or recorded")
    return record


def _with_json_flush_compatibility() -> tuple[Any, Any]:
    original = json.dumps

    def compatible_dumps(*args: Any, **kwargs: Any) -> str:
        kwargs.pop("flush", None)
        return original(*args, **kwargs)

    json.dumps = compatible_dumps
    return original, compatible_dumps


def score_once(root: Path) -> dict[str, Any]:
    recovery_lock = verify_lock(root)
    original_claim_path = root / base.CLAIM
    original_claim_hash = base.sha256(original_claim_path)
    old_claim_constant = base.CLAIM
    base.CLAIM = RECOVERY_CLAIM
    original_dumps, _ = _with_json_flush_compatibility()
    try:
        result = base.score_once(root)
    finally:
        json.dumps = original_dumps
        base.CLAIM = old_claim_constant
    receipt = {
        "experiment": "Model400 score-execution recovery",
        "status": "FIXED_MODEL400_RESULT_WRITTEN",
        "recovery_lock_sha256": recovery_lock["lock_sha256"],
        "base_model400_lock_sha256": recovery_lock["base_model400_lock_sha256"],
        "original_interrupted_score_claim_sha256": original_claim_hash,
        "recovery_score_claim_sha256": base.sha256(root / RECOVERY_CLAIM),
        "result_sha256": base.sha256(root / base.RESULT),
        "images_scored": result["images"],
        "metric": result["metric"],
        "official_valid_score_computed": True,
        "test_inputs_excluded": True,
        "no_additional_candidate_selection": True,
    }
    base.write_exclusive(root / RECOVERY_RECEIPT, receipt)
    return result


def self_test() -> None:
    original = json.dumps
    original_dumps, _ = _with_json_flush_compatibility()
    try:
        encoded = json.dumps({"ok": True}, flush=True)
        assert json.loads(encoded) == {"ok": True}
    finally:
        json.dumps = original
        assert json.dumps is original
        assert original_dumps is original
    print("Model400 score-recovery smoke passed; JSON logging shim only.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--self-test", action="store_true")
    actions.add_argument("--create-lock", action="store_true")
    actions.add_argument("--score", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.self_test:
        self_test()
    elif args.create_lock:
        record = create_lock(root)
        print(json.dumps({"status": "RECOVERY_LOCKED",
                          "lock_sha256": record["lock_sha256"]}, indent=2))
    else:
        result = score_once(root)
        print(json.dumps({"status": result["status"],
                          "image_mean_miou_percent": result["image_mean_miou_percent"],
                          "difference_from_recorded_model364_pp": result["difference_from_recorded_model364_pp"]},
                         indent=2))


if __name__ == "__main__":
    main()
