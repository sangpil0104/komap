#!/usr/bin/env python3
"""Complete Model400's interrupted fixed inference without rescoring/tuning."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image

import model400_model396_d2_valid_once as base


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "experiments/model400_inference_recovery.py"
SMOKE = "experiments/model400_inference_recovery_smoke.py"
ADDENDUM = "experiments/model400_inference_recovery_addendum.md"
RECOVERY_LOCK = "experiments/results/model400_inference_recovery_lock.json"
RECOVERY_RESULT = "experiments/results/model400_inference_recovery.json"
EXPECTED_PARTIAL_ID = "A117_step4_sample2"


def _canonical_hash(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _exclusive_json(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def _expected_map_path(root: Path, image_id: str) -> Path:
    return root / base.HRNET_MEAN_DIR / f"{image_id}.npz"


def _read_map(path: Path, image_id: str, shape: tuple[int, int]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {image_id}:
            raise ValueError(f"unexpected archive IDs in {path}")
        probs = np.asarray(archive[image_id], dtype=np.float32)
    base._check_probs(probs, shape, f"recovery/{image_id}")
    return probs


def create_recovery_lock(root: Path) -> dict[str, Any]:
    path = root / RECOVERY_LOCK
    if path.exists() or (root / RECOVERY_RESULT).exists():
        raise FileExistsError("refusing to overwrite/repeat the Model400 recovery lock")
    original_lock = base.verify_lock(root)
    if (root / base.RESULT).exists() or (root / base.CLAIM).exists():
        raise FileExistsError("Model400 scoring has already started; recovery is not allowed")
    output_root = root / base.INFER_ROOT
    claim_path = root / base.INFER_CLAIM
    manifest_path = root / base.INFER_MANIFEST
    mean_dir = root / base.HRNET_MEAN_DIR
    if (not output_root.is_dir() or not claim_path.is_file() or manifest_path.exists()
            or not mean_dir.is_dir()):
        raise ValueError("Model400 output is not in the expected interrupted-inference state")
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    if (claim.get("stage") != "inference"
            or claim.get("lock_sha256") != original_lock["lock_sha256"]):
        raise ValueError("existing inference claim does not match the frozen Model400 lock")
    images, _ = base.valid_pairs(root)
    expected_ids = sorted(images)
    if not expected_ids or expected_ids[0] != EXPECTED_PARTIAL_ID:
        raise ValueError("fixed valid image ordering differs from the observed interrupted map")
    existing_files = sorted(path.name for path in mean_dir.glob("*.npz"))
    if existing_files != [f"{EXPECTED_PARTIAL_ID}.npz"]:
        raise ValueError("expected exactly the original first-image map; refusing ambiguous recovery")
    map_path = _expected_map_path(root, EXPECTED_PARTIAL_ID)
    if map_path.is_symlink() or not map_path.is_file():
        raise ValueError("partial inference map is missing or is not a regular file")
    with Image.open(images[EXPECTED_PARTIAL_ID]) as image:
        shape = (image.height, image.width)
    _read_map(map_path, EXPECTED_PARTIAL_ID, shape)
    paths = (root / SCRIPT, root / SMOKE, root / ADDENDUM, root / base.LOCK,
             root / base.SCRIPT, claim_path, map_path)
    files = {
        item.resolve().relative_to(root.resolve()).as_posix(): {
            "size_bytes": item.stat().st_size,
            "sha256": base.sha256(item),
        }
        for item in paths
    }
    record = {
        "experiment": "Model400 interrupted inference recovery",
        "revision": 1,
        "decision": "resume only missing deterministic maps; no labels or score",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "base_model400_lock_sha256": original_lock["lock_sha256"],
        "expected_valid_ids": expected_ids,
        "preserved_partial_map": {
            "image_id": EXPECTED_PARTIAL_ID,
            "path": map_path.resolve().relative_to(root.resolve()).as_posix(),
            "shape": list(shape),
            "sha256": base.sha256(map_path),
        },
        "recovery_files": files,
        "runtime": base._runtime_record(),
        "official_valid_mask_pixels_decoded": False,
        "official_valid_score_computed": False,
        "test_inputs_excluded": True,
    }
    record["lock_sha256"] = _canonical_hash(record)
    _exclusive_json(path, record)
    return record


def verify_recovery_lock(root: Path) -> dict[str, Any]:
    original_lock = base.verify_lock(root)
    path = root / RECOVERY_LOCK
    record = json.loads(path.read_text(encoding="utf-8"))
    digest = record.get("lock_sha256")
    body = {key: value for key, value in record.items() if key != "lock_sha256"}
    if digest != _canonical_hash(body):
        raise ValueError("Model400 recovery lock hash is invalid")
    if (record.get("experiment") != "Model400 interrupted inference recovery"
            or record.get("base_model400_lock_sha256") != original_lock["lock_sha256"]
            or record.get("runtime") != base._runtime_record()
            or record.get("official_valid_mask_pixels_decoded") is not False
            or record.get("official_valid_score_computed") is not False
            or record.get("test_inputs_excluded") is not True):
        raise ValueError("Model400 recovery lock does not match the base frozen protocol")
    for relative, metadata in record.get("recovery_files", {}).items():
        source = root / relative
        if (not source.is_file() or source.stat().st_size != metadata["size_bytes"]
                or base.sha256(source) != metadata["sha256"]):
            raise ValueError(f"recovery-locked file changed: {relative}")
    if (root / base.RESULT).exists() or (root / base.CLAIM).exists():
        raise FileExistsError("Model400 score was already claimed; inference recovery is closed")
    return record


def _write_manifest(root: Path, recovery_lock: dict[str, Any], expected_ids: list[str]) -> dict[str, Any]:
    images, _ = base.valid_pairs(root)
    original_lock = base.verify_lock(root)
    claim_path = root / base.INFER_CLAIM
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    if (claim.get("stage") != "inference"
            or claim.get("lock_sha256") != original_lock["lock_sha256"]):
        raise ValueError("inference claim changed while recovery ran")
    map_receipts = []
    for image_id in expected_ids:
        map_path = _expected_map_path(root, image_id)
        with Image.open(images[image_id]) as image:
            shape = (image.height, image.width)
        _read_map(map_path, image_id, shape)
        map_receipts.append({
            "image_id": image_id,
            "path": map_path.resolve().relative_to(root.resolve()).as_posix(),
            "sha256": base.sha256(map_path),
            "shape": list(shape),
        })
    checkpoint_hashes = {
        str(fold): base.sha256(root / base.HRNET_ROOT / f"fold{fold}/last.pt")
        for fold in range(1, 5)
    }
    manifest = {
        "experiment": "Model400 Model396-D2 four-fold valid ensemble",
        "status": "INFERENCE_COMPLETE_NOT_SCORED",
        "lock_sha256": original_lock["lock_sha256"],
        "inference_claim_sha256": base.sha256(claim_path),
        "image_ids": expected_ids,
        "hrnet_seed2_fold_checkpoint_sha256": checkpoint_hashes,
        "hrnet_seed2_mean_probability_maps": map_receipts,
        "b4_valid_probabilities_sha256": base.sha256(
            root / base.B4_ROOT / "valid_ensemble/probabilities.npz"
        ),
        "configuration": base.INFERENCE_CONFIG,
        "official_valid_mask_pixels_decoded_during_inference": False,
        "official_valid_score_computed": False,
        "test_inputs_excluded": True,
        "recovery_lock_sha256": recovery_lock["lock_sha256"],
    }
    _exclusive_json(root / base.INFER_MANIFEST, manifest)
    recovery_result = {
        "experiment": "Model400 interrupted inference recovery",
        "status": "RECOVERED_INFERENCE_COMPLETE_NOT_SCORED",
        "recovery_lock_sha256": recovery_lock["lock_sha256"],
        "base_model400_lock_sha256": original_lock["lock_sha256"],
        "completed_images": len(expected_ids),
        "preserved_original_map": EXPECTED_PARTIAL_ID,
        "inference_manifest_sha256": base.sha256(root / base.INFER_MANIFEST),
        "official_valid_mask_pixels_decoded": False,
        "official_valid_score_computed": False,
        "test_inputs_excluded": True,
    }
    _exclusive_json(root / RECOVERY_RESULT, recovery_result)
    return recovery_result


def resume_inference(root: Path, device_name: str) -> dict[str, Any]:
    recovery_lock = verify_recovery_lock(root)
    manifest_path = root / base.INFER_MANIFEST
    if manifest_path.exists():
        raise FileExistsError("inference manifest already exists; refusing to resume/overwrite")
    images, _ = base.valid_pairs(root)
    expected_ids = sorted(images)
    if expected_ids != recovery_lock["expected_valid_ids"]:
        raise ValueError("valid IDs differ from the recovery lock")
    mean_dir = root / base.HRNET_MEAN_DIR
    allowed_names = {f"{image_id}.npz" for image_id in expected_ids}
    unexpected = {path.name for path in mean_dir.glob("*")} - allowed_names
    if unexpected:
        raise ValueError(f"unexpected files in recovery map directory: {sorted(unexpected)}")
    partial = recovery_lock["preserved_partial_map"]
    partial_path = root / partial["path"]
    if base.sha256(partial_path) != partial["sha256"]:
        raise ValueError("original partial map changed after recovery lock")

    model_dir = root / "model336_hrnet_best"
    sys.path.insert(0, str(model_dir))
    import torch
    from komap_seg.checkpoint import load_model
    from komap_seg.dataset import KomapFullDataset
    from komap_seg.inference import predict_probs_tta

    device = torch.device(device_name)
    dataset = KomapFullDataset(root / base.VALID_IMAGES, masks_dir=None)
    dataset_ids = sorted(sample["stem"] for sample in dataset.pairs)
    if dataset_ids != expected_ids or len(dataset) != base.EXPECTED_VALID_IMAGES:
        raise ValueError("image-only inference dataset differs from the recovery lock")
    sample_index = {sample["stem"]: index for index, sample in enumerate(dataset.pairs)}
    models = [
        load_model(root / base.HRNET_ROOT / f"fold{fold}/last.pt", device)
        for fold in range(1, 5)
    ]
    for index, image_id in enumerate(expected_ids, start=1):
        map_path = _expected_map_path(root, image_id)
        with Image.open(images[image_id]) as image:
            shape = (image.height, image.width)
        if map_path.exists():
            if image_id == EXPECTED_PARTIAL_ID:
                if base.sha256(map_path) != partial["sha256"]:
                    raise ValueError("preserved first-image map no longer matches recovery lock")
            _read_map(map_path, image_id, shape)
            print(json.dumps({"stage": "model400_recovery", "image": index,
                              "images": len(expected_ids), "action": "preserved"}), flush=True)
            continue
        sample = dataset[sample_index[image_id]]
        probability_sum = np.zeros((base.NUM_CLASSES, *shape), dtype=np.float32)
        for fold, model in enumerate(models, start=1):
            probs = predict_probs_tta(
                model,
                sample["image"],
                patch_size=512,
                overlap=170,
                batch_size=2,
                tta=True,
                input_scale=2.0,
                device=device,
            ).numpy().astype(np.float32)
            base._check_probs(probs, shape, f"HRNet seed2 fold{fold}/{image_id}")
            probability_sum += probs
        mean = probability_sum / 4.0
        base._check_probs(mean, shape, f"HRNet seed2 four-fold mean/{image_id}")
        base.save_npz_exclusive(map_path, **{image_id: mean.astype(np.float16)})
        print(json.dumps({"stage": "model400_recovery", "image": index,
                          "images": len(expected_ids), "action": "predicted"}), flush=True)
    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if set(path.name for path in mean_dir.glob("*.npz")) != allowed_names:
        raise ValueError("recovery did not produce exactly one map per locked valid image")
    return _write_manifest(root, recovery_lock, expected_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--create-lock", action="store_true")
    actions.add_argument("--resume-inference", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.create_lock:
        record = create_recovery_lock(root)
        print(json.dumps({"status": "RECOVERY_LOCKED",
                          "lock_sha256": record["lock_sha256"]}, indent=2))
    else:
        record = resume_inference(root, args.device)
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
