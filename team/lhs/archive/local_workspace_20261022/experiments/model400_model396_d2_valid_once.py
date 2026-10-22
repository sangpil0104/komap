#!/usr/bin/env python3
"""Locked one-shot Model396-D2 four-fold validation ensemble.

The scorer is gated by an independently reviewed protocol. Inference does not
decode masks; score mode is exclusive and reads labels only after all HRNet
maps are complete and verified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
from PIL import Image

import model396_fixed_architecture_oof as model396


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "experiments/model400_model396_d2_valid_ensemble_preregistration.md"
SCRIPT = "experiments/model400_model396_d2_valid_once.py"
SMOKE = "experiments/model400_model396_d2_valid_once_smoke.py"
LOCK = "experiments/results/model400_model396_d2_valid_lock.json"
RESULT = "experiments/results/model400_model396_d2_valid_once.json"
CLAIM = RESULT + ".claim"
INFER_ROOT = "outputs/model400_model396_d2_valid_once"
INFER_CLAIM = INFER_ROOT + "/inference.claim"
INFER_MANIFEST = INFER_ROOT + "/inference_manifest.json"
HRNET_MEAN_DIR = INFER_ROOT + "/hrnet_seed2_mean_valid"
VALID_IMAGES = "data/valid/images"
VALID_MASKS = "data/valid/masks"
B4_ROOT = "outputs/model329_fullstats_width4000_uniform_group_oof"
HRNET_ROOT = "outputs/model382_seed_count_control/seed2"
MODEL382_SEQUENCE = "experiments/results/model392_sequence_closure_lock.json"
MODEL382_RUNNER = "run_model382_seed_count_control_p100.sh"
MODEL329_RUNNER = "run_model329_hybrid_width4000_uniform_group_oof.sh"
MODEL364_RESULT = "experiments/results/model364_model361_entropy_into_360.json"
CLASS_NAMES = ("Al", "Al3Ni", "Primary_Si", "Eutectic_Si")
PALETTE = np.asarray(
    ((153, 127, 76), (25, 76, 153), (76, 178, 76), (204, 204, 204)),
    dtype=np.uint8,
)
NUM_CLASSES = 4
EXPECTED_VALID_IMAGES = 20
INFERENCE_CONFIG = {
    "family": "Model382 seed2, four fixed fold checkpoints",
    "folds": [1, 2, 3, 4],
    "patch_size": 512,
    "input_scale": 2.0,
    "target_width": 0,
    "overlap": 170,
    "tta": "flip4",
    "batch_size": 2,
    "tile_stitching": "Gaussian weighted probabilities at original resolution",
    "fold_weights": [0.25, 0.25, 0.25, 0.25],
    "blend": "0.5 * Model329 B4 four-fold valid probabilities + 0.5 * Model382 seed2 four-fold valid probabilities",
    "argmax_after_probability_averaging": True,
}
APPROVED_PROTOCOL_STATUS = "Status: APPROVED_FOR_ONE_SHOT_VALID_EVALUATION"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: dict[str, Any]) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def write_exclusive(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


def save_npz_exclusive(path: Path, **arrays: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return sha256(path)


def valid_pairs(root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    image_dir = root / VALID_IMAGES
    mask_dir = root / VALID_MASKS
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError("official-valid image/mask directories are incomplete")
    images = {p.name.removesuffix("_image.png"): p
              for p in sorted(image_dir.glob("*_image.png"))}
    masks = {p.name.removesuffix("_mask.png"): p
             for p in sorted(mask_dir.glob("*_mask.png"))}
    if len(images) != EXPECTED_VALID_IMAGES or set(images) != set(masks):
        raise ValueError("expected exactly 20 paired official-valid images and masks")
    return images, masks


def _protocol_reviewed(root: Path) -> None:
    text = (root / PROTOCOL).read_text(encoding="utf-8")
    statuses = [line.strip() for line in text.splitlines()
                if line.strip().startswith("Status:")]
    if statuses != [APPROVED_PROTOCOL_STATUS]:
        raise PermissionError("Model400 protocol lacks its exact independent-review approval status")


def _verify_model396_gate(root: Path) -> dict[str, Any]:
    model396.verify_lock(root)
    result = json.loads((root / model396.RESULT).read_text(encoding="utf-8"))
    if (result.get("experiment") != "Model396 fixed architecture-diverse OOF blend"
            or result.get("status") != "SCREEN_PASS"
            or result.get("official_valid_used") is not False
            or result.get("test_used") is not False
            or result.get("lock_sha256") != sha256(root / model396.LOCK)
            or result.get("pre_registered_gate", {}).get("pass") is not True):
        raise ValueError("Model396 train-OOF screen is not a valid frozen basis")
    checks = result["pre_registered_gate"].get("checks", {}).get("D2", {})
    if not checks or not all(checks.values()):
        raise ValueError("Model396 D2 did not pass every pre-registered screen check")
    return result


def _verify_b4_args(root: Path) -> None:
    base = root / B4_ROOT
    for fold in range(1, 5):
        args_path = base / f"fold{fold}/args.json"
        args = json.loads(args_path.read_text(encoding="utf-8"))
        if (any(args.get(key) != value for key, value in model396.B4_EXPECTED_ARGS.items())
                or not str(args.get("data", "")).endswith("/data_model322_group_cv")
                or args.get("train_splits") != f"train_fold{fold}"):
            raise ValueError(f"Model329 B4 fold{fold} recipe mismatch")


def _verify_hrnet_seed2(root: Path) -> None:
    sequence = json.loads((root / MODEL382_SEQUENCE).read_text(encoding="utf-8"))
    refs = sequence.get("model382_same_seed_fold_references", {})
    for fold in range(1, 5):
        run = root / HRNET_ROOT / f"fold{fold}"
        manifest_path = root / HRNET_ROOT / f"fold{fold}_oof/model382_manifest.json"
        args = json.loads((run / "args.json").read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = model396.MODEL382_EXPECTED_ARGS
        if (any(args.get(key) != value for key, value in expected.items())
                or args.get("seed") != 2):
            raise ValueError(f"Model382 seed2 fold{fold} args mismatch")
        if (manifest.get("model") != 382 or manifest.get("seed") != 2
                or manifest.get("fold") != fold or manifest.get("epochs") != 70
                or manifest.get("metric") != "image-wise 4-class mIoU; empty-union=0"
                or manifest.get("checkpoint_sha256") != sha256(run / "last.pt")):
            raise ValueError(f"Model382 seed2 fold{fold} manifest/checkpoint mismatch")
        ref = refs.get(f"seed2/fold{fold}", {}).get("checkpoint", {})
        if ref.get("sha256") != sha256(run / "last.pt"):
            raise ValueError(f"Model382 sequence closure differs for seed2/fold{fold}")


def _check_probs(probabilities: np.ndarray, shape: tuple[int, int], label: str) -> None:
    if probabilities.shape != (NUM_CLASSES, *shape) or not np.isfinite(probabilities).all():
        raise ValueError(f"invalid probability shape/values for {label}: {probabilities.shape}")
    if probabilities.min() < -1e-4 or probabilities.max() > 1.0001:
        raise ValueError(f"probability outside [0,1] for {label}")
    if np.max(np.abs(probabilities.sum(axis=0, dtype=np.float32) - 1.0)) > 0.03:
        raise ValueError(f"probabilities are not normalized for {label}")


def _verify_b4_valid_probs(root: Path, image_paths: dict[str, Path]) -> Path:
    archive_path = root / B4_ROOT / "valid_ensemble/probabilities.npz"
    score_path = root / B4_ROOT / "valid_ensemble/scores.json"
    scores = json.loads(score_path.read_text(encoding="utf-8"))
    if set(scores.get("images", {})) != set(image_paths):
        raise ValueError("B4 score receipt IDs differ from the fixed valid split")
    with np.load(archive_path, allow_pickle=False) as archive:
        if set(archive.files) != set(image_paths):
            raise ValueError("existing B4 valid ensemble IDs differ from the fixed valid split")
        for image_id, image_path in image_paths.items():
            with Image.open(image_path) as image:
                shape = (image.height, image.width)
            _check_probs(np.asarray(archive[image_id]), shape, f"B4/{image_id}")
    return archive_path


def _verify_model364_reference(root: Path) -> float:
    record = json.loads((root / MODEL364_RESULT).read_text(encoding="utf-8"))
    if (record.get("model") != 364
            or record.get("metric") != "20-image mean of four class IoUs; empty union=0"
            or not isinstance(record.get("raw_best_valid_image_mean_miou_empty0"), (int, float))):
        raise ValueError("recorded Model364 reference does not match the fixed empty=0 metric")
    score = float(record["raw_best_valid_image_mean_miou_empty0"]) * 100.0
    if not np.isfinite(score) or abs(score - 79.94297272293409) > 1e-9:
        raise ValueError("recorded Model364 reference score differs from the frozen comparison")
    return score


def input_paths(root: Path) -> list[Path]:
    images, masks = valid_pairs(root)
    b4 = root / B4_ROOT
    model336 = root / "model336_hrnet_best"
    paths = [
        root / PROTOCOL,
        root / SCRIPT,
        root / SMOKE,
        root / model396.PROTOCOL,
        root / model396.SCRIPT,
        root / model396.LOCK,
        root / model396.RESULT,
        root / model396.SEQUENCE_CLOSURE,
        root / MODEL382_SEQUENCE,
        root / MODEL382_RUNNER,
        root / MODEL329_RUNNER,
        root / MODEL364_RESULT,
        b4 / "valid_ensemble/probabilities.npz",
        b4 / "valid_ensemble/scores.json",
        b4 / "valid_ensemble.log",
        root / "model326_hybrid_fullstats/KoMaP/train.py",
        root / "model326_hybrid_fullstats/KoMaP/imagewise_lovasz.py",
        root / "model326_hybrid_fullstats/KoMaP/predict.py",
        root / "model326_hybrid_fullstats/KoMaP/common.py",
        root / "model326_hybrid_fullstats/KoMaP/prepare_group_cv.py",
        model336 / "predict.py",
        model336 / "train.py",
        model336 / "evaluate_image_mean.py",
        model336 / "komap_seg/checkpoint.py",
        model336 / "komap_seg/dataset.py",
        model336 / "komap_seg/inference.py",
        model336 / "komap_seg/losses.py",
        model336 / "komap_seg/metrics.py",
        model336 / "komap_seg/model.py",
        model336 / "komap_seg/palette.py",
    ]
    for fold in range(1, 5):
        paths.extend((
            b4 / f"fold{fold}/args.json",
            b4 / f"fold{fold}/last.pt",
            root / HRNET_ROOT / f"fold{fold}/args.json",
            root / HRNET_ROOT / f"fold{fold}/last.pt",
            root / HRNET_ROOT / f"fold{fold}_oof/model382_manifest.json",
            root / HRNET_ROOT / f"fold{fold}_oof/valid_probabilities.npz",
        ))
    paths.extend(images.values())
    paths.extend(masks.values())
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return sorted({path.resolve() for path in paths})


def _runtime_record() -> dict[str, str]:
    names = ("torch", "numpy", "Pillow", "albumentations", "segmentation-models-pytorch", "timm")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    receipt = {"python": platform.python_version(), **versions}
    try:
        import torch
        receipt["torch_cuda"] = str(torch.version.cuda)
        receipt["cudnn"] = str(torch.backends.cudnn.version())
    except ImportError:
        receipt["torch_cuda"] = "missing"
        receipt["cudnn"] = "missing"
    return receipt


def create_lock(root: Path) -> dict[str, Any]:
    lock_path = root / LOCK
    if lock_path.exists() or (root / RESULT).exists() or (root / CLAIM).exists():
        raise FileExistsError("refusing to overwrite/repeat the Model400 one-shot stage")
    if (root / INFER_ROOT).exists():
        raise FileExistsError("Model400 inference output already exists; refusing reuse")
    _protocol_reviewed(root)
    oof_result = _verify_model396_gate(root)
    _verify_b4_args(root)
    _verify_hrnet_seed2(root)
    model364_score_pct = _verify_model364_reference(root)
    images, _ = valid_pairs(root)
    b4_probs = _verify_b4_valid_probs(root, images)
    files = {
        path.resolve().relative_to(root.resolve()).as_posix(): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in input_paths(root)
    }
    record = {
        "experiment": "Model400 Model396-D2 four-fold valid ensemble",
        "revision": 1,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "metric": "image-wise 4-class mIoU; empty-union=0",
        "class_order": list(CLASS_NAMES),
        "valid_images": EXPECTED_VALID_IMAGES,
        "configuration": INFERENCE_CONFIG,
        "valid_inference_different_from_oof_estimand": True,
        "selection_conditional": True,
        "test_inputs_excluded": True,
        "model396_oof_result_sha256": sha256(root / model396.RESULT),
        "reference_model364_valid_percent": model364_score_pct,
        "b4_valid_probability_path": b4_probs.resolve().relative_to(root.resolve()).as_posix(),
        "locked_files": files,
        "runtime": _runtime_record(),
    }
    record["lock_sha256"] = canonical_sha256(record)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    write_exclusive(lock_path, record)
    return record


def verify_lock(root: Path) -> dict[str, Any]:
    _protocol_reviewed(root)
    path = root / LOCK
    record = json.loads(path.read_text(encoding="utf-8"))
    digest = record.get("lock_sha256")
    body = {key: value for key, value in record.items() if key != "lock_sha256"}
    if digest != canonical_sha256(body):
        raise ValueError("Model400 lock content hash is invalid")
    if record.get("experiment") != "Model400 Model396-D2 four-fold valid ensemble":
        raise ValueError("unexpected Model400 lock")
    if record.get("configuration") != INFERENCE_CONFIG:
        raise ValueError("Model400 inference configuration differs from the frozen protocol")
    if record.get("runtime") != _runtime_record():
        raise ValueError("Model400 runtime/dependency receipt changed after lock")
    expected_paths = {
        p.resolve().relative_to(root.resolve()).as_posix() for p in input_paths(root)
    }
    locked = record.get("locked_files", {})
    if set(locked) != expected_paths:
        raise ValueError("Model400 locked input inventory changed")
    for relative, metadata in locked.items():
        source = root / relative
        if source.stat().st_size != metadata["size_bytes"] or sha256(source) != metadata["sha256"]:
            raise ValueError(f"Model400 input changed after lock: {relative}")
    _verify_model396_gate(root)
    _verify_b4_args(root)
    _verify_hrnet_seed2(root)
    _verify_model364_reference(root)
    images, _ = valid_pairs(root)
    _verify_b4_valid_probs(root, images)
    return record


def _claim(path: Path, lock_hash: str, stage: str) -> str:
    return write_exclusive(path, {
        "experiment": "Model400 Model396-D2 four-fold valid ensemble",
        "stage": stage,
        "claimed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "lock_sha256": lock_hash,
        "policy": "one-shot; claim retained after interruption; no overwrite or rerun",
    })


def infer_hrnet_seed2(root: Path, device_name: str) -> dict[str, Any]:
    lock = verify_lock(root)
    output_root = root / INFER_ROOT
    output_root.mkdir(parents=True, exist_ok=False)
    inference_claim_hash = _claim(
        output_root / "inference.claim", lock["lock_sha256"], "inference"
    )

    model_dir = root / "model336_hrnet_best"
    sys.path.insert(0, str(model_dir))
    import torch
    from komap_seg.checkpoint import load_model
    from komap_seg.dataset import KomapFullDataset
    from komap_seg.inference import predict_probs_tta

    device = torch.device(device_name)
    image_dir = root / VALID_IMAGES
    dataset = KomapFullDataset(image_dir, masks_dir=None)
    expected_ids = sorted(valid_pairs(root)[0])
    actual_ids = sorted(sample["stem"] for sample in dataset.pairs)
    if actual_ids != expected_ids or len(dataset) != EXPECTED_VALID_IMAGES:
        raise ValueError("image-only inference dataset differs from the locked valid IDs")
    sample_indices = {sample["stem"]: i for i, sample in enumerate(dataset.pairs)}
    models = []
    checkpoint_hashes = {}
    for fold in range(1, 5):
        checkpoint = root / HRNET_ROOT / f"fold{fold}/last.pt"
        models.append(load_model(checkpoint, device))
        checkpoint_hashes[str(fold)] = sha256(checkpoint)

    mean_dir = root / HRNET_MEAN_DIR
    mean_dir.mkdir(parents=True, exist_ok=False)
    mean_map_receipts = []
    image_paths, _ = valid_pairs(root)
    for index, image_id in enumerate(expected_ids, start=1):
        sample_index = sample_indices[image_id]
        sample = dataset[sample_index]
        with Image.open(image_paths[image_id]) as image:
            shape = (image.height, image.width)
        probability_sum = np.zeros((NUM_CLASSES, *shape), dtype=np.float32)
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
            _check_probs(probs, shape, f"HRNet seed2 fold{fold}/{image_id}")
            probability_sum += probs
            del probs
        mean = probability_sum / 4.0
        _check_probs(mean, shape, f"HRNet seed2 four-fold mean/{image_id}")
        map_path = mean_dir / f"{image_id}.npz"
        map_sha = save_npz_exclusive(map_path, **{image_id: mean.astype(np.float16)})
        mean_map_receipts.append({
            "image_id": image_id,
            "path": map_path.resolve().relative_to(root.resolve()).as_posix(),
            "sha256": map_sha,
            "shape": list(shape),
        })
        print(json.dumps({"stage": "model400_inference", "image": index,
                          "images": len(dataset), "hrnet_folds": 4}, flush=True))
    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()
    manifest = {
        "experiment": "Model400 Model396-D2 four-fold valid ensemble",
        "status": "INFERENCE_COMPLETE_NOT_SCORED",
        "lock_sha256": lock["lock_sha256"],
        "inference_claim_sha256": inference_claim_hash,
        "image_ids": expected_ids,
        "hrnet_seed2_fold_checkpoint_sha256": checkpoint_hashes,
        "hrnet_seed2_mean_probability_maps": mean_map_receipts,
        "b4_valid_probabilities_sha256": sha256(root / B4_ROOT / "valid_ensemble/probabilities.npz"),
        "configuration": INFERENCE_CONFIG,
        "official_valid_mask_pixels_decoded_during_inference": False,
        "official_valid_score_computed": False,
        "test_inputs_excluded": True,
    }
    write_exclusive(root / INFER_MANIFEST, manifest)
    print(json.dumps({"status": manifest["status"], "manifest": INFER_MANIFEST,
                      "images": len(expected_ids), "folds": 4}, indent=2), flush=True)
    return manifest


def mask_to_ids(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    ids = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        ids[np.all(rgb == color, axis=2)] = class_id
    if np.any(ids == 255):
        raise ValueError(f"unknown label color in {path}")
    return ids


def per_class_iou(probabilities: np.ndarray, truth: np.ndarray) -> np.ndarray:
    if probabilities.shape != (NUM_CLASSES, *truth.shape):
        raise ValueError("prediction and target dimensions differ")
    prediction = probabilities.argmax(axis=0)
    values = np.zeros(NUM_CLASSES, dtype=np.float64)
    for class_id in range(NUM_CLASSES):
        gt = truth == class_id
        pred = prediction == class_id
        union = np.logical_or(gt, pred).sum()
        intersection = np.logical_and(gt, pred).sum()
        values[class_id] = intersection / union if union else 0.0
    return values


def score_once(root: Path) -> dict[str, Any]:
    lock = verify_lock(root)
    manifest_path = root / INFER_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inference_claim_path = root / INFER_CLAIM
    inference_claim = json.loads(inference_claim_path.read_text(encoding="utf-8"))
    b4_path = root / B4_ROOT / "valid_ensemble/probabilities.npz"
    if (manifest.get("status") != "INFERENCE_COMPLETE_NOT_SCORED"
            or manifest.get("experiment") != "Model400 Model396-D2 four-fold valid ensemble"
            or manifest.get("lock_sha256") != lock["lock_sha256"]
            or manifest.get("image_ids") != sorted(valid_pairs(root)[0])
            or manifest.get("b4_valid_probabilities_sha256") != sha256(b4_path)
            or manifest.get("inference_claim_sha256") != sha256(inference_claim_path)
            or inference_claim.get("stage") != "inference"
            or inference_claim.get("lock_sha256") != lock["lock_sha256"]
            or manifest.get("configuration") != INFERENCE_CONFIG
            or manifest.get("official_valid_mask_pixels_decoded_during_inference") is not False
            or manifest.get("official_valid_score_computed") is not False
            or manifest.get("test_inputs_excluded") is not True):
        raise ValueError("Model400 inference manifest/hash verification failed")
    result_path = root / RESULT
    claim_path = root / CLAIM
    if result_path.exists() or claim_path.exists():
        raise FileExistsError("Model400 valid score already claimed or written")
    images, masks = valid_pairs(root)
    expected_ids = sorted(images)
    map_receipts = manifest.get("hrnet_seed2_mean_probability_maps", [])
    mean_maps = {item["image_id"]: item for item in map_receipts}
    if (len(map_receipts) != EXPECTED_VALID_IMAGES
            or set(mean_maps) != set(expected_ids)
            or len(mean_maps) != EXPECTED_VALID_IMAGES):
        raise ValueError("HRNet inference manifest does not contain exactly 20 valid maps")
    expected_checkpoint_hashes = {
        str(fold): sha256(root / HRNET_ROOT / f"fold{fold}/last.pt")
        for fold in range(1, 5)
    }
    if manifest.get("hrnet_seed2_fold_checkpoint_sha256") != expected_checkpoint_hashes:
        raise ValueError("HRNet checkpoint receipts differ from the locked fold checkpoints")
    for image_id, item in mean_maps.items():
        map_path = (root / item["path"]).resolve()
        expected_path = (root / HRNET_MEAN_DIR / f"{image_id}.npz").resolve()
        if map_path != expected_path or item.get("shape") is None:
            raise ValueError(f"HRNet probability map path/shape receipt mismatch: {image_id}")
        if not map_path.is_file() or sha256(map_path) != item["sha256"]:
            raise ValueError(f"HRNet probability map hash mismatch: {image_id}")
    with np.load(b4_path, allow_pickle=False) as b4:
        if set(b4.files) != set(expected_ids):
            raise ValueError("Model400 B4 probability IDs differ from locked valid split")
        for image_id in expected_ids:
            with Image.open(images[image_id]) as image:
                shape = (image.height, image.width)
            _check_probs(np.asarray(b4[image_id]), shape, f"B4/{image_id}")
            with np.load(root / mean_maps[image_id]["path"], allow_pickle=False) as hrnet:
                if set(hrnet.files) != {image_id}:
                    raise ValueError(f"HRNet probability archive has wrong ID: {image_id}")
                probs = np.asarray(hrnet[image_id], dtype=np.float32)
            if tuple(mean_maps[image_id]["shape"]) != shape:
                raise ValueError(f"HRNet probability shape receipt differs from image: {image_id}")
            _check_probs(probs, shape, f"HRNet/{image_id}")
    claim_hash = _claim(claim_path, lock["lock_sha256"], "official_valid_scoring")
    rows = []
    class_scores = []
    with np.load(b4_path, allow_pickle=False) as b4:
        for image_id in expected_ids:
            truth = mask_to_ids(masks[image_id])
            with Image.open(images[image_id]) as image:
                shape = (image.height, image.width)
            if tuple(mean_maps[image_id]["shape"]) != shape:
                raise ValueError(f"HRNet probability shape receipt differs from image: {image_id}")
            p_b4 = np.asarray(b4[image_id], dtype=np.float32)
            with np.load(root / mean_maps[image_id]["path"], allow_pickle=False) as hrnet:
                if set(hrnet.files) != {image_id}:
                    raise ValueError(f"HRNet probability archive has wrong ID: {image_id}")
                p_hrnet = np.asarray(hrnet[image_id], dtype=np.float32)
            _check_probs(p_b4, shape, f"B4/{image_id}")
            _check_probs(p_hrnet, shape, f"HRNet/{image_id}")
            blended = 0.5 * p_b4 + 0.5 * p_hrnet
            _check_probs(blended, shape, f"Model400/{image_id}")
            class_iou = per_class_iou(blended, truth)
            group = model396.GROUP_RE.match(image_id)
            if group is None:
                raise ValueError(f"cannot parse specimen group from {image_id}")
            score = float(class_iou.mean())
            class_scores.append(class_iou)
            rows.append({
                "image_id": image_id,
                "specimen_group": group.group(1),
                "class_iou_empty0": class_iou.tolist(),
                "image_miou_empty0": score,
            })
            print(json.dumps({"stage": "model400_scored", "image": len(rows),
                              "images": EXPECTED_VALID_IMAGES}, flush=True))
    per_image = np.asarray(class_scores, dtype=np.float64)
    model364_score_pct = _verify_model364_reference(root)
    result = {
        "experiment": "Model400 Model396-D2 four-fold valid ensemble",
        "status": "DESCRIPTIVE_SELECTION_CONDITIONAL_VALID_RESULT",
        "metric": "image-wise 4-class mIoU; empty-union=0",
        "class_order": list(CLASS_NAMES),
        "images": len(rows),
        "image_mean_miou_percent": float(per_image.mean(axis=1).mean() * 100.0),
        "class_image_mean_iou_percent": {
            name: float(per_image[:, index].mean() * 100.0)
            for index, name in enumerate(CLASS_NAMES)
        },
        "difference_from_recorded_model364_pp": (
            float(per_image.mean(axis=1).mean() * 100.0) - model364_score_pct
        ),
        "reference_model364_valid_percent": model364_score_pct,
        "fixed_probability_weights": {"Model329_B4_four_fold": 0.5,
                                      "Model382_seed2_four_fold": 0.5},
        "inference_config": INFERENCE_CONFIG,
        "valid_inference_different_from_oof_estimand": True,
        "official_valid_used": True,
        "test_used": False,
        "independent_confirmation": False,
        "interpretation": "one additional adaptively reused 20-image validation look; descriptive only",
        "lock_sha256": lock["lock_sha256"],
        "inference_manifest_sha256": sha256(manifest_path),
        "claim_sha256": claim_hash,
        "per_image": rows,
    }
    write_exclusive(result_path, result)
    return result


def self_test() -> None:
    truth = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
    perfect = np.zeros((4, 2, 2), dtype=np.float32)
    perfect[0] = truth == 0
    perfect[1] = truth == 1
    assert np.array_equal(per_class_iou(perfect, truth), [1.0, 1.0, 0.0, 0.0])
    _check_probs(perfect, truth.shape, "synthetic perfect map")
    mean = 0.5 * perfect + 0.5 * perfect
    assert np.array_equal(per_class_iou(mean, truth), [1.0, 1.0, 0.0, 0.0])
    bad = perfect.copy()
    bad[0, 0, 0] = 0.2
    try:
        _check_probs(bad, truth.shape, "synthetic unnormalized map")
    except ValueError:
        pass
    else:
        raise AssertionError("unnormalized map was accepted")
    print("Model400 synthetic metric/probability smoke passed; no dataset accessed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--self-test", action="store_true")
    actions.add_argument("--create-lock", action="store_true")
    actions.add_argument("--verify-lock", action="store_true")
    actions.add_argument("--infer", action="store_true")
    actions.add_argument("--score", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.self_test:
        self_test()
    elif args.create_lock:
        lock = create_lock(root)
        print(json.dumps({"status": "LOCKED", "lock_sha256": lock["lock_sha256"],
                          "files": len(lock["locked_files"])}, indent=2))
    elif args.verify_lock:
        lock = verify_lock(root)
        print(json.dumps({"status": "LOCK_VERIFIED", "lock_sha256": lock["lock_sha256"],
                          "files": len(lock["locked_files"])}, indent=2))
    elif args.infer:
        result = infer_hrnet_seed2(root, args.device)
        print(json.dumps({"status": result["status"], "manifest": INFER_MANIFEST}, indent=2))
    else:
        result = score_once(root)
        print(json.dumps({"status": result["status"],
                          "image_mean_miou_percent": result["image_mean_miou_percent"],
                          "difference_from_recorded_model364_pp": result["difference_from_recorded_model364_pp"]}, indent=2))


if __name__ == "__main__":
    main()
