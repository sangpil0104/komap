#!/usr/bin/env python3
"""Locked one-shot valid inference/scoring for the frozen Model397 blend."""

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

import model397_fixed_b4_model392_blend as m397
import model400_model396_d2_valid_once as m400


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "experiments/model401_model397_blend_valid_once.md"
SCRIPT = "experiments/model401_model397_blend_valid_once.py"
SMOKE = "experiments/model401_model397_blend_valid_once_smoke.py"
LOCK = "experiments/results/model401_model397_blend_valid_lock.json"
INFER_ROOT = "outputs/model401_model397_blend_valid_once"
INFER_CLAIM = INFER_ROOT + "/inference.claim"
INFER_MANIFEST = INFER_ROOT + "/inference_manifest.json"
INFER_ROOT_SEED1 = INFER_ROOT + "/hrnet_seed1_mean_valid"
INFER_ROOT_SEED2 = INFER_ROOT + "/hrnet_seed2_mean_valid"
RESULT = "experiments/results/model401_model397_blend_valid_once.json"
SCORE_CLAIM = RESULT + ".claim"
VALID_IMAGES = m400.VALID_IMAGES
VALID_MASKS = m400.VALID_MASKS
B4_ARCHIVE = m400.B4_ROOT + "/valid_ensemble/probabilities.npz"
MODEL392_ROOT = m397.MODEL392_ROOT
MODEL392_LOCK = m397.MODEL392_LOCK
MODEL364_SCORE = 79.94297272293409
EXPECTED_IDS = 20
CONFIG = {
    "candidate": "0.50*Model329_B4_4fold + 0.25*Model392_seed1_4fold + 0.25*Model392_seed2_4fold",
    "family_weights": {"Model329_B4": 0.5, "Model392_seed1": 0.25, "Model392_seed2": 0.25},
    "folds_per_seed": [1, 2, 3, 4],
    "per_checkpoint_preprocessing": "read and enforce saved preprocessing_config",
    "base_input_scale": 2.0,
    "patch_size": 512,
    "base_overlap": 170,
    "tta": "flip4",
    "stitching": "Gaussian probability stitching at native resolution",
    "batch_size": 2,
    "argmax_after_probability_average": True,
    "metric": "image-wise 4-class mIoU; empty-union=0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_exclusive(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def runtime_record() -> dict[str, str]:
    names = ("numpy", "Pillow", "torch", "segmentation-models-pytorch", "timm", "albumentations")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    return {"python": platform.python_version(), "platform": platform.platform(), **versions}


def valid_pairs(root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    images, masks = m400.valid_pairs(root)
    if len(images) != EXPECTED_IDS:
        raise ValueError("Model401 requires exactly 20 paired valid images")
    return images, masks


def verify_model397_basis(root: Path) -> dict[str, Any]:
    m397.verify_lock(root)
    result_path = root / m397.RESULT
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (result.get("status") != "OOF_SCREEN_PASS"
            or result.get("metric") != CONFIG["metric"]
            or result.get("valid_stage_allowed") is not False
            or result.get("official_valid_used") is not False
            or result.get("test_used") is not False
            or not m397.gate(result.get("comparisons", {})).get("pass")):
        raise ValueError("frozen Model397 OOF candidate did not pass its own fixed screen")
    return result


def verify_inputs(root: Path) -> dict[str, Any]:
    result = verify_model397_basis(root)
    # This validates the existing B4 archive/receipts and exact Model364 metric.
    m400._verify_b4_args(root)
    images, _ = valid_pairs(root)
    m400._verify_b4_valid_probs(root, images)
    if abs(m400._verify_model364_reference(root) - MODEL364_SCORE) > 1e-9:
        raise ValueError("Model364 comparison differs from frozen reference")
    model392_hash = sha256(root / MODEL392_LOCK)
    for fold in range(1, 5):
        expected_ids = set(m397.model396.mask_files(root)[fold])
        for seed in (1, 2):
            run_dir = root / MODEL392_ROOT / f"seed{seed}/fold{fold}"
            oof_dir = root / MODEL392_ROOT / f"seed{seed}/fold{fold}_oof"
            archive = m397.model392.verify_candidate_manifest(
                root, run_dir, oof_dir, seed, fold,
                root / MODEL392_LOCK, model392_hash,
            )
            if not archive.is_file():
                raise FileNotFoundError(archive)
            checkpoint = run_dir / "last.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
    return {"model397_oof_status": result["status"],
            "model397_oof_miou_percent": result["arms"]["C"]["image_mean_miou_percent"],
            "valid_images": len(images)}


def input_paths(root: Path) -> list[Path]:
    images, masks = valid_pairs(root)
    paths = [
        root / PROTOCOL, root / SCRIPT, root / SMOKE,
        root / m397.PROTOCOL, root / m397.SCRIPT, root / m397.SMOKE,
        root / m397.model396.PROTOCOL, root / m397.model396.SCRIPT,
        root / "experiments/model392_widthnorm_preregistration.md",
        root / "experiments/model392_widthnorm_oof.py",
        root / m397.LOCK, root / m397.RESULT,
        root / m397.MODEL392_LOCK, root / m397.MODEL392_GATE,
        root / m397.SEQUENCE_CLOSURE,
        root / m400.PROTOCOL, root / m400.SCRIPT, root / m400.LOCK,
        root / m400.RESULT, root / m400.MODEL364_RESULT,
        root / B4_ARCHIVE,
        root / m400.B4_ROOT / "valid_ensemble/scores.json",
        root / "model336_hrnet_widthnorm_candidate/komap_seg/checkpoint.py",
        root / "model336_hrnet_widthnorm_candidate/komap_seg/dataset.py",
        root / "model336_hrnet_widthnorm_candidate/komap_seg/inference.py",
        root / "model336_hrnet_widthnorm_candidate/komap_seg/model.py",
        root / "model336_hrnet_widthnorm_candidate/komap_seg/palette.py",
    ]
    for fold in range(1, 5):
        paths.extend((
            root / m400.B4_ROOT / f"fold{fold}/args.json",
            root / m400.B4_ROOT / f"fold{fold}/last.pt",
        ))
        for seed in (1, 2):
            run = root / MODEL392_ROOT / f"seed{seed}/fold{fold}"
            oof = root / MODEL392_ROOT / f"seed{seed}/fold{fold}_oof"
            paths.extend((run / "last.pt", run / "args.json", run / "model392_run_receipt.json",
                          oof / "model392_manifest.json"))
    paths.extend(images.values())
    paths.extend(masks.values())
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing locked inputs: {missing[:5]}")
    return sorted({path.resolve() for path in paths})


def create_lock(root: Path) -> dict[str, Any]:
    lock_path = root / LOCK
    if lock_path.exists() or (root / RESULT).exists() or (root / SCORE_CLAIM).exists() or (root / INFER_ROOT).exists():
        raise FileExistsError("Model401 has already been locked, claimed, or started")
    basis = verify_inputs(root)
    files = {
        path.resolve().relative_to(root.resolve()).as_posix(): {
            "size_bytes": path.stat().st_size, "sha256": sha256(path)
        }
        for path in input_paths(root)
    }
    record = {
        "experiment": "Model401 one-shot Model397 blend valid evaluation",
        "revision": 1,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "metric": CONFIG["metric"],
        "config": CONFIG,
        "valid_images": EXPECTED_IDS,
        "model397_oof_status": basis["model397_oof_status"],
        "model397_oof_miou_percent": basis["model397_oof_miou_percent"],
        "old_model397_valid_stage_gate_superseded_by_this_lock": True,
        "selection_conditional": True,
        "test_inputs_excluded": True,
        "locked_files": files,
        "runtime": runtime_record(),
    }
    record["lock_sha256"] = canonical_sha256(record)
    write_exclusive(lock_path, record)
    return record


def verify_lock(root: Path) -> dict[str, Any]:
    record = json.loads((root / LOCK).read_text(encoding="utf-8"))
    body = {key: value for key, value in record.items() if key != "lock_sha256"}
    if record.get("lock_sha256") != canonical_sha256(body):
        raise ValueError("Model401 lock hash is invalid")
    if (record.get("experiment") != "Model401 one-shot Model397 blend valid evaluation"
            or record.get("config") != CONFIG or record.get("runtime") != runtime_record()
            or record.get("test_inputs_excluded") is not True
            or record.get("old_model397_valid_stage_gate_superseded_by_this_lock") is not True):
        raise ValueError("Model401 lock content/config/runtime mismatch")
    expected = {p.resolve().relative_to(root.resolve()).as_posix() for p in input_paths(root)}
    locked = record.get("locked_files", {})
    if set(locked) != expected:
        raise ValueError("Model401 locked input inventory changed")
    for relative, meta in locked.items():
        path = root / relative
        if path.stat().st_size != meta["size_bytes"] or sha256(path) != meta["sha256"]:
            raise ValueError(f"Model401 locked input changed: {relative}")
    return record


def claim(path: Path, lock_hash: str, stage: str) -> str:
    return write_exclusive(path, {
        "experiment": "Model401 one-shot Model397 blend valid evaluation",
        "stage": stage,
        "claimed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "lock_sha256": lock_hash,
        "policy": "exclusive one-shot claim; interruption does not authorize silent rerun",
    })


def _load_archive_map(path: Path, image_id: str, shape: tuple[int, int]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {image_id}:
            raise ValueError(f"probability archive key mismatch: {path}")
        probs = np.asarray(archive[image_id], dtype=np.float32)
    m400._check_probs(probs, shape, f"{path}/{image_id}")
    return probs


def infer(root: Path, device_name: str) -> dict[str, Any]:
    lock = verify_lock(root)
    output_root = root / INFER_ROOT
    output_root.mkdir(parents=True, exist_ok=False)
    inference_claim_hash = claim(output_root / "inference.claim", lock["lock_sha256"], "inference")

    model_dir = root / "model336_hrnet_widthnorm_candidate"
    sys.path.insert(0, str(model_dir))
    import torch
    from komap_seg.checkpoint import load_model_and_preprocessing, resolve_preprocessing
    from komap_seg.dataset import KomapFullDataset, effective_input_scale, scaled_overlap
    from komap_seg.inference import predict_probs_tta

    device = torch.device(device_name)
    images, _ = valid_pairs(root)
    dataset = KomapFullDataset(root / VALID_IMAGES, masks_dir=None)
    ids = sorted(images)
    if sorted(item["stem"] for item in dataset.pairs) != ids:
        raise ValueError("image-only inference IDs differ from locked valid images")
    checkpoints = []
    checkpoint_hashes: dict[str, dict[str, str]] = {"1": {}, "2": {}}
    for seed in (1, 2):
        for fold in range(1, 5):
            path = root / MODEL392_ROOT / f"seed{seed}/fold{fold}/last.pt"
            model, saved = load_model_and_preprocessing(path, device)
            preprocessing = resolve_preprocessing(saved)
            if (preprocessing["patch_size"] != 512 or preprocessing["input_scale"] != 2.0
                    or preprocessing["target_width"] <= 0 or not preprocessing["normalize_width"]):
                raise ValueError(f"Model392 preprocessing metadata mismatch: {path}")
            checkpoints.append((seed, fold, model, preprocessing))
            checkpoint_hashes[str(seed)][str(fold)] = sha256(path)

    seed_dirs = {
        1: root / INFER_ROOT_SEED1,
        2: root / INFER_ROOT_SEED2,
    }
    for path in seed_dirs.values():
        path.mkdir(parents=True, exist_ok=False)
    manifests: dict[int, list[dict[str, Any]]] = {1: [], 2: []}
    for number, image_id in enumerate(ids, start=1):
        sample = dataset[ next(i for i, pair in enumerate(dataset.pairs) if pair["stem"] == image_id) ]
        shape = (sample["height"], sample["width"])
        seed_sums = {1: np.zeros((4, *shape), dtype=np.float32),
                     2: np.zeros((4, *shape), dtype=np.float32)}
        for seed, fold, model, preprocessing in checkpoints:
            scale = effective_input_scale(
                preprocessing["input_scale"], preprocessing["target_width"], sample["width"]
            )
            overlap = scaled_overlap(512, 2.0, 170, scale)
            probs_t = predict_probs_tta(
                model, sample["image"], patch_size=512, overlap=overlap,
                batch_size=2, tta=True, input_scale=scale, device=device,
                tta_mode="flip4",
            )
            probs = probs_t.numpy().astype(np.float32)
            m400._check_probs(probs, shape, f"seed{seed}/fold{fold}/{image_id}")
            seed_sums[seed] += probs * 0.25
            del probs_t, probs
        for seed in (1, 2):
            mean = seed_sums[seed]
            m400._check_probs(mean, shape, f"seed{seed}/four-fold/{image_id}")
            path = seed_dirs[seed] / f"{image_id}.npz"
            map_sha = m400.save_npz_exclusive(path, **{image_id: mean.astype(np.float16)})
            manifests[seed].append({
                "image_id": image_id,
                "path": path.resolve().relative_to(root.resolve()).as_posix(),
                "sha256": map_sha,
                "shape": list(shape),
            })
        print(json.dumps({"stage": "model401_inference", "image": number,
                          "images": len(ids), "hrnet_checkpoints": 8}, flush=True))
    del checkpoints
    if device.type == "cuda":
        torch.cuda.empty_cache()
    record = {
        "experiment": "Model401 one-shot Model397 blend valid evaluation",
        "status": "INFERENCE_COMPLETE_NOT_SCORED",
        "lock_sha256": lock["lock_sha256"],
        "inference_claim_sha256": inference_claim_hash,
        "image_ids": ids,
        "hrnet_checkpoint_sha256": checkpoint_hashes,
        "hrnet_seed1_mean_maps": manifests[1],
        "hrnet_seed2_mean_maps": manifests[2],
        "b4_valid_archive_sha256": sha256(root / B4_ARCHIVE),
        "config": CONFIG,
        "valid_masks_decoded_during_inference": False,
        "valid_score_computed": False,
        "test_inputs_excluded": True,
    }
    write_exclusive(root / INFER_MANIFEST, record)
    return record


def score(root: Path) -> dict[str, Any]:
    lock = verify_lock(root)
    manifest_path = root / INFER_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inference_claim_path = root / INFER_CLAIM
    inference_claim = json.loads(inference_claim_path.read_text(encoding="utf-8"))
    images, masks = valid_pairs(root)
    ids = sorted(images)
    if (manifest.get("status") != "INFERENCE_COMPLETE_NOT_SCORED"
            or manifest.get("lock_sha256") != lock["lock_sha256"]
            or manifest.get("image_ids") != ids
            or manifest.get("valid_masks_decoded_during_inference") is not False
            or manifest.get("valid_score_computed") is not False
            or manifest.get("test_inputs_excluded") is not True
            or inference_claim.get("lock_sha256") != lock["lock_sha256"]
            or inference_claim.get("stage") != "inference"):
        raise ValueError("Model401 inference manifest/claim mismatch")
    if manifest.get("b4_valid_archive_sha256") != sha256(root / B4_ARCHIVE):
        raise ValueError("B4 valid archive changed after inference")
    maps: dict[int, dict[str, dict[str, Any]]] = {}
    for seed, key in ((1, "hrnet_seed1_mean_maps"), (2, "hrnet_seed2_mean_maps")):
        receipts = manifest.get(key, [])
        lookup = {item["image_id"]: item for item in receipts}
        if len(receipts) != EXPECTED_IDS or set(lookup) != set(ids):
            raise ValueError(f"seed{seed} inference map coverage is incomplete")
        maps[seed] = lookup
        expected_dir = root / (INFER_ROOT_SEED1 if seed == 1 else INFER_ROOT_SEED2)
        for image_id, item in lookup.items():
            path = (root / item["path"]).resolve()
            if path != (expected_dir / f"{image_id}.npz").resolve() or sha256(path) != item["sha256"]:
                raise ValueError(f"seed{seed} map path/hash mismatch for {image_id}")
            _load_archive_map(path, image_id, tuple(item["shape"]))
    for seed in (1, 2):
        expected_ckpts = {str(f): sha256(root / MODEL392_ROOT / f"seed{seed}/fold{f}/last.pt")
                          for f in range(1, 5)}
        if manifest.get("hrnet_checkpoint_sha256", {}).get(str(seed)) != expected_ckpts:
            raise ValueError(f"seed{seed} checkpoint hashes differ from inference receipt")
    with np.load(root / B4_ARCHIVE, allow_pickle=False) as b4:
        if set(b4.files) != set(ids):
            raise ValueError("B4 archive IDs differ from valid split")
        for image_id in ids:
            with Image.open(images[image_id]) as image:
                shape = (image.height, image.width)
            m400._check_probs(np.asarray(b4[image_id], dtype=np.float32), shape, f"B4/{image_id}")
            for seed in (1, 2):
                item = maps[seed][image_id]
                if tuple(item["shape"]) != shape:
                    raise ValueError(f"seed{seed} map shape differs for {image_id}")
    result_path, claim_path = root / RESULT, root / SCORE_CLAIM
    if result_path.exists() or claim_path.exists():
        raise FileExistsError("Model401 valid score already claimed or published")
    claim_hash = claim(claim_path, lock["lock_sha256"], "official_valid_scoring")
    rows = []
    with np.load(root / B4_ARCHIVE, allow_pickle=False) as b4:
        for image_id in ids:
            truth = m400.mask_to_ids(masks[image_id])
            shape = truth.shape
            p_b4 = np.asarray(b4[image_id], dtype=np.float32)
            p1 = _load_archive_map(root / maps[1][image_id]["path"], image_id, shape)
            p2 = _load_archive_map(root / maps[2][image_id]["path"], image_id, shape)
            blended = 0.50 * p_b4 + 0.25 * p1 + 0.25 * p2
            m400._check_probs(blended, shape, f"Model401/{image_id}")
            class_iou = m400.per_class_iou(blended, truth)
            rows.append({"image": image_id,
                         "class_iou_percent": {name: float(value * 100.0)
                                                for name, value in zip(m400.CLASS_NAMES, class_iou)},
                         "image_miou_percent": float(class_iou.mean() * 100.0)})
            print(json.dumps({"stage": "model401_scoring", "image": len(rows),
                              "images": EXPECTED_IDS}, flush=True))
    image_scores = np.asarray([row["image_miou_percent"] for row in rows], dtype=np.float64)
    class_means = {
        name: float(np.mean([row["class_iou_percent"][name] for row in rows]))
        for name in m400.CLASS_NAMES
    }
    score_pct = float(image_scores.mean())
    result = {
        "experiment": "Model401 one-shot Model397 blend valid evaluation",
        "status": "DESCRIPTIVE_SELECTION_CONDITIONAL_VALID_RESULT",
        "metric": CONFIG["metric"],
        "class_order": list(m400.CLASS_NAMES),
        "images": len(rows),
        "image_mean_miou_percent": score_pct,
        "class_image_mean_iou_percent": class_means,
        "difference_from_model364_pp": score_pct - MODEL364_SCORE,
        "reference_model364_valid_percent": MODEL364_SCORE,
        "model397_train_oof_percent": lock["model397_oof_miou_percent"],
        "fixed_probability_weights": CONFIG["family_weights"],
        "config": CONFIG,
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
    probs = np.zeros((4, 2, 2), dtype=np.float32)
    probs[0] = truth == 0
    probs[1] = truth == 1
    m400._check_probs(probs, truth.shape, "synthetic")
    blend = 0.50 * probs + 0.25 * probs + 0.25 * probs
    m400._check_probs(blend, truth.shape, "synthetic blend")
    assert np.array_equal(m400.per_class_iou(blend, truth), [1.0, 1.0, 0.0, 0.0])
    bad = probs.copy()
    bad[0, 0, 0] = 0.2
    try:
        m400._check_probs(bad, truth.shape, "bad")
    except ValueError:
        pass
    else:
        raise AssertionError("unnormalized probabilities were accepted")
    print("Model401 synthetic metric/blend smoke passed; no dataset accessed.")


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
        record = create_lock(root)
        print(json.dumps({"status": "LOCKED", "lock_sha256": record["lock_sha256"],
                          "model397_oof_percent": record["model397_oof_miou_percent"]}, indent=2))
    elif args.verify_lock:
        record = verify_lock(root)
        print(json.dumps({"status": "LOCK_VERIFIED", "lock_sha256": record["lock_sha256"]}, indent=2))
    elif args.infer:
        record = infer(root, args.device)
        print(json.dumps({"status": record["status"], "images": len(record["image_ids"])}, indent=2))
    elif args.score:
        record = score(root)
        print(json.dumps({"status": record["status"],
                          "image_mean_miou_percent": record["image_mean_miou_percent"],
                          "difference_from_model364_pp": record["difference_from_model364_pp"],
                          "class_image_mean_iou_percent": record["class_image_mean_iou_percent"]}, indent=2))


if __name__ == "__main__":
    main()
