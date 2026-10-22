#!/usr/bin/env python3
"""Complete Model401 after its progress-logger-only inference interruption."""

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

import model397_fixed_b4_model392_blend as m397
import model400_model396_d2_valid_once as m400
import model401_model397_blend_valid_once as m401


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "experiments/model401_model397_blend_recovery.md"
SCRIPT = "experiments/model401_model397_blend_recovery.py"
SMOKE = "experiments/model401_model397_blend_recovery_smoke.py"
LOCK = "experiments/results/model401_model397_blend_recovery_lock.json"
CLAIM = m401.INFER_ROOT + "/recovery.claim"
RECEIPT = m401.INFER_ROOT + "/recovery_receipt.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def exclusive_json(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def base_lock(root: Path) -> dict[str, Any]:
    path = root / m401.LOCK
    record = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in record.items() if key != "lock_sha256"}
    if record.get("lock_sha256") != m401.canonical_sha256(body):
        raise ValueError("base Model401 lock is corrupt")
    expected_script = record.get("locked_files", {}).get(m401.SCRIPT, {}).get("sha256")
    if expected_script != sha256(root / m401.SCRIPT):
        raise ValueError("base Model401 script differs from its one-shot lock")
    return record


def validate_interrupted_state(
    root: Path, *, require_partial: bool = False
) -> tuple[list[str], dict[str, Path], dict[str, Path]]:
    lock = base_lock(root)
    images, masks = m401.valid_pairs(root)
    ids = sorted(images)
    first = ids[0]
    claim_path = root / m401.INFER_CLAIM
    if not claim_path.is_file():
        raise FileNotFoundError("original Model401 inference claim is missing")
    claim_record = json.loads(claim_path.read_text(encoding="utf-8"))
    if (claim_record.get("lock_sha256") != lock["lock_sha256"]
            or claim_record.get("stage") != "inference"):
        raise ValueError("original inference claim does not match the frozen base lock")
    manifest_exists = (root / m401.INFER_MANIFEST).exists()
    if (root / m401.RESULT).exists() or (root / m401.SCORE_CLAIM).exists():
        raise FileExistsError("Model401 score already exists; recovery scope is invalid")
    observed: dict[int, list[str]] = {}
    for seed, directory in ((1, root / m401.INFER_ROOT_SEED1),
                            (2, root / m401.INFER_ROOT_SEED2)):
        actual = sorted(path.name for path in directory.glob("*.npz"))
        complete = sorted(f"{image_id}.npz" for image_id in ids)
        if actual not in ([f"{first}.npz"], complete):
            raise ValueError(f"unexpected seed{seed} map inventory: {actual[:5]}")
        observed[seed] = actual
        with Image.open(images[first]) as image:
            shape = (image.height, image.width)
        m401._load_archive_map(directory / f"{first}.npz", first, shape)
    if observed[1] != observed[2]:
        raise ValueError("seed1 and seed2 maps are at different completion stages")
    partial = observed[1] == [f"{first}.npz"]
    if require_partial and (not partial or manifest_exists):
        raise ValueError("recovery lock must start from exactly the original partial state")
    if manifest_exists and partial:
        raise ValueError("inference manifest exists before all Model401 maps are complete")
    return ids, images, masks


def input_paths(root: Path) -> list[Path]:
    ids, images, masks = validate_interrupted_state(root)
    first = ids[0]
    paths = [
        root / PROTOCOL, root / SCRIPT, root / SMOKE,
        root / m401.PROTOCOL, root / m401.SCRIPT, root / m401.SMOKE, root / m401.LOCK,
        root / m397.PROTOCOL, root / m397.SCRIPT, root / m397.LOCK, root / m397.RESULT,
        root / m400.SCRIPT, root / m401.INFER_CLAIM,
        root / m401.INFER_ROOT_SEED1 / f"{first}.npz",
        root / m401.INFER_ROOT_SEED2 / f"{first}.npz",
        root / m401.B4_ARCHIVE,
        root / m400.B4_ROOT / "valid_ensemble/scores.json",
        root / m401.MODEL392_LOCK,
        root / "model336_hrnet_widthnorm_candidate/komap_seg/checkpoint.py",
        root / "model336_hrnet_widthnorm_candidate/komap_seg/dataset.py",
        root / "model336_hrnet_widthnorm_candidate/komap_seg/inference.py",
        root / "model336_hrnet_widthnorm_candidate/komap_seg/model.py",
        root / "model336_hrnet_widthnorm_candidate/komap_seg/palette.py",
    ]
    for fold in range(1, 5):
        paths.extend((root / m400.B4_ROOT / f"fold{fold}/args.json",
                      root / m400.B4_ROOT / f"fold{fold}/last.pt"))
        for seed in (1, 2):
            run = root / m401.MODEL392_ROOT / f"seed{seed}/fold{fold}"
            oof = root / m401.MODEL392_ROOT / f"seed{seed}/fold{fold}_oof"
            paths.extend((run / "last.pt", run / "args.json", run / "model392_run_receipt.json",
                          oof / "model392_manifest.json"))
    paths.extend(images.values())
    paths.extend(masks.values())
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing recovery input: {missing[:5]}")
    return sorted({path.resolve() for path in paths})


def create_lock(root: Path) -> dict[str, Any]:
    if (root / LOCK).exists() or (root / CLAIM).exists() or (root / RECEIPT).exists():
        raise FileExistsError("Model401 recovery already locked or claimed")
    ids, _, _ = validate_interrupted_state(root, require_partial=True)
    paths = input_paths(root)
    files = {
        path.resolve().relative_to(root.resolve()).as_posix(): {
            "size_bytes": path.stat().st_size, "sha256": sha256(path)
        }
        for path in paths
    }
    base = base_lock(root)
    record = {
        "experiment": "Model401 logger-only inference recovery",
        "revision": 1,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "base_model401_lock_sha256": base["lock_sha256"],
        "preserved_image_id": ids[0],
        "remaining_image_ids": ids[1:],
        "same_candidate_and_inference_config": True,
        "valid_labels_not_decoded": True,
        "test_excluded": True,
        "locked_files": files,
        "runtime": m401.runtime_record(),
    }
    record["lock_sha256"] = m401.canonical_sha256(record)
    exclusive_json(root / LOCK, record)
    return record


def verify_lock(root: Path) -> dict[str, Any]:
    record = json.loads((root / LOCK).read_text(encoding="utf-8"))
    body = {key: value for key, value in record.items() if key != "lock_sha256"}
    if record.get("lock_sha256") != m401.canonical_sha256(body):
        raise ValueError("Model401 recovery lock hash is invalid")
    base = base_lock(root)
    ids, _, _ = validate_interrupted_state(root)
    if (record.get("experiment") != "Model401 logger-only inference recovery"
            or record.get("base_model401_lock_sha256") != base["lock_sha256"]
            or record.get("preserved_image_id") != ids[0]
            or record.get("remaining_image_ids") != ids[1:]
            or record.get("runtime") != m401.runtime_record()):
        raise ValueError("Model401 recovery lock/base state mismatch")
    expected = {p.resolve().relative_to(root.resolve()).as_posix() for p in input_paths(root)}
    locked = record.get("locked_files", {})
    if set(locked) != expected:
        raise ValueError("Model401 recovery input inventory changed")
    for relative, meta in locked.items():
        path = root / relative
        if path.stat().st_size != meta["size_bytes"] or sha256(path) != meta["sha256"]:
            raise ValueError(f"Model401 recovery input changed: {relative}")
    return record


def recovery_claim(root: Path, lock: dict[str, Any]) -> str:
    return exclusive_json(root / CLAIM, {
        "experiment": "Model401 logger-only inference recovery",
        "stage": "complete_missing_maps",
        "claimed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "base_model401_lock_sha256": lock["base_model401_lock_sha256"],
        "recovery_lock_sha256": lock["lock_sha256"],
        "policy": "exclusive; only missing maps; no overwrite or rerun",
    })


def complete_inference(root: Path, device_name: str) -> dict[str, Any]:
    lock = verify_lock(root)
    ids, images, _ = validate_interrupted_state(root)
    claim_hash = recovery_claim(root, lock)
    model_dir = root / "model336_hrnet_widthnorm_candidate"
    sys.path.insert(0, str(model_dir))
    import torch
    from komap_seg.checkpoint import load_model_and_preprocessing, resolve_preprocessing
    from komap_seg.dataset import KomapFullDataset, effective_input_scale, scaled_overlap
    from komap_seg.inference import predict_probs_tta

    device = torch.device(device_name)
    dataset = KomapFullDataset(root / m401.VALID_IMAGES, masks_dir=None)
    index_by_id = {pair["stem"]: index for index, pair in enumerate(dataset.pairs)}
    if set(index_by_id) != set(ids):
        raise ValueError("image-only dataset differs from the locked 20 valid IDs")
    models = []
    checkpoint_hashes: dict[str, dict[str, str]] = {"1": {}, "2": {}}
    for seed in (1, 2):
        for fold in range(1, 5):
            path = root / m401.MODEL392_ROOT / f"seed{seed}/fold{fold}/last.pt"
            model, saved = load_model_and_preprocessing(path, device)
            prep = resolve_preprocessing(saved)
            if (prep["patch_size"] != 512 or prep["input_scale"] != 2.0
                    or prep["target_width"] <= 0 or not prep["normalize_width"]):
                raise ValueError(f"Model392 preprocessing mismatch: {path}")
            models.append((seed, fold, model, prep))
            checkpoint_hashes[str(seed)][str(fold)] = sha256(path)

    dirs = {1: root / m401.INFER_ROOT_SEED1, 2: root / m401.INFER_ROOT_SEED2}
    first = ids[0]
    receipts: dict[int, list[dict[str, Any]]] = {1: [], 2: []}
    for seed in (1, 2):
        first_path = dirs[seed] / f"{first}.npz"
        with Image.open(images[first]) as im:
            shape = (im.height, im.width)
        m401._load_archive_map(first_path, first, shape)
        receipts[seed].append({"image_id": first,
                               "path": first_path.resolve().relative_to(root.resolve()).as_posix(),
                               "sha256": sha256(first_path), "shape": list(shape)})

    for number, image_id in enumerate(ids[1:], start=2):
        sample = dataset[index_by_id[image_id]]
        shape = (sample["height"], sample["width"])
        sums = {1: np.zeros((4, *shape), dtype=np.float32),
                2: np.zeros((4, *shape), dtype=np.float32)}
        for seed, fold, model, prep in models:
            scale = effective_input_scale(prep["input_scale"], prep["target_width"], sample["width"])
            overlap = scaled_overlap(512, 2.0, 170, scale)
            probability = predict_probs_tta(
                model, sample["image"], patch_size=512, overlap=overlap,
                batch_size=2, tta=True, input_scale=scale, device=device,
                tta_mode="flip4",
            )
            probs = probability.numpy().astype(np.float32)
            m400._check_probs(probs, shape, f"seed{seed}/fold{fold}/{image_id}")
            sums[seed] += probs * 0.25
            del probability, probs
        for seed in (1, 2):
            path = dirs[seed] / f"{image_id}.npz"
            digest = m400.save_npz_exclusive(path, **{image_id: sums[seed].astype(np.float16)})
            receipts[seed].append({"image_id": image_id,
                                   "path": path.resolve().relative_to(root.resolve()).as_posix(),
                                   "sha256": digest, "shape": list(shape)})
        emit({"stage": "model401_recovery_inference", "image": number,
              "images": len(ids), "seed_fold_models": 8})

    # Include all maps, including the two preserved maps from the original attempt.
    for seed in (1, 2):
        receipts[seed] = []
        for image_id in ids:
            sample = dataset[index_by_id[image_id]]
            path = dirs[seed] / f"{image_id}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            m401._load_archive_map(path, image_id, (sample["height"], sample["width"]))
            receipts[seed].append({"image_id": image_id,
                                   "path": path.resolve().relative_to(root.resolve()).as_posix(),
                                   "sha256": sha256(path),
                                   "shape": [sample["height"], sample["width"]]})
    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()
    base = base_lock(root)
    manifest = {
        "experiment": "Model401 one-shot Model397 blend valid evaluation",
        "status": "INFERENCE_COMPLETE_NOT_SCORED",
        "lock_sha256": base["lock_sha256"],
        "inference_claim_sha256": sha256(root / m401.INFER_CLAIM),
        "image_ids": ids,
        "hrnet_checkpoint_sha256": checkpoint_hashes,
        "hrnet_seed1_mean_maps": receipts[1],
        "hrnet_seed2_mean_maps": receipts[2],
        "b4_valid_archive_sha256": sha256(root / m401.B4_ARCHIVE),
        "config": m401.CONFIG,
        "valid_masks_decoded_during_inference": False,
        "valid_score_computed": False,
        "test_inputs_excluded": True,
        "recovery": {
            "recovery_lock_sha256": lock["lock_sha256"],
            "recovery_claim_sha256": claim_hash,
            "preserved_first_image": first,
            "newly_inferred_images": ids[1:],
        },
    }
    exclusive_json(root / m401.INFER_MANIFEST, manifest)
    receipt = {
        "experiment": "Model401 logger-only inference recovery",
        "status": "INFERENCE_COMPLETE_NOT_SCORED",
        "base_lock_sha256": base["lock_sha256"],
        "recovery_lock_sha256": lock["lock_sha256"],
        "recovery_claim_sha256": claim_hash,
        "preserved_first_image": first,
        "newly_inferred_images": ids[1:],
        "manifest_sha256": sha256(root / m401.INFER_MANIFEST),
        "official_valid_masks_decoded": False,
        "test_inputs_excluded": True,
    }
    exclusive_json(root / RECEIPT, receipt)
    return receipt


def score_once(root: Path) -> dict[str, Any]:
    lock = verify_lock(root)
    base = base_lock(root)
    manifest_path = root / m401.INFER_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recovery_receipt = json.loads((root / RECEIPT).read_text(encoding="utf-8"))
    recovery_claim_record = json.loads((root / CLAIM).read_text(encoding="utf-8"))
    images, masks = m401.valid_pairs(root)
    ids = sorted(images)
    if (manifest.get("status") != "INFERENCE_COMPLETE_NOT_SCORED"
            or manifest.get("lock_sha256") != base["lock_sha256"]
            or manifest.get("image_ids") != ids
            or manifest.get("config") != m401.CONFIG
            or manifest.get("valid_masks_decoded_during_inference") is not False
            or manifest.get("valid_score_computed") is not False
            or manifest.get("test_inputs_excluded") is not True
            or manifest.get("recovery", {}).get("recovery_lock_sha256") != lock["lock_sha256"]
            or recovery_receipt.get("manifest_sha256") != sha256(manifest_path)
            or recovery_claim_record.get("recovery_lock_sha256") != lock["lock_sha256"]):
        raise ValueError("recovered inference manifest/receipt mismatch")
    if (root / m401.RESULT).exists() or (root / m401.SCORE_CLAIM).exists():
        raise FileExistsError("Model401 score already claimed or published")
    b4_path = root / m401.B4_ARCHIVE
    if manifest.get("b4_valid_archive_sha256") != sha256(b4_path):
        raise ValueError("B4 probability archive changed after inference")
    seed_receipts: dict[int, dict[str, dict[str, Any]]] = {}
    for seed, key, relative_dir in (
        (1, "hrnet_seed1_mean_maps", m401.INFER_ROOT_SEED1),
        (2, "hrnet_seed2_mean_maps", m401.INFER_ROOT_SEED2),
    ):
        entries = manifest.get(key, [])
        lookup = {item["image_id"]: item for item in entries}
        if len(entries) != len(ids) or set(lookup) != set(ids):
            raise ValueError(f"seed{seed} map coverage mismatch")
        expected_checkpoints = {
            str(fold): sha256(root / m401.MODEL392_ROOT / f"seed{seed}/fold{fold}/last.pt")
            for fold in range(1, 5)
        }
        if manifest.get("hrnet_checkpoint_sha256", {}).get(str(seed)) != expected_checkpoints:
            raise ValueError(f"seed{seed} checkpoint hashes changed")
        for image_id, item in lookup.items():
            expected_path = (root / relative_dir / f"{image_id}.npz").resolve()
            actual_path = (root / item["path"]).resolve()
            if actual_path != expected_path or sha256(actual_path) != item["sha256"]:
                raise ValueError(f"seed{seed} probability map hash/path mismatch: {image_id}")
        seed_receipts[seed] = lookup
    with np.load(b4_path, allow_pickle=False) as b4:
        if set(b4.files) != set(ids):
            raise ValueError("B4 archive IDs differ from fixed valid IDs")
        for image_id in ids:
            with Image.open(images[image_id]) as im:
                shape = (im.height, im.width)
            m400._check_probs(np.asarray(b4[image_id], dtype=np.float32), shape, f"B4/{image_id}")
            for seed in (1, 2):
                item = seed_receipts[seed][image_id]
                if tuple(item["shape"]) != shape:
                    raise ValueError(f"seed{seed} probability shape mismatch: {image_id}")
                m401._load_archive_map(root / item["path"], image_id, shape)

    claim_hash = m401.claim(root / m401.SCORE_CLAIM, base["lock_sha256"], "official_valid_scoring")
    rows = []
    with np.load(b4_path, allow_pickle=False) as b4:
        for image_id in ids:
            truth = m400.mask_to_ids(masks[image_id])
            shape = truth.shape
            p_b4 = np.asarray(b4[image_id], dtype=np.float32)
            p1 = m401._load_archive_map(root / seed_receipts[1][image_id]["path"], image_id, shape)
            p2 = m401._load_archive_map(root / seed_receipts[2][image_id]["path"], image_id, shape)
            blend = 0.50 * p_b4 + 0.25 * p1 + 0.25 * p2
            m400._check_probs(blend, shape, f"Model401/{image_id}")
            ious = m400.per_class_iou(blend, truth)
            rows.append({"image": image_id,
                         "class_iou_percent": {name: float(value * 100)
                                                for name, value in zip(m400.CLASS_NAMES, ious)},
                         "image_miou_percent": float(ious.mean() * 100)})
            emit({"stage": "model401_recovery_score", "image": len(rows), "images": len(ids)})
    image_scores = np.asarray([row["image_miou_percent"] for row in rows], dtype=np.float64)
    class_means = {name: float(np.mean([row["class_iou_percent"][name] for row in rows]))
                   for name in m400.CLASS_NAMES}
    score_pct = float(image_scores.mean())
    result = {
        "experiment": "Model401 one-shot Model397 blend valid evaluation",
        "status": "DESCRIPTIVE_SELECTION_CONDITIONAL_VALID_RESULT",
        "metric": m401.CONFIG["metric"],
        "class_order": list(m400.CLASS_NAMES),
        "images": len(rows),
        "image_mean_miou_percent": score_pct,
        "class_image_mean_iou_percent": class_means,
        "difference_from_model364_pp": score_pct - m401.MODEL364_SCORE,
        "reference_model364_valid_percent": m401.MODEL364_SCORE,
        "model397_train_oof_percent": base["model397_oof_miou_percent"],
        "fixed_probability_weights": m401.CONFIG["family_weights"],
        "config": m401.CONFIG,
        "valid_inference_different_from_oof_estimand": True,
        "official_valid_used": True,
        "test_used": False,
        "independent_confirmation": False,
        "interpretation": "one additional adaptively reused 20-image validation look; descriptive only",
        "lock_sha256": base["lock_sha256"],
        "recovery_lock_sha256": lock["lock_sha256"],
        "inference_manifest_sha256": sha256(manifest_path),
        "recovery_receipt_sha256": sha256(root / RECEIPT),
        "claim_sha256": claim_hash,
        "per_image": rows,
    }
    exclusive_json(root / m401.RESULT, result)
    return result


def self_test() -> None:
    from contextlib import redirect_stdout
    from io import StringIO

    stream = StringIO()
    with redirect_stdout(stream):
        emit({"stage": "smoke", "image": 1})
    parsed = json.loads(stream.getvalue())
    if parsed != {"image": 1, "stage": "smoke"}:
        raise AssertionError("progress logger did not emit valid JSON")
    truth = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
    probabilities = np.zeros((4, 2, 2), dtype=np.float32)
    probabilities[0] = truth == 0
    probabilities[1] = truth == 1
    blended = 0.50 * probabilities + 0.25 * probabilities + 0.25 * probabilities
    m400._check_probs(blended, truth.shape, "synthetic recovery blend")
    if not np.array_equal(m400.per_class_iou(blended, truth), [1.0, 1.0, 0.0, 0.0]):
        raise AssertionError("official empty-union=0 metric smoke failed")
    print("Model401 recovery smoke passed; no dataset or labels accessed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--self-test", action="store_true")
    actions.add_argument("--create-lock", action="store_true")
    actions.add_argument("--verify-lock", action="store_true")
    actions.add_argument("--complete-inference", action="store_true")
    actions.add_argument("--score", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.self_test:
        self_test()
    elif args.create_lock:
        record = create_lock(root)
        emit({"status": "RECOVERY_LOCKED", "lock_sha256": record["lock_sha256"]})
    elif args.verify_lock:
        record = verify_lock(root)
        emit({"status": "RECOVERY_LOCK_VERIFIED", "lock_sha256": record["lock_sha256"]})
    elif args.complete_inference:
        record = complete_inference(root, args.device)
        emit({"status": record["status"], "new_images": len(record["newly_inferred_images"])})
    elif args.score:
        record = score_once(root)
        emit({"status": record["status"], "image_mean_miou_percent": record["image_mean_miou_percent"],
              "difference_from_model364_pp": record["difference_from_model364_pp"],
              "class_image_mean_iou_percent": record["class_image_mean_iou_percent"]})


if __name__ == "__main__":
    main()
