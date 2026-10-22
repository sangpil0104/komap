#!/usr/bin/env python3
"""One-shot, fixed Model397 blend evaluation on the official valid set."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

import model396_fixed_architecture_oof as model396
import model397_fixed_b4_model392_blend as model397


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "experiments/model397_fixed_b4_model392_valid_once.py"
SMOKE = "experiments/model397_fixed_b4_model392_valid_once_smoke.py"
LOCK = "experiments/results/model397_fixed_b4_model392_valid_lock.json"
RESULT = "experiments/results/model397_fixed_b4_model392_blend_valid_once.json"
CLAIM = RESULT + ".claim"
VALID_ROOT = "data/valid"
B4_ROOT = "outputs/model329_fullstats_width4000_uniform_group_oof"
MODEL392_ROOT = "outputs/model392_widthnorm"
MODEL392_LOCK = "experiments/results/model392_widthnorm_runtime_lock.json"
MODEL392_GATE = "experiments/results/model392_widthnorm_two_seed_confirmatory.json"
GROUP_RE = re.compile(r"^(A\d+)_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_masks(root: Path) -> dict[str, Path]:
    directory = root / VALID_ROOT / "masks"
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    found = {path.name.removesuffix("_mask.png"): path
             for path in sorted(directory.glob("*_mask.png"))}
    if len(found) != 20 or len(found) != len(set(found)):
        raise ValueError(f"expected 20 unique official-valid masks, found {len(found)}")
    return found


def verify_authorization(root: Path) -> dict:
    oof_lock = model397.verify_lock(root)
    oof_result_path = root / model397.RESULT
    oof_result = json.loads(oof_result_path.read_text(encoding="utf-8"))
    model392_gate_path = root / MODEL392_GATE
    model392_gate = json.loads(model392_gate_path.read_text(encoding="utf-8"))
    if (oof_result.get("experiment") != "Model397 fixed B4/Model392 grouped-OOF blend"
            or oof_result.get("status") != "OOF_SCREEN_PASS"
            or oof_result.get("valid_stage_allowed") is not True
            or oof_result.get("lock_sha256") != sha256(root / model397.LOCK)
            or oof_lock.get("experiment") != "Model397 fixed B4/Model392 grouped-OOF blend"):
        raise ValueError("Model397 OOF gate did not authorize its one-shot valid stage")
    if (model392_gate.get("model") != 392
            or model392_gate.get("stage") != "two_seed_confirmatory"
            or model392_gate.get("valid_evaluation_authorized") is not True
            or model392_gate.get("summary", {}).get("gate_passed") is not True
            or model392_gate.get("model392_lock_sha256") != sha256(root / MODEL392_LOCK)):
        raise ValueError("Model392 primary OOF gate is not a valid authorization")

    valid_dir = root / MODEL392_ROOT / "official_valid_once"
    manifest_path = valid_dir / "valid_manifest.json"
    summary_path = valid_dir / "valid_summary.json"
    probability_path = valid_dir / "valid_probabilities.npz"
    required = (manifest_path, summary_path, probability_path,
                valid_dir / "valid_per_image.csv")
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("Model392 one-shot valid artifacts are incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (manifest.get("metric") != "image-wise 4-class mIoU; empty-union=0"
            or manifest.get("model392_lock_sha256") != sha256(root / MODEL392_LOCK)
            or manifest.get("oof_gate_sha256") != sha256(model392_gate_path)
            or manifest.get("summary_sha256") != sha256(summary_path)
            or summary.get("stage") != "official_valid_once"
            or summary.get("metric") != "image-wise 4-class mIoU; empty-union=0"
            or summary.get("probabilities_sha256") != sha256(probability_path)):
        raise ValueError("Model392 one-shot valid provenance/hash mismatch")
    ids = set(valid_masks(root))
    if set(manifest.get("valid_ids", [])) != ids or summary.get("images") != 20:
        raise ValueError("Model392 one-shot valid IDs/count differ from the official split")
    expected_checkpoints = [
        sha256(root / MODEL392_ROOT / "full_train" / f"seed{seed}" / "last.pt")
        for seed in (1, 2)
    ]
    if manifest.get("checkpoint_sha256") != expected_checkpoints:
        raise ValueError("Model392 valid manifest checkpoint hashes differ from full-train models")
    return {"oof_result": oof_result, "model392_gate": model392_gate,
            "valid_ids": sorted(ids)}


def input_paths(root: Path) -> list[Path]:
    ids = valid_masks(root)
    b4_dir = root / B4_ROOT
    model392_valid = root / MODEL392_ROOT / "official_valid_once"
    paths = [
        root / model397.PROTOCOL,
        root / model397.SCRIPT,
        root / model397.SMOKE,
        root / SCRIPT,
        root / SMOKE,
        root / model397.LOCK,
        root / model397.RESULT,
        root / MODEL392_LOCK,
        root / MODEL392_GATE,
        model392_valid / "valid_manifest.json",
        model392_valid / "valid_summary.json",
        model392_valid / "valid_probabilities.npz",
        model392_valid / "valid_per_image.csv",
        b4_dir / "valid_ensemble" / "probabilities.npz",
        b4_dir / "valid_ensemble" / "scores.json",
        root / "run_model329_hybrid_width4000_uniform_group_oof.sh",
        root / "model326_hybrid_fullstats/KoMaP/predict.py",
        root / "model326_hybrid_fullstats/KoMaP/common.py",
    ]
    for fold in range(1, 5):
        paths.extend((b4_dir / f"fold{fold}/args.json",
                      b4_dir / f"fold{fold}/last.pt"))
    paths.extend(ids.values())
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return sorted({path.resolve() for path in paths})


def verify_b4_valid(root: Path, ids: set[str]) -> Path:
    b4_dir = root / B4_ROOT
    for fold in range(1, 5):
        args = json.loads((b4_dir / f"fold{fold}/args.json").read_text(encoding="utf-8"))
        if (any(args.get(key) != value for key, value in model396.B4_EXPECTED_ARGS.items())
                or not str(args.get("data", "")).endswith("/data_model322_group_cv")
                or args.get("train_splits") != f"train_fold{fold}"):
            raise ValueError(f"Model329 valid ensemble fold{fold} recipe mismatch")
    archive_path = b4_dir / "valid_ensemble" / "probabilities.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        if set(archive.files) != ids:
            raise ValueError("Model329 valid ensemble probability IDs differ from the 20 fixed masks")
    return archive_path


def create_lock(root: Path) -> dict:
    lock_path = root / LOCK
    if lock_path.exists() or (root / RESULT).exists() or (root / CLAIM).exists():
        raise FileExistsError("refusing to overwrite or repeat Model397 valid stage")
    verify_authorization(root)
    ids = set(valid_masks(root))
    verify_b4_valid(root, ids)
    model392_archive = root / MODEL392_ROOT / "official_valid_once" / "valid_probabilities.npz"
    with np.load(model392_archive, allow_pickle=False) as archive:
        if set(archive.files) != ids:
            raise ValueError("Model392 valid probability IDs differ from the 20 fixed masks")
    files = {
        path.resolve().relative_to(root.resolve()).as_posix(): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in input_paths(root)
    }
    receipt = {
        "experiment": "Model397 fixed B4/Model392 one-shot valid blend",
        "revision": 1,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "metric": "image-wise 4-class mIoU; empty-union=0",
        "class_order": list(model396.CLASS_NAMES),
        "fixed_probability_weights": {"Model329_B4": 0.5, "Model392_seed_mean": 0.5},
        "valid_images": 20,
        "test_inputs_excluded": True,
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
    if (receipt.get("experiment") != "Model397 fixed B4/Model392 one-shot valid blend"
            or receipt.get("revision") != 1
            or receipt.get("metric") != "image-wise 4-class mIoU; empty-union=0"
            or receipt.get("fixed_probability_weights") != {
                "Model329_B4": 0.5, "Model392_seed_mean": 0.5
            }
            or receipt.get("test_inputs_excluded") is not True):
        raise ValueError("unexpected Model397 valid lock or blend weights")
    expected = {item.resolve().relative_to(root.resolve()).as_posix()
                for item in input_paths(root)}
    locked = receipt.get("locked_files", {})
    if set(locked) != expected:
        raise ValueError("Model397 valid input inventory differs from the frozen lock")
    for name, metadata in locked.items():
        item = root / name
        if item.stat().st_size != metadata["size_bytes"] or sha256(item) != metadata["sha256"]:
            raise ValueError(f"Model397 valid locked input changed: {name}")
    return receipt


def _claim(path: Path, lock_hash: str) -> str:
    payload = (json.dumps({
        "experiment": "Model397 fixed B4/Model392 one-shot valid blend",
        "claimed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "lock_sha256": lock_hash,
        "policy": "one-shot; claim retained after interruption",
    }, sort_keys=True, indent=2) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(path)


def score_valid(root: Path) -> dict:
    output_path = root / RESULT
    claim_path = root / CLAIM
    if output_path.exists() or claim_path.exists():
        raise FileExistsError("Model397 official-valid scoring was already claimed/published")
    claim_hash = _claim(claim_path, sha256(root / LOCK))
    receipt = verify_lock(root)
    ids = valid_masks(root)
    b4_path = verify_b4_valid(root, set(ids))
    model392_path = root / MODEL392_ROOT / "official_valid_once" / "valid_probabilities.npz"
    rows = []
    scores = {arm: [] for arm in ("Model329_B4", "Model392_seed_mean", "Model397_blend")}
    with np.load(b4_path, allow_pickle=False) as b4, np.load(model392_path, allow_pickle=False) as hrnet:
        if set(b4.files) != set(ids) or set(hrnet.files) != set(ids):
            raise ValueError("fixed Model397 valid blend IDs changed after the input lock")
        for image_id in sorted(ids):
            truth = model396.mask_to_ids(ids[image_id])
            p_b4 = np.asarray(b4[image_id], dtype=np.float32)
            p_hrnet = np.asarray(hrnet[image_id], dtype=np.float32)
            blended = 0.5 * p_b4 + 0.5 * p_hrnet
            arm_probs = {"Model329_B4": p_b4, "Model392_seed_mean": p_hrnet,
                         "Model397_blend": blended}
            arm_class_iou = {
                arm: model396.class_ious(probabilities, truth)
                for arm, probabilities in arm_probs.items()
            }
            for arm, class_iou in arm_class_iou.items():
                scores[arm].append(class_iou)
            group = GROUP_RE.match(image_id)
            if group is None:
                raise ValueError(f"cannot parse valid specimen ID: {image_id}")
            rows.append({
                "image_id": image_id,
                "specimen_group": group.group(1),
                "arms": {
                    arm: {"class_iou_empty0": class_iou.tolist(),
                          "image_miou_empty0": float(class_iou.mean())}
                    for arm, class_iou in arm_class_iou.items()
                },
            })
    if len(rows) != 20:
        raise ValueError(f"scored {len(rows)} valid images, expected 20")
    summaries = {}
    for arm, arm_scores in scores.items():
        per_class = np.asarray(arm_scores, dtype=np.float64)
        summaries[arm] = {
            "image_mean_miou_percent": float(per_class.mean(axis=1).mean() * 100.0),
            "class_image_mean_iou_percent": {
                name: float(per_class[:, index].mean() * 100.0)
                for index, name in enumerate(model396.CLASS_NAMES)
            },
        }
    result = {
        "experiment": "Model397 fixed B4/Model392 one-shot valid blend",
        "status": "DESCRIPTIVE_VALID_RESULT",
        "metric": "image-wise 4-class mIoU; empty-union=0",
        "class_order": list(model396.CLASS_NAMES),
        "split": "official_valid",
        "images": len(rows),
        "image_mean_miou_percent": summaries["Model397_blend"]["image_mean_miou_percent"],
        "class_image_mean_iou_percent": summaries["Model397_blend"]["class_image_mean_iou_percent"],
        "fixed_arms_descriptive_only": summaries,
        "fixed_probability_weights": {"Model329_B4": 0.5, "Model392_seed_mean": 0.5},
        "official_valid_used": True,
        "test_used": False,
        "independent_confirmation": False,
        "interpretation": "one additional, adaptively reused 20-image validation look; descriptive only",
        "lock_sha256": sha256(root / LOCK),
        "claim_sha256": claim_hash,
        "oof_gate_sha256": sha256(root / model397.RESULT),
        "Model392_valid_result_sha256": sha256(root / MODEL392_ROOT / "official_valid_once" / "valid_summary.json"),
        "locked_file_count": len(receipt["locked_files"]),
        "per_image": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result


def self_test() -> None:
    truth = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
    one_hot = np.zeros((4, 2, 2), dtype=np.float32)
    one_hot[0] = truth == 0
    one_hot[1] = truth == 1
    blend = 0.5 * one_hot + 0.5 * one_hot
    assert np.allclose(model396.class_ious(blend, truth), [1.0, 1.0, 0.0, 0.0])
    print("Model397 valid-stage smoke passed: fixed 50:50 probability blend and official empty=0 metric.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--create-lock", action="store_true")
    modes.add_argument("--verify-lock", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.self_test:
        self_test()
    elif args.create_lock:
        lock = create_lock(root)
        print(json.dumps({"locked_files": len(lock["locked_files"]), "lock": str(root / LOCK)}, indent=2))
    elif args.verify_lock:
        lock = verify_lock(root)
        print(json.dumps({"verified_files": len(lock["locked_files"]), "lock_sha256": sha256(root / LOCK)}, indent=2))
    else:
        print(json.dumps(score_valid(root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
