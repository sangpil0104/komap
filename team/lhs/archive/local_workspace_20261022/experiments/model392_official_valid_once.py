#!/usr/bin/env python3
"""One-shot official-valid evaluation of the fixed Model392 two-seed ensemble.

The evaluator has no test-split option. It refuses to run unless the locked
two-seed OOF gate passed, both full-train checkpoints match the registered
recipe, and the output directory is new.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
import zipfile

import numpy as np
import torch

from model392_widthnorm_oof import (
    REPO_ROOT,
    modal_width_from_training_split,
    sha256,
    verify_model392_lock,
)


CANDIDATE_ROOT = REPO_ROOT / "model336_hrnet_widthnorm_candidate"
METRIC = "image-wise 4-class mIoU; empty-union=0"
EXPECTED_TRAIN_IMAGES = 70
EXPECTED_VALID_IMAGES = 20
CLASS_NAMES = ["Al", "Al3Ni", "Primary_Si", "Eutectic_Si"]
NUM_CLASSES = len(CLASS_NAMES)


def image_mask_stems(split_root: Path) -> tuple[list[str], list[str]]:
    images = sorted(path.name.removesuffix("_image.png")
                    for path in (split_root / "images").glob("*_image.png"))
    masks = sorted(path.name.removesuffix("_mask.png")
                   for path in (split_root / "masks").glob("*_mask.png"))
    if not images or not masks:
        raise FileNotFoundError(f"images/masks are missing under {split_root}")
    if images != masks:
        raise ValueError(f"image/mask IDs differ under {split_root}")
    return images, masks


def official_image_miou(probabilities: np.ndarray, target: np.ndarray) -> tuple[float, list[float]]:
    if probabilities.ndim != 3 or probabilities.shape[0] != NUM_CLASSES:
        raise ValueError(f"expected (4,H,W) probabilities, got {probabilities.shape}")
    if probabilities.shape[1:] != target.shape:
        raise ValueError("prediction and target dimensions differ")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain NaN or infinity")
    prediction = probabilities.argmax(axis=0)
    values = []
    for class_id in range(NUM_CLASSES):
        truth = target == class_id
        guess = prediction == class_id
        union = int(np.logical_or(truth, guess).sum())
        intersection = int(np.logical_and(truth, guess).sum())
        values.append(intersection / union if union else 0.0)
    return float(np.mean(values)), values


def full_train_args(project_root: Path, seed: int, target_width: int) -> dict[str, Any]:
    data_root = project_root / "data_model392_full"
    run_dir = project_root / "outputs/model392_widthnorm/full_train" / f"seed{seed}"
    return {
        "data_root": str(data_root),
        "output_dir": str(run_dir),
        "device": "cuda:0",
        "architecture": "UnetPlusPlus",
        "encoder_name": "tu-hrnet_w48",
        "encoder_weights": "imagenet",
        "patch_size": 512,
        "input_scale": 2.0,
        "normalize_width": True,
        "samples_per_image": 12,
        "batch_size": 4,
        "accumulation_steps": 2,
        "num_workers": 4,
        "epochs": 70,
        "lr": 3e-4,
        "encoder_lr_mult": 1.0,
        "weight_decay": 1e-4,
        "ce_class_weights": None,
        "init_checkpoint": None,
        "no_boundary_head": False,
        "boundary_head_weight": 1.0,
        "seed": seed,
        "rng_mode": "legacy",
        "val_every": 0,
        "no_amp": False,
        "resume": True,
        "target_width": target_width,
    }


def verify_oof_gate(path: Path, lock_hash: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Model392 confirmatory OOF gate is missing: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if (result.get("model") != 392
            or result.get("stage") != "two_seed_confirmatory"
            or result.get("metric") != METRIC
            or result.get("model392_lock_sha256") != lock_hash
            or result.get("valid_evaluation_authorized") is not True
            or result.get("summary", {}).get("gate_passed") is not True):
        raise ValueError("Model392 OOF result did not authorize the one-shot valid evaluation")
    return result


def claim_one_shot_attempt(path: Path, record: dict[str, Any]) -> str:
    """Persist an exclusive, non-removable claim before reading official valid labels."""
    payload = (json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise FileExistsError(f"Model392 valid evaluation was already claimed: {path}") from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        # Keep the claim even on a failed write: a retry requires a new protocol decision.
        raise
    return sha256(path)


def verify_full_train_checkpoint(
    project_root: Path,
    seed: int,
    target_width: int,
    lock_hash: str,
    gate_hash: str,
    oof_gate_mtime: float,
) -> tuple[Path, dict[str, Any]]:
    project_root = project_root.resolve()
    run_dir = project_root / "outputs/model392_widthnorm/full_train" / f"seed{seed}"
    checkpoint = run_dir / "last.pt"
    args_path = run_dir / "args.json"
    state_path = run_dir / "train_state.pt"
    train_log_path = run_dir / "train.log"
    receipt_path = run_dir / "model392_full_train_receipt.json"
    if not all(path.is_file() for path in (checkpoint, args_path, state_path,
                                           train_log_path, receipt_path)):
        raise FileNotFoundError(f"incomplete full-train Model392 seed{seed} run: {run_dir}")
    expected = full_train_args(project_root, seed, target_width)
    actual = json.loads(args_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise ValueError(f"full-train args mismatch for seed{seed}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    args_hash = hashlib.sha256(
        json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    receipt_checks = {
        "model": (receipt.get("model"), 392),
        "stage": (receipt.get("stage"), "full_train"),
        "seed": (receipt.get("seed"), seed),
        "model392_lock_sha256": (receipt.get("model392_lock_sha256"), lock_hash),
        "oof_gate_sha256": (receipt.get("oof_gate_sha256"), gate_hash),
        "training_args_sha256": (receipt.get("training_args_sha256"), args_hash),
    }
    mismatches = {key: values for key, values in receipt_checks.items()
                  if values[0] != values[1]}
    if mismatches or not receipt.get("attempts"):
        raise ValueError(f"full-train run receipt mismatch for seed{seed}: {mismatches}")
    initialized = datetime.fromisoformat(receipt["initialized_at_utc"].replace("Z", "+00:00"))
    if initialized.tzinfo is None or initialized.timestamp() <= oof_gate_mtime:
        raise ValueError(f"seed{seed} full-train receipt predates its OOF authorization")
    prior_epoch = 0
    for index, attempt in enumerate(receipt["attempts"]):
        started = datetime.fromisoformat(attempt["started_at_utc"].replace("Z", "+00:00"))
        resume_epoch = int(attempt["resume_from_epoch"])
        if (started.tzinfo is None or started.timestamp() <= oof_gate_mtime
                or resume_epoch < prior_epoch or resume_epoch >= 70
                or (index == 0 and resume_epoch != 0)
                or (resume_epoch > 0 and not attempt.get("previous_checkpoint_sha256"))):
            raise ValueError(f"seed{seed} full-train resume history is invalid")
        prior_epoch = resume_epoch
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if int(state.get("epoch", -1)) != 70 or state.get("args") != expected:
        raise ValueError(f"seed{seed} full-train state is not the locked 70-epoch run")
    if "done. final model:" not in train_log_path.read_text(encoding="utf-8", errors="replace"):
        raise ValueError(f"seed{seed} full-train log does not confirm clean completion")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected_preprocessing = {
        "patch_size": 512,
        "input_scale": 2.0,
        "target_width": target_width,
        "normalize_width": True,
    }
    if saved.get("preprocessing_config") != expected_preprocessing:
        raise ValueError(f"seed{seed} checkpoint preprocessing metadata mismatch")
    if saved.get("model_config") != {
        "architecture": "UnetPlusPlus",
        "encoder_name": "tu-hrnet_w48",
        "boundary_head": True,
    }:
        raise ValueError(f"seed{seed} checkpoint architecture mismatch")
    if any(path.stat().st_mtime <= oof_gate_mtime
           for path in (checkpoint, args_path, state_path, train_log_path, receipt_path)):
        raise ValueError(f"seed{seed} full-train artifacts predate the authorized OOF gate")
    return checkpoint, expected


def run(args: argparse.Namespace) -> dict[str, Any]:
    project_root = args.project_root.resolve()
    lock_path = args.model392_lock.resolve()
    verify_model392_lock(lock_path)
    lock_hash = sha256(lock_path)
    gate_path = (project_root / "experiments/results/model392_widthnorm_two_seed_confirmatory.json")
    verify_oof_gate(gate_path, lock_hash)
    gate_hash = sha256(gate_path)
    gate_mtime = gate_path.stat().st_mtime

    data_root = project_root / "data"
    train_root = data_root / "train"
    valid_root = data_root / "valid"
    if not train_root.is_dir() or not valid_root.is_dir():
        raise FileNotFoundError("expected official data/train and data/valid directories")
    train_ids, _ = image_mask_stems(train_root)
    if len(train_ids) != EXPECTED_TRAIN_IMAGES:
        raise ValueError(f"expected 70 official train images, found {len(train_ids)}")

    full_data_root = project_root / "data_model392_full"
    train_link = full_data_root / "train"
    if not train_link.is_dir() or train_link.resolve() != train_root.resolve():
        raise ValueError("full-train Model392 data link does not resolve to official data/train")

    if str(CANDIDATE_ROOT) not in sys.path:
        sys.path.insert(0, str(CANDIDATE_ROOT))
    from komap_seg.checkpoint import load_model_and_preprocessing, resolve_preprocessing
    from komap_seg.dataset import (
        KomapFullDataset,
        effective_input_scale,
        scaled_overlap,
    )
    from komap_seg.inference import predict_probs_tta

    target_width = modal_width_from_training_split(train_root / "images")
    checkpoints = [verify_full_train_checkpoint(
        project_root, seed, target_width, lock_hash, gate_hash, gate_mtime
    )[0]
                   for seed in (1, 2)]
    if sha256(checkpoints[0]) == sha256(checkpoints[1]):
        raise ValueError("two-seed Model392 checkpoints unexpectedly have identical hashes")

    output_dir = project_root / "outputs/model392_widthnorm/official_valid_once"
    claim_path = output_dir.with_suffix(".claim")
    if output_dir.exists():
        raise FileExistsError(f"one-shot Model392 valid output already exists: {output_dir}")
    if claim_path.exists():
        raise FileExistsError(f"one-shot Model392 valid attempt was already claimed: {claim_path}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")
    models = []
    preprocessings = []
    for checkpoint in checkpoints:
        model, saved = load_model_and_preprocessing(checkpoint, device)
        preprocessing = resolve_preprocessing(saved)
        models.append(model)
        preprocessings.append(preprocessing)
    if preprocessings[0] != preprocessings[1]:
        raise ValueError("two full-train seed checkpoints use different preprocessing")
    preprocessing = preprocessings[0]
    if preprocessing["target_width"] != target_width:
        raise ValueError("checkpoint target width differs from full-train image-width mode")

    claim_hash = claim_one_shot_attempt(claim_path, {
        "model": 392,
        "stage": "official_valid_once_claimed",
        "model392_lock_sha256": lock_hash,
        "oof_gate_sha256": gate_hash,
        "checkpoint_sha256": [sha256(path) for path in checkpoints],
        "claimed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "policy": "claim is never removed; after claim, any failure requires a new protocol decision before retry",
    })

    valid_ids, _ = image_mask_stems(valid_root)
    if len(valid_ids) != EXPECTED_VALID_IMAGES:
        raise ValueError(f"expected 20 official valid images, found {len(valid_ids)}")
    dataset = KomapFullDataset(valid_root / "images", valid_root / "masks")
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any]
    with tempfile.TemporaryDirectory(prefix=".model392_valid_once_", dir=output_dir.parent) as tmp:
        staging = Path(tmp)
        probabilities_path = staging / "valid_probabilities.npz"
        with zipfile.ZipFile(probabilities_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            with torch.inference_mode():
                for sample in dataset:
                    scale = effective_input_scale(
                        preprocessing["input_scale"], preprocessing["target_width"], sample["width"]
                    )
                    overlap = scaled_overlap(
                        preprocessing["patch_size"], preprocessing["input_scale"], 170, scale
                    )
                    seed_probabilities = []
                    for model in models:
                        value = predict_probs_tta(
                            model,
                            sample["image"],
                            patch_size=preprocessing["patch_size"],
                            overlap=overlap,
                            batch_size=2,
                            tta=True,
                            input_scale=scale,
                            device=device,
                            tta_mode="flip4",
                            downsample_antialias=False,
                        )
                        seed_probabilities.append(value.cpu().numpy().astype(np.float32))
                    mean_probabilities = (seed_probabilities[0] + seed_probabilities[1]) * 0.5
                    if not np.isfinite(mean_probabilities).all():
                        raise ValueError(f"non-finite ensemble probabilities for {sample['stem']}")
                    if not np.allclose(mean_probabilities.sum(axis=0), 1.0, atol=2e-3, rtol=2e-3):
                        raise ValueError(f"ensemble probabilities are not normalized for {sample['stem']}")
                    image_miou, class_iou = official_image_miou(
                        mean_probabilities, sample["mask"].numpy()
                    )
                    buffer = io.BytesIO()
                    np.save(buffer, mean_probabilities.astype(np.float16), allow_pickle=False)
                    archive.writestr(f"{sample['stem']}.npy", buffer.getvalue())
                    row: dict[str, Any] = {
                        "image_id": sample["stem"],
                        "native_width": sample["width"],
                        "image_miou_empty_zero": image_miou,
                        "effective_input_scale": scale,
                        "effective_overlap": overlap,
                    }
                    for name, value in zip(CLASS_NAMES, class_iou):
                        row[f"iou_{name}_empty_zero"] = value
                    rows.append(row)

        if len(rows) != EXPECTED_VALID_IMAGES:
            raise ValueError(f"evaluated {len(rows)} valid images, expected {EXPECTED_VALID_IMAGES}")
        summary = {
            "model": 392,
            "stage": "official_valid_once",
            "metric": METRIC,
            "split": "valid",
            "images": len(rows),
            "image_miou_empty_zero": float(np.mean([row["image_miou_empty_zero"] for row in rows])),
            "class_image_mean_iou_empty_zero": {
                name: float(np.mean([row[f"iou_{name}_empty_zero"] for row in rows]))
                for name in CLASS_NAMES
            },
            "class_order": CLASS_NAMES,
            "ensemble": {"seeds": [1, 2], "probability_weights": [0.5, 0.5]},
            "inference": {
                "patch_size": 512,
                "base_input_scale": 2.0,
                "target_width": target_width,
                "overlap_base": 170,
                "overlap_rule": "preserve overlap/native-tile ratio",
                "tta": "flip4",
                "batch_size": 2,
                "device": str(device),
                "downsample_antialias": False,
            },
            "valid_evaluation_authorized_by_oof_gate": True,
            "attempt_claim_path": str(claim_path),
            "attempt_claim_sha256": claim_hash,
            "oof_gate_sha256": sha256(gate_path),
            "model392_lock_sha256": lock_hash,
            "checkpoints": [
                {"seed": seed, "path": str(path), "sha256": sha256(path)}
                for seed, path in zip((1, 2), checkpoints)
            ],
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        with (staging / "valid_per_image.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary["probabilities_sha256"] = sha256(staging / "valid_probabilities.npz")
        summary["per_image_csv_sha256"] = sha256(staging / "valid_per_image.csv")
        (staging / "valid_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "valid_manifest.json").write_text(
            json.dumps({
                "model392_lock_sha256": lock_hash,
                "oof_gate_sha256": sha256(gate_path),
                "attempt_claim_sha256": claim_hash,
                "checkpoint_sha256": [sha256(path) for path in checkpoints],
                "valid_ids": valid_ids,
                "metric": METRIC,
                "summary_sha256": sha256(staging / "valid_summary.json"),
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.rename(staging, output_dir)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--model392-lock", type=Path,
        default=REPO_ROOT / "experiments/results/model392_widthnorm_runtime_lock.json",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
