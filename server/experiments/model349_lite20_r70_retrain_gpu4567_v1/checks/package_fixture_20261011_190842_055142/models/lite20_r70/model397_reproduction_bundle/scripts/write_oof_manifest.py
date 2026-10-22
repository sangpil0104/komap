#!/usr/bin/env python3
"""Create/verify checkpoint-bound provenance for one train-only OOF archive."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FOLD_FILE = ROOT / "config/model397_group_folds.json"
CLASS_ORDER = ["Al", "Al3Ni", "Primary Si", "Eutectic Si"]
GROUP_RE = re.compile(r"^(A[0-9]+)_")
IMMUTABLE_KEYS = {
    "schema", "family", "seed", "fold", "metric", "class_order",
    "oof_image_ids", "heldout_groups", "fold_config_sha256",
    "checkpoint", "checkpoint_sha256", "training_args", "training_args_sha256",
    "probabilities", "probabilities_sha256", "training_source_sha256",
    "inference_source_sha256", "official_valid_read", "test_read",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config() -> tuple[dict[int, set[str]], str]:
    record = json.loads(FOLD_FILE.read_text(encoding="utf-8"))
    mapping = {int(item["fold"]): set(item["heldout_groups"]) for item in record["folds"]}
    return mapping, sha256(FOLD_FILE)


def expected_record(output_root: Path, family: str, fold: int, seed: int | None) -> tuple[dict, Path]:
    output_root = output_root.resolve()
    fold_groups, fold_hash = config()
    if fold not in fold_groups:
        raise ValueError(f"unknown fold {fold}")
    if family == "model329":
        if seed not in (None, 0):
            raise ValueError("Model329 uses its fixed training seed 0")
        seed = 0
        data_root = output_root / "splits/model329_cv"
        split = f"oof_fold{fold}"
        run_dir = output_root / f"model329/fold{fold}"
        prediction_dir = output_root / f"model329/fold{fold}_oof"
        probability_path = prediction_dir / "probabilities.npz"
        training_source = ROOT / "model329/train.py"
        inference_source = ROOT / "model329/predict.py"
        family_name = "Model329 EfficientNet-B4 U-Net"
    else:
        if seed not in (1, 2):
            raise ValueError("Model392 seed must be 1 or 2")
        data_root = output_root / f"splits/model392_cv/seed{seed}_fold{fold}"
        split = "valid"
        run_dir = output_root / f"model392/seed{seed}/fold{fold}"
        prediction_dir = output_root / f"model392/seed{seed}/fold{fold}_oof"
        probability_path = prediction_dir / "valid_probabilities.npz"
        training_source = ROOT / "model392/train.py"
        inference_source = ROOT / "model392/evaluate_image_mean.py"
        family_name = "Model392 HRNet-W48 U-Net++"

    image_dir = data_root / split / "images"
    ids = sorted(path.name.removesuffix("_image.png") for path in image_dir.glob("*_image.png"))
    groups = sorted({GROUP_RE.match(image_id + "_").group(1) for image_id in ids})
    if not ids or set(groups) != fold_groups[fold]:
        raise ValueError(f"{family_name} fold{fold} split does not match the fixed group manifest")
    expected_ids = {
        image_id for image_id in ids
        if GROUP_RE.match(image_id + "_").group(1) in fold_groups[fold]
    }
    if len(expected_ids) != len(ids):
        raise ValueError("OOF split contains a non-held-out group")

    args_path = run_dir / "args.json"
    checkpoint_path = run_dir / "last.pt"
    for path in (args_path, checkpoint_path, probability_path, training_source, inference_source):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(probability_path, allow_pickle=False) as archive:
        if set(archive.files) != expected_ids:
            raise ValueError(f"{family_name} fold{fold} probability archive ID mismatch")
        for image_id in ids:
            probability = np.asarray(archive[image_id], dtype=np.float32)
            if (probability.ndim != 3 or probability.shape[0] != 4
                    or not np.isfinite(probability).all()
                    or probability.min() < -1e-4 or probability.max() > 1.0001
                    or np.max(np.abs(probability.sum(axis=0) - 1.0)) > 0.03):
                raise ValueError(f"invalid OOF probability tensor for {image_id}")

    args_record = json.loads(args_path.read_text(encoding="utf-8"))
    record = {
        "schema": 1,
        "family": family_name,
        "seed": seed,
        "fold": fold,
        "metric": "image-wise 4-class mIoU; empty-union=0",
        "class_order": CLASS_ORDER,
        "oof_image_ids": ids,
        "heldout_groups": groups,
        "fold_config_sha256": fold_hash,
        "checkpoint": checkpoint_path.relative_to(output_root).as_posix(),
        "checkpoint_sha256": sha256(checkpoint_path),
        "training_args": args_record,
        "training_args_sha256": sha256(args_path),
        "probabilities": probability_path.relative_to(output_root).as_posix(),
        "probabilities_sha256": sha256(probability_path),
        "training_source_sha256": sha256(training_source),
        "inference_source_sha256": sha256(inference_source),
        "official_valid_read": False,
        "test_read": False,
    }
    return record, prediction_dir / "model397_oof_manifest.json"


def verify_manifest(output_root: Path, family: str, fold: int, seed: int | None) -> dict:
    expected, path = expected_record(output_root, family, fold, seed)
    if not path.is_file():
        raise FileNotFoundError(f"missing OOF provenance: {path}")
    existing = json.loads(path.read_text(encoding="utf-8"))
    if any(existing.get(key) != expected[key] for key in IMMUTABLE_KEYS):
        raise ValueError(f"OOF provenance/checkpoint/archive mismatch: {path}")
    return existing


def create_or_verify(output_root: Path, family: str, fold: int, seed: int | None) -> dict:
    expected, path = expected_record(output_root, family, fold, seed)
    if path.exists():
        return verify_manifest(output_root, family, fold, seed)
    receipt = {
        **expected,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return verify_manifest(output_root, family, fold, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--family", choices=("model329", "model392"), required=True)
    parser.add_argument("--fold", choices=(1, 2, 3, 4), required=True, type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    record = create_or_verify(args.output_root, args.family, args.fold, args.seed)
    print(f"verified {record['family']} seed={record['seed']} fold={record['fold']} OOF provenance")


if __name__ == "__main__":
    main()
