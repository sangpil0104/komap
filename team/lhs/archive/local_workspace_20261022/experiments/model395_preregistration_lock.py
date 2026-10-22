#!/usr/bin/env python3
"""Create a one-time source/data/provenance lock for Model395 CPU preflight."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/workspace/komap_model1")
GROUP_RE = re.compile(r"^(A\d+)_")
LOCKED_FILES = (
    "experiments/model395_component_sampler_preregistration.md",
    "experiments/model395_component_sampler_simulation.py",
    "experiments/model395_component_sampler_simulation_smoke.py",
    "experiments/model395_component_sampler_smoke.py",
    "experiments/model395_preregistration_lock.py",
    "experiments/model388_component_sampler_audit_preregistration.md",
    "experiments/model388_component_sampler_audit.py",
    "experiments/model388_component_sampler_safety_addendum.md",
    "experiments/results/model388_preregistration_lock.json",
    "experiments/results/model388_safety_addendum_lock.json",
    "experiments/results/model388_exposure_folds3_4.json",
    "experiments/results/model388_headroom_folds3_4_recovery.json",
    "experiments/results/model382_e5_preregistration_lock.json",
    "experiments/results/model382_two_seed_mean_preregistration_lock.json",
    "run_model382_seed_count_control_p100.sh",
)
ALLOWED_SOURCE_DIFFS = {"train.py", "komap_seg/dataset.py"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_inventory(directory: Path) -> dict[str, str]:
    files = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    if not files:
        raise FileNotFoundError(f"empty source/data directory: {directory}")
    return {str(path.relative_to(directory)): sha256(path) for path in files}


def paired_split_inventory(root: Path, split: str) -> dict[str, Any]:
    images_dir = root / "images"
    masks_dir = root / "masks"
    images = {path.name.removesuffix("_image.png"): path for path in images_dir.glob("*_image.png")}
    masks = {path.name.removesuffix("_mask.png"): path for path in masks_dir.glob("*_mask.png")}
    if not images or set(images) != set(masks):
        raise ValueError(f"image/mask inventory mismatch in {root}")
    ids = sorted(images)
    groups: dict[str, list[str]] = {}
    hashes: dict[str, str] = {}
    for image_id in ids:
        match = GROUP_RE.match(image_id)
        if not match:
            raise ValueError(f"cannot parse specimen group for {image_id}")
        groups.setdefault(match.group(1), []).append(image_id)
        for path in (images[image_id], masks[image_id]):
            hashes[str(path.resolve())] = sha256(path)
    return {
        "image_count": len(ids),
        "group_count": len(groups),
        "image_ids": ids,
        "group_to_image_ids": groups,
        "file_sha256": hashes,
        "split_type": split,
    }


def runtime_metadata() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in ("numpy", "opencv-python", "Pillow", "torch", "albumentations", "timm", "segmentation-models-pytorch"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": packages,
        "training_started": False,
        "official_valid_or_test_read": False,
        "gpu_occupancy_or_memory_queried": False,
    }


def model382_seed1_artifacts() -> dict[str, Any]:
    output_root = ROOT / "outputs/model382_seed_count_control"
    protocol_path = output_root / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("model") != 382 or protocol.get("official_valid_or_test_used") is not False:
        raise ValueError("unexpected Model382 baseline protocol")
    folds: dict[str, Any] = {}
    for fold in range(1, 5):
        run_dir = output_root / f"seed1/fold{fold}"
        oof_dir = output_root / f"seed1/fold{fold}_oof"
        args = json.loads((run_dir / "args.json").read_text(encoding="utf-8"))
        expected = {
            "architecture": "UnetPlusPlus",
            "encoder_name": "tu-hrnet_w48",
            "patch_size": 512,
            "input_scale": 2.0,
            "samples_per_image": 12,
            "batch_size": 4,
            "accumulation_steps": 2,
            "num_workers": 4,
            "epochs": 70,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "seed": 1,
            "val_every": 0,
        }
        if any(args.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Model382 seed1 fold{fold} recipe differs from preregistered control")
        manifest_path = oof_dir / "model382_manifest.json"
        archive_path = oof_dir / "valid_probabilities.npz"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint_path = run_dir / "last.pt"
        if manifest.get("seed") != 1 or manifest.get("fold") != fold:
            raise ValueError(f"wrong Model382 seed/fold manifest: {manifest_path}")
        if manifest.get("oof_probabilities_sha256") != sha256(archive_path):
            raise ValueError(f"Model382 OOF archive hash mismatch: {archive_path}")
        if manifest.get("checkpoint_sha256") != sha256(checkpoint_path):
            raise ValueError(f"Model382 checkpoint hash mismatch: {checkpoint_path}")
        folds[str(fold)] = {
            "args_sha256": sha256(run_dir / "args.json"),
            "checkpoint_sha256": sha256(checkpoint_path),
            "oof_manifest_sha256": sha256(manifest_path),
            "oof_probability_sha256": sha256(archive_path),
            "oof_image_count": manifest.get("oof_images"),
            "metric_definition": manifest.get("metric"),
        }
    return {"protocol_sha256": sha256(protocol_path), "folds": folds}


def make_lock(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing lock: {output}")
    artifact_hashes: dict[str, str] = {}
    for relative in LOCKED_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing preregistration artifact: {path}")
        artifact_hashes[relative] = sha256(path)

    base_dir = ROOT / "model336_hrnet_best"
    candidate_dir = ROOT / "model336_hrnet_component_candidate"
    base_source = file_inventory(base_dir)
    candidate_source = file_inventory(candidate_dir)
    if set(base_source) != set(candidate_source):
        raise ValueError("base and candidate source trees have different file inventories")
    changed = {name for name in base_source if base_source[name] != candidate_source[name]}
    if changed != ALLOWED_SOURCE_DIFFS:
        raise ValueError(f"unexpected candidate source diff: {sorted(changed)}")

    cv_root = ROOT / "data_model322_group_cv"
    train_folds: dict[str, Any] = {}
    oof_folds: dict[str, Any] = {}
    oof_union: set[str] = set()
    group_fold: dict[str, int] = {}
    for fold in range(1, 5):
        oof = paired_split_inventory(cv_root / f"oof_fold{fold}", "oof")
        train = paired_split_inventory(cv_root / f"train_fold{fold}", "train")
        for group in oof["group_to_image_ids"]:
            if group in group_fold:
                raise ValueError(f"specimen group occurs in multiple OOF folds: {group}")
            group_fold[group] = fold
        oof_ids = set(oof["image_ids"])
        if oof_union.intersection(oof_ids):
            raise ValueError("image id occurs in multiple OOF folds")
        oof_union.update(oof_ids)
        expected_train_ids = oof_union.copy()
        # Construct the expected training set from all OOF folds except this one.
        expected_train_ids = set()
        for source_fold in range(1, 5):
            if source_fold == fold:
                continue
            source_ids = paired_split_inventory(
                cv_root / f"oof_fold{source_fold}", "oof"
            )["image_ids"]
            expected_train_ids.update(source_ids)
        if set(train["image_ids"]) != expected_train_ids:
            raise ValueError(f"train_fold{fold} is not the complement of oof_fold{fold}")
        train_folds[str(fold)] = train
        oof_folds[str(fold)] = oof
    if len(oof_union) != 70 or len(group_fold) != 59:
        raise ValueError(f"fixed CV inventory mismatch: {len(oof_union)} images / {len(group_fold)} groups")

    return {
        "lock_type": "pre-Model395 train-mask CPU-preflight and paired-OOF protocol lock",
        "model": 395,
        "runtime": runtime_metadata(),
        "artifacts_sha256": artifact_hashes,
        "source_tree_sha256": {
            "model336_hrnet_best": base_source,
            "model336_hrnet_component_candidate": candidate_source,
        },
        "only_candidate_source_diffs": sorted(changed),
        "training_cv_inventory": {"folds": train_folds, "oof_folds": oof_folds},
        "fixed_model382_seed1_baseline": model382_seed1_artifacts(),
        "enforcement": {
            "component_candidate_is_the_only_training_treatment": True,
            "CPU_simulation_must_pass_before_GPU_training": True,
            "OOF_is_selection_conditional_development_evidence": True,
            "official_valid_is_deferred": True,
            "test_is_never_read": True,
            "GPU_utilization_or_memory_is_not_queried": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    result = make_lock(output)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "model": 395,
        "lock_created": str(output),
        "training_started": False,
        "official_valid_or_test_read": False,
        "train_images": 70,
        "specimen_groups": 59,
        "only_candidate_source_diffs": sorted(ALLOWED_SOURCE_DIFFS),
        "folds_locked": 4,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
