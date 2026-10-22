#!/usr/bin/env python3
"""V2 preflight wrapper for the locked Model382 two-seed sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model382_oof_ensemble_compare import _verify_model382_manifest
from model382_two_seed_mean_sensitivity import analyze as analyze_v1


ROOT = Path(__file__).resolve().parents[1]
LOCKED_FILES = (
    "experiments/model382_two_seed_mean_sensitivity_v2_addendum.md",
    "experiments/model382_two_seed_mean_sensitivity_v2.py",
    "experiments/model382_two_seed_mean_sensitivity_v2_smoke.py",
    "experiments/model382_two_seed_mean_sensitivity_v2_lock.py",
    "experiments/model382_two_seed_mean_sensitivity_preregistration.md",
    "experiments/model382_two_seed_mean_sensitivity.py",
    "experiments/model382_two_seed_mean_sensitivity_smoke.py",
    "experiments/model382_two_seed_mean_sensitivity_lock.py",
    "experiments/model382_oof_ensemble_compare.py",
    "experiments/model382_seed_count_control_preregistration.md",
    "experiments/model382_five_model_ensemble_addendum.md",
    "experiments/results/model382_e5_preregistration_lock.json",
    "experiments/results/model382_two_seed_mean_preregistration_lock.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_metadata() -> dict[str, Any]:
    versions = {}
    for package in ("numpy", "Pillow", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "evaluation_started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "package_versions": versions,
        "device_policy": "CPU only; no GPU inference",
    }


def verify_v2_lock(lock_path: Path) -> dict[str, Any]:
    receipt = json.loads(lock_path.read_text(encoding="utf-8"))
    if receipt.get("analysis") != "model382_two_seed_mean_sensitivity_v2":
        raise ValueError("unexpected v2 lock identity")
    if receipt.get("seed1_oof_scores_read_before_v2_lock") is not False:
        raise ValueError("v2 receipt does not attest unopened seed1 scores")
    if receipt.get("seed2_oof_scores_read_before_v2_lock") is not False:
        raise ValueError("v2 receipt does not attest unopened seed2 scores")
    for relative in LOCKED_FILES:
        expected = receipt.get("source_sha256", {}).get(relative)
        path = ROOT / relative
        if not expected or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"v2 source hash mismatch: {relative}")
    v1_receipt = ROOT / "experiments/results/model382_two_seed_mean_preregistration_lock.json"
    if receipt.get("v1_lock_sha256") != sha256(v1_receipt):
        raise ValueError("v1 secondary lock changed since v2 lock")
    return receipt


def verify_masks(cv_root: Path, receipt: dict[str, Any]) -> dict[str, str]:
    from model382_two_seed_mean_sensitivity_v2_lock import inventory_masks

    actual_hashes, actual_inventory = inventory_masks(cv_root)
    if actual_inventory != receipt.get("cv_root_inventory"):
        raise ValueError("fixed train-OOF mask inventory differs from the v2 lock")
    if actual_hashes != receipt.get("fixed_train_oof_mask_sha256"):
        raise ValueError("one or more fixed train-OOF mask hashes differ from the v2 lock")
    return actual_hashes


def preflight(cv_root: Path, model382_root: Path, lock_path: Path) -> dict[str, Any]:
    receipt = verify_v2_lock(lock_path)
    mask_hashes = verify_masks(cv_root, receipt)
    original_receipt = json.loads(
        (ROOT / "experiments/results/model382_e5_preregistration_lock.json").read_text(encoding="utf-8")
    )
    distinct_hashes: dict[str, dict[str, str]] = {}
    for fold in range(1, 5):
        archives: dict[int, Path] = {}
        checkpoints: dict[int, Path] = {}
        for seed in (1, 2):
            oof_dir = model382_root / f"seed{seed}/fold{fold}_oof"
            archive = _verify_model382_manifest(
                oof_dir, seed, fold, float(original_receipt["locked_at_unix"])
            )
            manifest_path = oof_dir / "model382_manifest.json"
            checkpoint = model382_root / f"seed{seed}/fold{fold}/last.pt"
            if seed == 2:
                lock_time = float(receipt["locked_at_unix"])
                for path in (archive, manifest_path, checkpoint):
                    if path.stat().st_mtime <= lock_time:
                        raise ValueError(f"seed2 artifact is not newer than v2 lock: {path}")
            archives[seed] = archive
            checkpoints[seed] = checkpoint
        fold_hashes = {
            "seed1_archive": sha256(archives[1]),
            "seed2_archive": sha256(archives[2]),
            "seed1_checkpoint": sha256(checkpoints[1]),
            "seed2_checkpoint": sha256(checkpoints[2]),
        }
        if fold_hashes["seed1_archive"] == fold_hashes["seed2_archive"]:
            raise ValueError(f"seed1/seed2 probability archives are identical for fold {fold}")
        if fold_hashes["seed1_checkpoint"] == fold_hashes["seed2_checkpoint"]:
            raise ValueError(f"seed1/seed2 checkpoints are identical for fold {fold}")
        distinct_hashes[str(fold)] = fold_hashes
    return {
        "mask_sha256": mask_hashes,
        "seed_artifact_sha256": distinct_hashes,
    }


def analyze_v2(cv_root: Path, model382_root: Path, lock_path: Path) -> dict[str, Any]:
    receipt = verify_v2_lock(lock_path)
    provenance = preflight(cv_root, model382_root, lock_path)
    # The v1 evaluator performs the scoring only after all v2 preflight checks.
    result = analyze_v1(cv_root, model382_root, ROOT / "experiments/results/model382_two_seed_mean_preregistration_lock.json")
    result["analysis"] = "secondary development-only fixed equal-probability two-seed mean (v2 audited)"
    result["v2_lock_sha256"] = sha256(lock_path)
    result["v2_locked_at_utc"] = receipt["locked_at_utc"]
    result["v2_mask_inventory"] = receipt["cv_root_inventory"]
    result["v2_mask_sha256"] = provenance["mask_sha256"]
    result["v2_seed_artifact_sha256"] = provenance["seed_artifact_sha256"]
    result["runtime_metadata"] = runtime_metadata()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--model382-root", type=Path, required=True)
    parser.add_argument("--lock-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_v2(args.cv_root, args.model382_root, args.lock_receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "analysis": result["analysis"],
        "secondary_gate_passed": result["secondary_gate_passed"],
        "adoption_gate_passed": result["adoption_gate_passed"],
        "mask_count": len(result["v2_mask_sha256"]),
        "all_seed_artifact_pairs_distinct": True,
        "official_valid_or_test_read": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
