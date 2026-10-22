#!/usr/bin/env python3
"""Freeze/verify the Model392 prerequisite closure and bind fold receipts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from model382_oof_ensemble_compare import _verify_model382_manifest  # noqa: E402
from model392_widthnorm_oof import (  # noqa: E402
    sha256, verify_candidate_manifest, verify_model392_lock,
)

ADDENDUM = "experiments/model392_sequence_closure_addendum.md"
DRIVE_INVENTORY = "experiments/results/model374_drive_metadata_inventory.json"
OUTPUT = "experiments/results/model392_sequence_closure_lock.json"
MODEL392_LOCK = "experiments/results/model392_widthnorm_runtime_lock.json"
MODEL392_LOCK_SHA256 = "68e7f2d524e338f24d9e88472d1463aadc69d5cb4daf7a56e0d5997edc3ebc98"
MODEL382_LOCK = "experiments/results/model382_e5_preregistration_lock.json"
AUDIT_RESULTS = (
    "experiments/results/model387_resolution_headroom_folds3_4.json",
    "experiments/results/model388_headroom_folds3_4_recovery.json",
    "experiments/results/model395_realized_dose_folds1_4.json",
    "experiments/results/model389_headroom_folds3_4.json",
)
SELF = "experiments/create_model392_sequence_closure_lock.py"
RUNNER = "run_model392_widthnorm_oof_p100.sh"


def utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def record(path: Path, known_hash: str | None = None) -> dict:
    stat = path.stat()
    return {
        "path": str(path.relative_to(ROOT)), "size_bytes": stat.st_size,
        "mtime_utc": utc_iso(stat.st_mtime), "sha256": known_hash or sha256(path),
    }


def get_runtime_lock(root: Path) -> tuple[dict, Path, str]:
    path = root / MODEL392_LOCK
    receipt = verify_model392_lock(path)
    digest = sha256(path)
    if digest != MODEL392_LOCK_SHA256 or receipt.get("revision") != 3:
        raise ValueError("Model392 revision-3 runtime lock differs from the reviewed lock")
    return receipt, path, digest


def collect_references(root: Path, locked_at: float) -> tuple[dict, dict[str, str]]:
    refs: dict[str, dict] = {}
    hashes: dict[str, str] = {}
    for seed in (1, 2):
        for fold in range(1, 5):
            key = f"seed{seed}/fold{fold}"
            run_dir = root / "outputs/model382_seed_count_control" / f"seed{seed}" / f"fold{fold}"
            oof_dir = root / "outputs/model382_seed_count_control" / f"seed{seed}" / f"fold{fold}_oof"
            archive = _verify_model382_manifest(oof_dir, seed, fold, locked_at)
            manifest_path = oof_dir / "model382_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            checkpoint = run_dir / "last.pt"
            args_path = run_dir / "args.json"
            if manifest.get("model") != 382 or manifest.get("seed") != seed or manifest.get("fold") != fold:
                raise ValueError(f"Model382 reference identity mismatch: {manifest_path}")
            paths = {
                "checkpoint": (checkpoint, manifest["checkpoint_sha256"]),
                "training_args": (args_path, None),
                "manifest": (manifest_path, None),
                "oof_probabilities": (archive, manifest["oof_probabilities_sha256"]),
            }
            for path, _ in paths.values():
                if not path.is_file():
                    raise FileNotFoundError(f"missing Model382 reference artifact: {path}")
            for path in (checkpoint, manifest_path, archive):
                if path.stat().st_mtime <= locked_at:
                    raise ValueError(f"stale Model382 score-bearing reference artifact: {path}")
            refs[key] = {}
            for name, (path, digest) in paths.items():
                item = record(path, digest)
                refs[key][name] = item
                hashes[item["path"]] = item["sha256"]
    return refs, hashes


def static_hashes(root: Path, runtime: dict) -> dict[str, str]:
    result = dict(runtime["locked_files_sha256"])
    for relative in (ADDENDUM, DRIVE_INVENTORY, SELF, RUNNER, MODEL392_LOCK,
                     MODEL382_LOCK, *AUDIT_RESULTS):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"sequence-closure source/evidence missing: {relative}")
        digest = sha256(path)
        if relative in result and result[relative] != digest:
            raise ValueError(f"conflicting locked hash for {relative}")
        result[relative] = digest
    return result


def create(root: Path) -> dict:
    output = root / OUTPUT
    if output.exists():
        raise FileExistsError(f"refusing to overwrite sequence-closure lock: {output}")
    if (root / "outputs/model392_widthnorm").exists():
        raise FileExistsError("Model392 output tree exists; cannot create a pre-run lock")
    if (root / "outputs").is_symlink():
        raise ValueError("refusing a symlinked outputs root for the Model392 run")
    gate_output = root / "experiments/results/model392_widthnorm_two_seed_confirmatory.json"
    gate_claim = gate_output.with_suffix(gate_output.suffix + ".claim")
    if gate_output.exists() or gate_claim.exists():
        raise FileExistsError("a Model392 one-shot OOF analysis already exists or was claimed")
    if (root / "outputs/model382_seed_count_control/.model382.lock").exists():
        raise RuntimeError("Model382 lock exists; refusing another GPU sequence")

    runtime, _, runtime_hash = get_runtime_lock(root)
    model382_path = root / MODEL382_LOCK
    model382_lock = json.loads(model382_path.read_text(encoding="utf-8"))
    model382_hash = sha256(model382_path)
    if runtime["analysis_dependency_sha256"].get(MODEL382_LOCK) != model382_hash:
        raise ValueError("Model392 runtime lock is not bound to the current Model382 receipt")
    refs, ref_hashes = collect_references(root, float(model382_lock["locked_at_unix"]))
    files = static_hashes(root, runtime)
    for relative, digest in ref_hashes.items():
        if relative in files and files[relative] != digest:
            raise ValueError(f"conflicting baseline reference hash: {relative}")
        files[relative] = digest

    receipt = {
        "experiment": "Model392 sequence closure", "revision": 1,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"),
        "status": "prerequisite closure only; Model392 candidate score not accessed",
        "decision": {
            "candidate": 392,
            "run_order": [f"seed{s}/fold{f}" for s in (1, 2) for f in range(1, 5)],
            "unavailable_model382_matched_count_contrast": "closed as unestimable; no valid/test or retrained-arm substitution",
            "model390": "deferred; not mixed into Model392",
            "score_interpretation": "development-only and selection-conditional because baseline OOF results were previously exposed",
        },
        "supersedes_sequence_prerequisite_only": {
            "description": "closes the unavailable Model382 matched-count prerequisite; changes no model recipe or numeric gate",
            "model392_runtime_lock": {"path": MODEL392_LOCK, "sha256": runtime_hash},
            "model382_pre_result_lock": {"path": MODEL382_LOCK, "sha256": model382_hash},
        },
        "drive_recovery_metadata": {
            "path": DRIVE_INVENTORY, "sha256": sha256(root / DRIVE_INVENTORY),
            "scope": "names, sizes, timestamps and Drive MD5 only; no archive contents accessed",
        },
        "terminal_audit_result_sha256": {p: sha256(root / p) for p in AUDIT_RESULTS},
        "model382_same_seed_fold_references": refs,
        "reference_metric": "image-wise 4-class mIoU; empty-union=0; 70 train images / 59 specimen groups",
        "gate": "unchanged from Model392 runtime revision 3; all eight runs and held-out-train OOF inference precede the one-shot paired analysis",
        "official_valid_test_policy": "official valid 20-image evaluation only after the unchanged OOF gate passes; no tuning; test excluded",
        "locked_files_sha256": files,
        "evidence_scope": "Model382 manifests were checked and reference files SHA-256 hashed; OOF archives were not decompressed or scored. No Model392 output, official-valid input, or test input was read.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


def verify(root: Path) -> dict:
    receipt = json.loads((root / OUTPUT).read_text(encoding="utf-8"))
    if receipt.get("experiment") != "Model392 sequence closure" or receipt.get("revision") != 1:
        raise ValueError("unexpected Model392 sequence-closure receipt")
    get_runtime_lock(root)
    for relative, expected in receipt["locked_files_sha256"].items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"sequence-closure source/reference hash mismatch: {relative}")
    for key, artifacts in receipt["model382_same_seed_fold_references"].items():
        for item in artifacts.values():
            path = root / item["path"]
            stat = path.stat()
            if stat.st_size != item["size_bytes"] or utc_iso(stat.st_mtime) != item["mtime_utc"]:
                raise ValueError(f"Model382 reference metadata changed: {key}/{item['path']}")
    return receipt


def bind_receipt(root: Path, seed: int, fold: int) -> dict:
    closure_path = root / OUTPUT
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    closure_hash = sha256(closure_path)
    _, _, runtime_hash = get_runtime_lock(root)
    path = root / "outputs/model392_widthnorm" / f"seed{seed}" / f"fold{fold}" / "model392_run_receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("model") != 392 or value.get("seed") != seed or value.get("fold") != fold
            or value.get("model392_lock_sha256") != runtime_hash or not value.get("attempts")):
        raise ValueError("Model392 run receipt does not match revision-3 lock/seed/fold")
    started = datetime.fromisoformat(value["initialized_at_utc"].replace("Z", "+00:00"))
    closed = datetime.fromisoformat(closure["locked_at_utc"].replace("Z", "+00:00"))
    if started <= closed:
        raise ValueError("Model392 run receipt predates sequence-closure lock")
    if value.get("model392_sequence_closure_lock_sha256") not in (None, closure_hash):
        raise ValueError("Model392 run receipt references a different sequence-closure lock")
    value["model392_sequence_closure_lock_sha256"] = closure_hash
    value["model392_sequence_addendum_sha256"] = sha256(root / ADDENDUM)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=path.name + ".sequence.",
        suffix=".tmp", dir=path.parent, delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return {"seed": seed, "fold": fold, "closure_lock_sha256": closure_hash}


def verify_bound_receipt(root: Path, seed: int, fold: int) -> dict:
    closure_path = root / OUTPUT
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    closure_hash = sha256(closure_path)
    _, _, runtime_hash = get_runtime_lock(root)
    path = root / "outputs/model392_widthnorm" / f"seed{seed}" / f"fold{fold}" / "model392_run_receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("model") != 392 or value.get("seed") != seed or value.get("fold") != fold
            or value.get("model392_lock_sha256") != runtime_hash
            or value.get("model392_sequence_closure_lock_sha256") != closure_hash
            or value.get("model392_sequence_addendum_sha256") != sha256(root / ADDENDUM)):
        raise ValueError("Model392 run receipt is not bound to this protocol and sequence closure")
    initialized = datetime.fromisoformat(value["initialized_at_utc"].replace("Z", "+00:00"))
    closed = datetime.fromisoformat(closure["locked_at_utc"].replace("Z", "+00:00"))
    if initialized <= closed:
        raise ValueError("Model392 run receipt predates sequence-closure lock")
    return value


def verify_split(root: Path, seed: int, fold: int) -> dict:
    cv_root = root / "data_model322_group_cv"
    data_root = root / f"data_model382_seed{seed}_fold{fold}"
    train_path = data_root / "train"
    heldout_path = data_root / "valid"
    expected_train = cv_root / f"train_fold{fold}"
    expected_heldout = cv_root / f"oof_fold{fold}"
    if (not train_path.is_dir() or train_path.resolve() != expected_train.resolve()
            or not heldout_path.is_dir() or heldout_path.resolve() != expected_heldout.resolve()):
        raise ValueError(f"seed{seed}/fold{fold} data links do not match the locked grouped split")
    return {"seed": seed, "fold": fold, "train": str(expected_train),
            "heldout_train_oof": str(expected_heldout), "official_valid_or_test": False}


def status(root: Path, seed: int, fold: int) -> str:
    run_dir = root / "outputs/model392_widthnorm" / f"seed{seed}" / f"fold{fold}"
    if not run_dir.exists():
        return "new"
    receipt_path = run_dir / "model392_run_receipt.json"
    if not receipt_path.is_file():
        return "unreceipted"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("model392_sequence_closure_lock_sha256") is None:
        bind_receipt(root, seed, fold)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    verify_bound_receipt(root, seed, fold)
    state_path, checkpoint = run_dir / "train_state.pt", run_dir / "last.pt"
    if not state_path.exists():
        if checkpoint.exists():
            raise ValueError("checkpoint exists without resumable train_state.pt")
        return "incomplete"
    import torch
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    epoch = int(state.get("epoch", -1))
    if epoch < 0 or epoch > 70 or not checkpoint.is_file():
        raise ValueError(f"invalid Model392 state at epoch {epoch}")
    if epoch < 70:
        return "incomplete"
    log = run_dir / "train.log"
    if not log.is_file() or "done. final model:" not in log.read_text(encoding="utf-8", errors="replace"):
        raise RuntimeError("epoch-70 state lacks clean completion marker; refusing inference")
    return "complete"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--create", action="store_true")
    modes.add_argument("--verify", action="store_true")
    modes.add_argument("--bind-run-receipt", action="store_true")
    modes.add_argument("--verify-fold", action="store_true")
    modes.add_argument("--training-status", action="store_true")
    modes.add_argument("--verify-split", action="store_true")
    parser.add_argument("--seed", type=int, choices=(1, 2))
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4))
    args = parser.parse_args()
    root = args.project_root.resolve()
    if args.create:
        receipt = create(root)
        print(json.dumps({"created": OUTPUT, "locked_files": len(receipt["locked_files_sha256"]),
                          "baseline_folds": len(receipt["model382_same_seed_fold_references"]),
                          "candidate_score_accessed": False}, indent=2))
        return
    if args.verify:
        receipt = verify(root)
        print(json.dumps({"verified": OUTPUT, "locked_files": len(receipt["locked_files_sha256"]),
                          "baseline_folds": len(receipt["model382_same_seed_fold_references"]),
                          "Model392_score_accessed": False}, indent=2))
        return
    if args.seed is None or args.fold is None:
        parser.error("--seed and --fold are required for this mode")
    if args.bind_run_receipt:
        result = bind_receipt(root, args.seed, args.fold)
    elif args.training_status:
        result = {"seed": args.seed, "fold": args.fold,
                  "training_status": status(root, args.seed, args.fold)}
    elif args.verify_split:
        result = verify_split(root, args.seed, args.fold)
    else:
        verify_bound_receipt(root, args.seed, args.fold)
        _, lock_path, lock_hash = get_runtime_lock(root)
        verify_candidate_manifest(
            root,
            root / "outputs/model392_widthnorm" / f"seed{args.seed}" / f"fold{args.fold}",
            root / "outputs/model392_widthnorm" / f"seed{args.seed}" / f"fold{args.fold}_oof",
            args.seed, args.fold, lock_path, lock_hash,
        )
        result = {"seed": args.seed, "fold": args.fold, "candidate_manifest_verified": True}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
