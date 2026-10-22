#!/usr/bin/env python3
"""Create the immutable revision-2 Model392 pre-score amendment lock."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_LOCK = "experiments/results/model392_widthnorm_preregistration_lock.json"
OUTPUT = ROOT / "experiments/results/model392_widthnorm_amendment_lock.json"
AMENDMENT = "experiments/model392_widthnorm_amendment.md"
GEOMETRY_AUDIT = "experiments/results/model392_widthnorm_fold_geometry_audit.json"

ANALYSIS_FILES = (
    "experiments/model392_widthnorm_oof.py",
    "experiments/model392_widthnorm_oof_smoke.py",
    "experiments/model392_widthnorm_candidate_smoke.py",
    "experiments/model392_widthnorm_fold_geometry_audit.py",
    "experiments/model392_widthnorm_fold_geometry_audit_smoke.py",
    "experiments/model392_official_valid_once.py",
    "experiments/model392_official_valid_once_smoke.py",
)
EXECUTION_FILES = (
    "experiments/model392_record_fold.py",
    "experiments/model392_prepare_run.py",
    "experiments/model392_prepare_full_train.py",
    "run_model392_full_train_p100.sh",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(paths: list[str] | tuple[str, ...]) -> dict[str, str]:
    result = {}
    for relative in sorted(set(paths)):
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required Model392 locked source is missing: {relative}")
        result[relative] = sha256(path)
    return result


def create_lock() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing Model392 amendment lock: {OUTPUT}")
    base_path = ROOT / BASE_LOCK
    if not base_path.is_file():
        raise FileNotFoundError(f"original Model392 pre-score lock is missing: {base_path}")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base.get("experiment") != "Model392 per-image pixel-scale normalization":
        raise ValueError("unexpected original Model392 lock identity")

    candidate_paths = sorted(base.get("candidate_source_sha256", {}))
    dependency_paths = sorted(base.get("analysis_dependency_sha256", {}))
    if not candidate_paths or "experiments/results/model382_e5_preregistration_lock.json" not in dependency_paths:
        raise ValueError("original lock is missing candidate source or Model382 dependency provenance")
    protocol_paths = [
        base["protocol"]["path"],
        AMENDMENT,
    ]
    geometry_path = ROOT / GEOMETRY_AUDIT
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    if geometry.get("scope") != "fixed train grouped split image filenames and native widths only":
        raise ValueError("unexpected Model392 geometry-audit scope")

    candidate = source_hashes(candidate_paths)
    protocol = source_hashes(protocol_paths)
    dependencies = source_hashes(dependency_paths)
    analysis = source_hashes(ANALYSIS_FILES)
    execution = source_hashes(EXECUTION_FILES)
    geometry_source = source_hashes((GEOMETRY_AUDIT,))
    lock_generator_path = "experiments/create_model392_amendment_lock.py"
    lock_generator = source_hashes((lock_generator_path,))

    runner_name = Path(base["reference_oof_runner"]["path"]).name
    runner_relative = runner_name
    runner_path = ROOT / runner_relative
    runner_hash = sha256(runner_path)
    if runner_hash != base["reference_oof_runner"]["sha256"]:
        raise ValueError("Model382 reference P100 runner differs from its original lock")
    runner = {
        **base["reference_oof_runner"],
        "local_path": runner_relative,
    }

    locked_files = {
        **candidate,
        **protocol,
        **dependencies,
        **analysis,
        **execution,
        **geometry_source,
        **lock_generator,
        runner_relative: runner_hash,
        BASE_LOCK: sha256(base_path),
    }
    if len(locked_files) != (
        len(candidate) + len(protocol) + len(dependencies) + len(analysis)
        + len(execution) + len(geometry_source) + len(lock_generator) + 2
    ):
        # Overlap is allowed only if the exact same path and digest occur in
        # multiple categories; the merged manifest still verifies every path.
        for relative, digest in locked_files.items():
            if sha256(ROOT / relative) != digest:
                raise ValueError(f"conflicting duplicate hash for {relative}")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    receipt = {
        "experiment": "Model392 per-image pixel-scale normalization",
        "revision": 2,
        "locked_at_utc": now,
        "status": "amended pre-score lock; both seeds × four folds required; no Model392 score accessed",
        "supersedes": {"path": BASE_LOCK, "sha256": sha256(base_path)},
        "protocol": {
            "path": AMENDMENT,
            "sha256": protocol[AMENDMENT],
            "supersedes_original": {
                base["protocol"]["path"]: protocol[base["protocol"]["path"]],
            },
        },
        "reference_oof_runner": runner,
        "candidate_source_sha256": candidate,
        "analysis_source_sha256": analysis,
        "analysis_smoke_sha256": {
            key: value for key, value in analysis.items() if key.endswith("_smoke.py")
        },
        "execution_source_sha256": execution,
        "analysis_dependency_sha256": dependencies,
        "lock_generator": {
            "path": lock_generator_path,
            "sha256": lock_generator[lock_generator_path],
        },
        "geometry_audit": {
            "path": GEOMETRY_AUDIT,
            "sha256": geometry_source[GEOMETRY_AUDIT],
            "scope": geometry["scope"],
        },
        "full_train_and_valid_once": {
            "runner": "run_model392_full_train_p100.sh",
            "prepare": "experiments/model392_prepare_full_train.py",
            "valid_evaluator": "experiments/model392_official_valid_once.py",
            "valid_evaluator_smoke": "experiments/model392_official_valid_once_smoke.py",
            "policy": "after passing two-seed OOF gate; train both seeds on 70 train images; evaluate valid exactly once; test is inaccessible to evaluator",
        },
        "source_smokes": {
            "candidate": {
                "path": "experiments/model392_widthnorm_candidate_smoke.py",
                "sha256": analysis["experiments/model392_widthnorm_candidate_smoke.py"],
                "result": "passed synthetic compatibility checks before revision-2 lock",
            },
            "paired_oof": {
                "path": "experiments/model392_widthnorm_oof_smoke.py",
                "sha256": analysis["experiments/model392_widthnorm_oof_smoke.py"],
                "result": "passed two-seed paired analysis, bootstrap, width diagnostics, manifests and resume receipts",
            },
            "geometry": {
                "path": "experiments/model392_widthnorm_fold_geometry_audit_smoke.py",
                "sha256": analysis["experiments/model392_widthnorm_fold_geometry_audit_smoke.py"],
                "result": "passed synthetic metadata-only fold audit",
            },
            "official_valid_once": {
                "path": "experiments/model392_official_valid_once_smoke.py",
                "sha256": analysis["experiments/model392_official_valid_once_smoke.py"],
                "result": "passed empty=0 metric, failed-gate rejection, image/mask ID checks, full-train receipt preparation",
            },
        },
        "locked_files_sha256": locked_files,
        "scope": "hashes of source, protocols, the fixed image-width-only geometry result, and preregistration dependencies; no Model382/Model392 OOF score values, official valid values, or test artifacts read",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return receipt


def main() -> None:
    receipt = create_lock()
    print(json.dumps({
        "lock": str(OUTPUT),
        "revision": receipt["revision"],
        "locked_at_utc": receipt["locked_at_utc"],
        "locked_files": len(receipt["locked_files_sha256"]),
        "lock_sha256": sha256(OUTPUT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
