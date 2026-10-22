#!/usr/bin/env python3
"""Create the immutable revision-3 Model392 runtime-safety lock."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_LOCK = "experiments/results/model392_widthnorm_amendment_lock.json"
PREVIOUS_LOCK_SHA256 = "ee043caf1221afc5be07536f8f6c0cc02082cc51d44dbbd515a73f34e0277ebd"
ORIGINAL_LOCK = "experiments/results/model392_widthnorm_preregistration_lock.json"
OUTPUT = ROOT / "experiments/results/model392_widthnorm_runtime_lock.json"
ADDENDUM = "experiments/model392_runtime_safety_addendum.md"
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


def create_lock(output: Path = OUTPUT) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Model392 runtime lock: {output}")
    previous_path = ROOT / PREVIOUS_LOCK
    if not previous_path.is_file() or sha256(previous_path) != PREVIOUS_LOCK_SHA256:
        raise ValueError("revision-2 Model392 lock is missing or changed")
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    if previous.get("revision") != 2 or previous.get("experiment") != (
        "Model392 per-image pixel-scale normalization"
    ):
        raise ValueError("unexpected superseded Model392 lock")

    candidate_paths = sorted(previous.get("candidate_source_sha256", {}))
    dependency_paths = sorted(previous.get("analysis_dependency_sha256", {}))
    if (not candidate_paths
            or "experiments/results/model382_e5_preregistration_lock.json" not in dependency_paths):
        raise ValueError("revision-2 lock lacks candidate source or Model382 dependency provenance")
    if previous.get("reference_oof_runner", {}).get("sha256") != sha256(
        ROOT / previous["reference_oof_runner"]["local_path"]
    ):
        raise ValueError("Model382 reference P100 runner differs from the locked receipt")

    candidate = source_hashes(candidate_paths)
    if candidate != previous["candidate_source_sha256"]:
        raise ValueError("Model392 candidate implementation differs from the revision-2 lock")
    original_protocol_paths = tuple(previous["protocol"]["supersedes_original"])
    protocol_paths = (*original_protocol_paths, previous["protocol"]["path"], ADDENDUM)
    protocol = source_hashes(protocol_paths)
    dependencies = source_hashes(dependency_paths)
    if dependencies != previous["analysis_dependency_sha256"]:
        raise ValueError("Model392 preregistration dependencies differ from the revision-2 lock")
    analysis = source_hashes(ANALYSIS_FILES)
    execution = source_hashes(EXECUTION_FILES)
    geometry_source = source_hashes((GEOMETRY_AUDIT,))
    if geometry_source[GEOMETRY_AUDIT] != previous["geometry_audit"]["sha256"]:
        raise ValueError("Model392 geometry audit differs from the revision-2 lock")
    generator_path = "experiments/create_model392_runtime_lock.py"
    generator = source_hashes((generator_path,))

    runner = {
        **previous["reference_oof_runner"],
        "local_path": previous["reference_oof_runner"]["local_path"],
    }
    runner_relative = runner["local_path"]
    runner_hash = sha256(ROOT / runner_relative)
    runner["sha256"] = runner_hash

    superseded = {
        PREVIOUS_LOCK: sha256(previous_path),
        ORIGINAL_LOCK: sha256(ROOT / ORIGINAL_LOCK),
    }
    if superseded[ORIGINAL_LOCK] != previous["supersedes"]["sha256"]:
        raise ValueError("original Model392 pre-score lock differs from revision-2 provenance")
    locked_files = {
        **candidate,
        **protocol,
        **dependencies,
        **analysis,
        **execution,
        **geometry_source,
        **generator,
        **superseded,
        runner_relative: runner_hash,
    }
    geometry = json.loads((ROOT / GEOMETRY_AUDIT).read_text(encoding="utf-8"))
    if geometry.get("scope") != "fixed train grouped split image filenames and native widths only":
        raise ValueError("unexpected Model392 geometry-audit scope")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    receipt = {
        "experiment": "Model392 per-image pixel-scale normalization",
        "revision": 3,
        "locked_at_utc": now,
        "status": "runtime-safety pre-score lock; both seeds × four folds required; no Model392 score accessed",
        "supersedes": {"path": PREVIOUS_LOCK, "sha256": sha256(previous_path)},
        "protocol": {
            "path": ADDENDUM,
            "sha256": protocol[ADDENDUM],
            "supersedes_original": {
                previous["protocol"]["path"]: protocol[previous["protocol"]["path"]],
                **{
                    path: protocol[path]
                    for path in previous["protocol"]["supersedes_original"]
                },
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
        "lock_generator": {"path": generator_path, "sha256": generator[generator_path]},
        "geometry_audit": {
            "path": GEOMETRY_AUDIT,
            "sha256": geometry_source[GEOMETRY_AUDIT],
            "scope": geometry["scope"],
        },
        "full_train_and_valid_once": previous["full_train_and_valid_once"],
        "source_smokes": {
            "candidate": {
                "path": "experiments/model392_widthnorm_candidate_smoke.py",
                "sha256": analysis["experiments/model392_widthnorm_candidate_smoke.py"],
                "result": "synthetic compatibility smoke passed under revision-2 lock; candidate unchanged",
            },
            "paired_oof": {
                "path": "experiments/model392_widthnorm_oof_smoke.py",
                "sha256": analysis["experiments/model392_widthnorm_oof_smoke.py"],
                "result": "revision-3 synthetic paired-gate, pooled-fold gate, one-shot claim, geometry, and resume tests passed",
            },
            "geometry": {
                "path": "experiments/model392_widthnorm_fold_geometry_audit_smoke.py",
                "sha256": analysis["experiments/model392_widthnorm_fold_geometry_audit_smoke.py"],
                "result": "synthetic metadata-only fold audit passed under revision-2 lock; audit unchanged",
            },
            "official_valid_once": {
                "path": "experiments/model392_official_valid_once_smoke.py",
                "sha256": analysis["experiments/model392_official_valid_once_smoke.py"],
                "result": "revision-3 synthetic metric, claim exclusivity, gate, IDs, full-train resume/finalize tests passed",
            },
        },
        "locked_files_sha256": locked_files,
        "scope": "hashes of current source, protocols, fixed image-width-only geometry metadata, and preregistration dependencies; no Model392 OOF score, official valid score, or test artifact read",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    receipt = create_lock(args.output)
    print(json.dumps({
        "lock": str(args.output.resolve()),
        "revision": receipt["revision"],
        "locked_at_utc": receipt["locked_at_utc"],
        "locked_files": len(receipt["locked_files_sha256"]),
        "lock_sha256": sha256(args.output.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
