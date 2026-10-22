#!/usr/bin/env python3
"""Post-hoc recovery runner for a path-join bug in the locked v2 evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import model382_two_seed_mean_sensitivity as v1
import model382_two_seed_mean_sensitivity_v2 as v2


def normalize_checkpoint_path(path: Path) -> Path:
    """Undo only the v1 bookkeeping typo: seed + 'seed1' -> 'seedseed1'."""
    replacements = {"seedseed1": "seed1", "seedseed2": "seed2"}
    if not any(part in replacements for part in path.parts):
        return path
    return Path(*(replacements.get(part, part) for part in path.parts))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(cv_root: Path, model382_root: Path, lock_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Model382 recovery output: {output}")
    original_hash = v1.sha256
    corrections: list[dict[str, str]] = []

    def corrected_hash(path: Path) -> str:
        path = Path(path)
        corrected = normalize_checkpoint_path(path)
        if corrected != path:
            if path.name != "last.pt" or not corrected.is_file():
                raise FileNotFoundError(f"unexpected malformed checkpoint path: {path}")
            corrections.append({"bugged_path": str(path), "actual_path": str(corrected)})
            path = corrected
        return original_hash(path)

    # Keep the hash correction narrowly scoped to v1's provenance bookkeeping.
    v1.sha256 = corrected_hash
    try:
        result = v2.analyze_v2(cv_root, model382_root, lock_path)
    finally:
        v1.sha256 = original_hash

    repaired_sources = {}
    for name, digest in result.get("source_sha256", {}).items():
        fixed = str(normalize_checkpoint_path(Path(name)))
        if fixed != name and sha256(Path(fixed)) != digest:
            raise ValueError(f"checkpoint provenance hash mismatch after path repair: {fixed}")
        if fixed in repaired_sources:
            raise ValueError(f"duplicate repaired provenance path: {fixed}")
        repaired_sources[fixed] = digest
    result["source_sha256"] = repaired_sources
    result["analysis"] = "secondary development-only fixed equal-probability two-seed mean; v2 path-bookkeeping recovery"
    result["recovery_note"] = (
        "The locked v2/v1 evaluator computed fold-1 per-image metrics but aborted while hashing the first checkpoint: "
        "it constructed seedseed1 instead of seed1. This wrapper repairs only that checkpoint path during source hashing; "
        "metric, predictions, bootstrap, thresholds, gates, and locked source files are unchanged. Some per-image OOF "
        "metrics were inadvertently exposed during progress logging; treat this full result as exploratory, not confirmatory."
    )
    result["recovered_checkpoint_source_paths"] = corrections
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--model382-root", type=Path, required=True)
    parser.add_argument("--lock-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.cv_root, args.model382_root, args.lock_receipt, args.output)
    print(json.dumps({
        "analysis": result["analysis"],
        "secondary_gate_passed": result["secondary_gate_passed"],
        "adoption_gate_passed": result["adoption_gate_passed"],
        "recovery_path_count": len(result["recovered_checkpoint_source_paths"]),
        "official_valid_or_test_read": False,
        "interpretation": "exploratory after partial per-image OOF score exposure",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
