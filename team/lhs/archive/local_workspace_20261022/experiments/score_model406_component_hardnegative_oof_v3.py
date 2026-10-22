#!/usr/bin/env python3
"""One-shot scoring for locked Model406 v3 train-only grouped OOF maps."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import model406_component_hardnegative_v3_lock as lock
import score_model406_component_hardnegative_oof as scorer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--control-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    claim_path = Path(str(args.out) + ".claim")
    if args.out.exists() or claim_path.exists():
        raise FileExistsError(f"refusing to overwrite prior Model406 v3 score: {args.out}")
    lock.verify_oof(root)
    result = scorer.score(
        args.data_root, args.candidate_root, args.control_root, args.baseline_root
    )
    claim = {
        "experiment": "Model406 v3 small-component correction grouped OOF",
        "claimed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "oof_lock_sha256": sha256(root / lock.OOF_LOCK),
        "policy": "one-shot score; claim retained after interruption",
    }
    claim_payload = (json.dumps(claim, sort_keys=True, indent=2) + "\n").encode()
    descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(claim_payload)
        stream.flush()
        os.fsync(stream.fileno())
    result["experiment"] = "Model406 v3 small-component recall with hard-negative island suppression"
    result["source_lock_sha256"] = sha256(root / lock.SOURCE_LOCK)
    result["calibration_lock_sha256"] = sha256(root / "experiments/results/model406_component_hardnegative_calibration_lock_v2.json")
    result["oof_lock_sha256"] = sha256(root / lock.OOF_LOCK)
    result["claim_sha256"] = sha256(claim_path)
    result["valid_read"] = False
    result["test_read"] = False
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "result": str(args.out),
        "candidate_oof_percent": result["candidate_scores"]["image_mean_miou_percent"],
        "matched_control_oof_percent": result["control_scores"]["image_mean_miou_percent"],
        "delta_pp": result["paired_candidate_vs_control"]["delta_image_mean_pp"],
        "gate_pass": result["preregistered_gate"]["pass"],
        "official_valid_used": False,
        "test_used": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
