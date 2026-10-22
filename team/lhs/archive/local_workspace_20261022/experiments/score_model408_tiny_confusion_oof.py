#!/usr/bin/env python3
"""One-shot paired grouped-OOF evaluation for Model408."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import model406_component_hardnegative_v3_lock as m406lock
import score_model406_component_hardnegative_oof as m406score
import model408_tiny_confusion_lock as m408lock


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_claim(path: Path, oof_lock_hash: str) -> str:
    record = {
        "experiment": "Model408 tiny-GT confusion hinge grouped OOF",
        "claimed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "oof_lock_sha256": oof_lock_hash,
        "policy": "one-shot; claim retained after interruption",
    }
    payload = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(path)


def main() -> None:
    import argparse

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
        raise FileExistsError(f"refusing to overwrite prior Model408 score: {args.out}")

    lock = m408lock.verify_oof_lock(root)
    claim_hash = create_claim(claim_path, sha256(root / m408lock.OOF_LOCK))
    result = m406score.score(
        args.data_root, args.candidate_root, args.control_root, args.baseline_root
    )
    result["experiment"] = "Model408 tiny-GT confusion hinge fine-tuning"
    result["objective"] = (
        "component-normalized zero-margin hinge on misclassified augmented-grid "
        "1-64px-proxy Al3Ni/Eutectic-Si GT components; no prediction-side FP loss"
    )
    result["fixed_confusers"] = {"Al3Ni": ["Al"], "Eutectic Si": ["Al", "Al3Ni"]}
    result["status"] = "OOF_SCREEN_PASS" if result["preregistered_gate"]["pass"] else "NO_GO"
    result["claim_sha256"] = claim_hash
    result["source_lock_sha256"] = sha256(root / m408lock.SOURCE_LOCK)
    result["calibration_lock_sha256"] = sha256(root / m408lock.CALIBRATION_LOCK)
    result["oof_lock_sha256"] = sha256(root / m408lock.OOF_LOCK)
    result["model406_reference_oof_lock_sha256"] = lock["reference_model406_oof_lock_sha256"]
    result["valid_read"] = False
    result["test_read"] = False
    result["valid_stage_allowed"] = False
    result["decision_note"] = (
        "No official validation/test look is authorized. A passing result only permits an "
        "independently locked seed-2 grouped-OOF confirmation."
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "result": str(args.out),
        "status": result["status"],
        "candidate_oof_percent": result["candidate_scores"]["image_mean_miou_percent"],
        "matched_control_oof_percent": result["control_scores"]["image_mean_miou_percent"],
        "delta_pp": result["paired_candidate_vs_control"]["delta_image_mean_pp"],
        "gate_pass": result["preregistered_gate"]["pass"],
        "official_valid_used": False,
        "test_used": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
