#!/usr/bin/env python3
"""One-shot paired grouped-OOF evaluation for Model407."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import model407_component_fp_lock as m407lock
import score_model406_component_hardnegative_oof as m406score


ROOT = Path(__file__).resolve().parents[1]
M406_CANDIDATE_ROOT = ROOT / m407lock.M406_CANDIDATE_ROOT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_claim(path: Path, oof_lock_hash: str) -> str:
    record = {
        "experiment": "Model407 prediction-component false-positive suppression grouped OOF",
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
        raise FileExistsError(f"refusing to overwrite prior Model407 score: {args.out}")
    lock = m407lock.verify_oof_lock(root)
    claim_hash = create_claim(claim_path, sha256(root / m407lock.OOF_LOCK))

    primary = m406score.score(
        args.data_root, args.candidate_root, args.control_root, args.baseline_root
    )
    prior = m406score.score(
        args.data_root, args.candidate_root, M406_CANDIDATE_ROOT, args.baseline_root
    )
    prior_fp = m406score.component_fp_diagnostics(
        args.candidate_root, M406_CANDIDATE_ROOT
    )

    result = primary
    result["experiment"] = "Model407 GT-component recall plus prediction-component FP suppression"
    result["objective"] = (
        "Model404 v4 balanced GT-component recall + count-sensitive penalty on detached-argmax "
        "8-connected target-class components outside same-class GT halo and crop border"
    )
    result["status"] = (
        "OOF_SCREEN_PASS" if result["preregistered_gate"]["pass"] else "NO_GO"
    )
    result["reference_model406_v3"] = {
        "oof_lock_sha256": lock["reference_model406_oof_lock_sha256"],
        "comparison_vs_pixel_hard_negative_candidate": {
            "candidate_image_mean_miou_percent": prior["candidate_scores"]["image_mean_miou_percent"],
            "reference_image_mean_miou_percent": prior["control_scores"]["image_mean_miou_percent"],
            "delta_image_mean_pp": prior["paired_candidate_vs_control"]["delta_image_mean_pp"],
            "paired_group_bootstrap_95ci_pp": prior["paired_candidate_vs_control"]["paired_group_bootstrap_95ci_pp"],
            "positive_folds": prior["paired_candidate_vs_control"]["positive_folds"],
        },
        "tiny_component_diagnostics_vs_pixel_hard_negative_candidate": prior_fp["classes"],
        "descriptive_only_not_an_adoption_gate": True,
    }
    result["claim_sha256"] = claim_hash
    result["source_lock_sha256"] = sha256(root / m407lock.SOURCE_LOCK)
    result["calibration_lock_sha256"] = sha256(root / m407lock.CALIBRATION_LOCK)
    result["oof_lock_sha256"] = sha256(root / m407lock.OOF_LOCK)
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
