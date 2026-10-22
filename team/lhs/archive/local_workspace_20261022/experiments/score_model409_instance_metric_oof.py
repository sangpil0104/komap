#!/usr/bin/env python3
"""One-shot Model409 grouped train-OOF scoring against Model406 control."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path

import model409_instance_metric_lock as lock
import score_model406_component_hardnegative_oof as score406


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
        raise FileExistsError(f"refusing to overwrite one-shot Model409 result: {args.out}")
    oof_lock = lock.verify_oof_lock(root)
    claim = {
        "experiment": "Model409 v2 tiny-component local feature metric grouped train-OOF",
        "claimed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "oof_lock_sha256": lock.sha256(root / lock.OOF_LOCK),
        "policy": "one-shot; claim retained after interruption",
    }
    descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(claim, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    result = score406.score(args.data_root, args.candidate_root, args.control_root, args.baseline_root)
    result["experiment"] = "Model409 v2 tiny-component local feature metric fine-tuning"
    result["objective"] = (
        "training-only component-instance contrast: same tiny GT connected component positives "
        "vs nearby 2-6px fixed confuser-class negatives; projection head discarded at inference"
    )
    result["local_metric"] = {
        "temperature": 0.2,
        "positive_definition": "same 8-connected GT component only",
        "negative_definition": "fixed Al/Al3Ni confuser labels in 2-6 augmented-pixel annulus",
        "classes": {"Al3Ni": ["Al"], "Eutectic Si": ["Al", "Al3Ni"]},
        "official_valid_read": False,
        "test_read": False,
    }
    result["status"] = "OOF_SCREEN_PASS" if result["preregistered_gate"]["pass"] else "NO_GO"
    result["valid_read"] = False
    result["test_read"] = False
    result["valid_stage_allowed"] = False
    result["model409_source_lock_sha256"] = lock.sha256(root / lock.SOURCE_LOCK)
    result["model409_calibration_lock_sha256"] = lock.sha256(root / lock.CALIBRATION_LOCK)
    result["model409_oof_lock_sha256"] = lock.sha256(root / lock.OOF_LOCK)
    result["model406_reference_oof_lock_sha256"] = oof_lock["reference_model406_oof_lock_sha256"]
    result["claim_sha256"] = lock.sha256(claim_path)
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
        "group_bootstrap_95ci_pp": result["paired_candidate_vs_control"]["paired_group_bootstrap_95ci_pp"],
        "gate_pass": result["preregistered_gate"]["pass"],
        "official_valid_used": False,
        "test_used": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
