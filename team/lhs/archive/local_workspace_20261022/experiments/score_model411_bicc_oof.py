#!/usr/bin/env python3
"""One-shot paired grouped-OOF score for Model411; never accesses official valid/test."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import model406_component_hardnegative_v3_lock as m406lock
import score_model406_component_hardnegative_oof as m406score
import model411_bicc_lock as m411lock


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments/results/model411_bicc_oof.json"
CLAIM = Path(str(RESULT) + ".claim")
CANDIDATE_ROOT = ROOT / "outputs/model411_bicc/candidate"
CONTROL_ROOT = ROOT / "outputs/model406_component_hardnegative_v3/control"
BASELINE_ROOT = ROOT / "outputs/model392_widthnorm"
DATA_ROOT = ROOT / "data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_claim(oof_lock_hash: str) -> str:
    record = {
        "experiment": "Model411 component-balanced bidirectional small-island OOF",
        "claimed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "oof_lock_sha256": oof_lock_hash,
        "policy": "one-shot; claim retained after interruption",
    }
    payload = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()
    descriptor = os.open(CLAIM, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(CLAIM)


def score_once() -> dict:
    if RESULT.exists() or CLAIM.exists():
        raise FileExistsError(f"refusing to overwrite Model411 one-shot result/claim: {RESULT}")
    oof_lock = m411lock.verify_oof(ROOT)
    m406lock.verify_oof(ROOT)
    claim_hash = create_claim(sha256(ROOT / m411lock.OOF_LOCK))

    result = m406score.score(DATA_ROOT, CANDIDATE_ROOT, CONTROL_ROOT, BASELINE_ROOT)
    checks = result["preregistered_gate"]["checks"]
    classes = result["paired_candidate_vs_control"]["class_delta_pp"]
    class_safety = {name: float(value) >= -0.30 for name, value in classes.items()}
    checks["all_class_delta_ge_minus_0_30pp"] = all(class_safety.values())

    tiny_hit_keys = ("Al3Ni_tiny_1_to_lt64", "Eutectic Si_tiny_1_to_lt64")
    hit_rates = result["native_targeted_component_hit_rates"]
    tiny_hit_safety = {
        name: float(hit_rates[name]["delta_pp"]) >= -0.30
        for name in tiny_hit_keys
    }
    checks["tiny_gt_hit_rate_delta_ge_minus_0_30pp_both_targets"] = all(
        tiny_hit_safety.values()
    )

    fp_classes = result["small_island_false_positive_diagnostics"]["classes"]
    target_fp_safety = {
        name: float(fp_classes[name]["candidate_vs_control"][
            "target_fp_pixels_per_image_delta"
        ]) <= 0.0
        for name in ("Al3Ni", "Eutectic Si")
    }
    checks["target_fp_pixel_burden_nonincreasing_both_targets"] = all(
        target_fp_safety.values()
    )
    result["preregistered_gate"]["pass"] = all(checks.values())
    result["status"] = "OOF_SCREEN_PASS" if result["preregistered_gate"]["pass"] else "NO_GO"
    result["experiment"] = "Model411 component-balanced bidirectional small-island loss"
    result["objective"] = (
        "0.75 mean-reduced GT small-component recall + 0.25 mean-reduced unmatched "
        "argmax predicted-component suppression, with fixed CE+Dice+boundary main loss"
    )
    result["scorer_revision"] = (
        "Model406 v6 paired grouped-OOF scorer plus preregistered all-class, tiny-GT-hit, "
        "and target-FP-pixel safety gates"
    )
    result["additional_fixed_safety_gates"] = {
        "all_class_delta_ge_minus_0_30pp": class_safety,
        "tiny_gt_hit_rate_delta_ge_minus_0_30pp": tiny_hit_safety,
        "target_fp_pixel_burden_nonincreasing": target_fp_safety,
    }
    result["source_lock_sha256"] = sha256(ROOT / m411lock.SOURCE_LOCK)
    result["calibration_lock_sha256"] = sha256(ROOT / m411lock.CALIBRATION_LOCK)
    result["oof_lock_sha256"] = sha256(ROOT / m411lock.OOF_LOCK)
    result["claim_sha256"] = claim_hash
    result["valid_read"] = False
    result["test_read"] = False
    result["valid_stage_allowed"] = False
    result["decision_note"] = (
        "Train-only grouped OOF screen. No official validation/test look is authorized; "
        "a pass permits only a separately locked seed-2 train-OOF confirmation."
    )
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with RESULT.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = score_once()
    print(json.dumps({
        "result": str(RESULT),
        "status": result["status"],
        "candidate_image_mean_miou_percent": result["candidate_scores"]["image_mean_miou_percent"],
        "control_image_mean_miou_percent": result["control_scores"]["image_mean_miou_percent"],
        "delta_pp": result["paired_candidate_vs_control"]["delta_image_mean_pp"],
        "paired_group_bootstrap_95ci_pp": result["paired_candidate_vs_control"]["paired_group_bootstrap_95ci_pp"],
        "official_valid_read": False,
        "test_read": False,
        "valid_stage_allowed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
