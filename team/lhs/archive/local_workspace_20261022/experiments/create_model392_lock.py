#!/usr/bin/env python3
"""Create the one-time Model392 pre-score source/data-protocol lock."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "experiments/results/model392_widthnorm_preregistration_lock.json"
PROTOCOL = "experiments/model392_widthnorm_preregistration.md"
CANDIDATE_FILES = (
    "model336_hrnet_widthnorm_candidate/README.md",
    "model336_hrnet_widthnorm_candidate/evaluate_image_mean.py",
    "model336_hrnet_widthnorm_candidate/predict.py",
    "model336_hrnet_widthnorm_candidate/requirements.txt",
    "model336_hrnet_widthnorm_candidate/train.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/__init__.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/checkpoint.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/dataset.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/inference.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/losses.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/metrics.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/model.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/palette.py",
)
DEPENDENCY_FILES = (
    "experiments/model382_oof_ensemble_compare.py",
    "experiments/results/model382_e5_preregistration_lock.json",
    "experiments/model390_per_item_rng_oof.py",
    "experiments/results/model390_preregistration_lock.json",
    "experiments/model391_physical_scale_mask_audit.md",
    "experiments/model391_physical_scale_mask_audit.py",
    "experiments/model391_physical_scale_mask_audit_smoke.py",
    "experiments/model391_interpretation_addendum.md",
    "experiments/results/model391_physical_scale_mask_audit.json",
    "experiments/results/model391_physical_scale_mask_audit_lock.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(paths: tuple[str, ...]) -> dict[str, str]:
    result = {}
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required pre-score source is missing: {relative}")
        result[relative] = sha256(path)
    return result


def main() -> None:
    if LOCK_PATH.exists():
        raise FileExistsError(f"refusing to replace an existing pre-score lock: {LOCK_PATH}")

    protocol = ROOT / PROTOCOL
    candidate = source_hashes(CANDIDATE_FILES)
    dependencies = source_hashes(DEPENDENCY_FILES)
    analysis = source_hashes(("experiments/model392_widthnorm_oof.py",))
    analysis_smoke = source_hashes(("experiments/model392_widthnorm_oof_smoke.py",))
    execution = source_hashes(("experiments/model392_record_fold.py",))
    lock_generator = source_hashes(("experiments/create_model392_lock.py",))
    candidate_smoke_path = ROOT / "experiments/model392_widthnorm_candidate_smoke.py"
    if not candidate_smoke_path.is_file():
        raise FileNotFoundError(candidate_smoke_path)

    prior = json.loads((ROOT / "experiments/results/model390_preregistration_lock.json").read_text())
    runner = prior["reference_oof_runner"]
    local_runner = ROOT / Path(runner["path"]).name
    runner_hash = sha256(local_runner)
    if runner_hash != runner["sha256"]:
        raise ValueError("local Model382 reference runner differs from the previously locked source")
    runner["local_path"] = str(local_runner.relative_to(ROOT))

    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    receipt = {
        "experiment": "Model392 per-image pixel-scale normalization",
        "locked_at_utc": now,
        "status": "pre-score preregistration; Model392 training/OOF not started",
        "protocol": {"path": PROTOCOL, "sha256": sha256(protocol)},
        "reference_oof_runner": runner,
        "candidate_source_sha256": candidate,
        "synthetic_smoke": {
            "path": "experiments/model392_widthnorm_candidate_smoke.py",
            "sha256": sha256(candidate_smoke_path),
            "environment": "P100 CPU execution, CUDA_VISIBLE_DEVICES empty",
            "result": "passed synthetic crop compatibility, width-mode geometry, per-image tile scaling, native probability-map shape, checkpoint metadata, and no-validation guard",
        },
        "analysis_source_sha256": analysis,
        "analysis_smoke_sha256": analysis_smoke,
        "execution_source_sha256": execution,
        "lock_generator": {
            "path": "experiments/create_model392_lock.py",
            "sha256": lock_generator["experiments/create_model392_lock.py"],
        },
        "analysis_dependency_sha256": dependencies,
        "scope": "only source/protocol hashes and synthetic smoke metadata; no Model382 OOF scores, official validation, or test artifacts read",
    }
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "lock": str(LOCK_PATH),
        "locked_at_utc": now,
        "candidate_sources": len(candidate),
        "dependencies": len(dependencies),
        "lock_sha256": sha256(LOCK_PATH),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
