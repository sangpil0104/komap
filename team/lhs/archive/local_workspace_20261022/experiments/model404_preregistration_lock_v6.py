#!/usr/bin/env python3
"""Create/verify immutable source hashes for Model404's parent-guarded run."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


LOCK_REL = Path("experiments/results/model404_preregistration_lock_v6.json")
PREVIOUS_LOCK_REL = Path("experiments/results/model404_preregistration_lock_v5.json")
FILES = (
    "experiments/model404_component_recall_preregistration_v6.md",
    "experiments/score_model404_component_recall_oof_v6.py",
    "experiments/run_model404_component_recall_oof_v6.sh",
    "experiments/queue_model404_component_recall_v6.sh",
    "experiments/model404_preregistration_lock_v6.py",
    "experiments/model404_component_recall_preregistration_v5.md",
    "experiments/score_model404_component_recall_oof_v5.py",
    "experiments/run_model404_component_recall_oof_v5.sh",
    "experiments/queue_model404_component_recall_v5.sh",
    "experiments/model404_preregistration_lock_v5.py",
    "experiments/model404_tiny_component_confusion_audit.py",
    "experiments/results/model404_tiny_component_confusion_audit.json",
    "experiments/model404_component_recall_preregistration_v4.md",
    "experiments/model404_component_recall_loss_v4.py",
    "experiments/model404_component_recall_smoke_v4.py",
    "experiments/model404_component_recall_train_v4.py",
    "experiments/model404_component_recall_calibrate_v4.py",
    "experiments/score_model404_component_recall_oof_v4.py",
    "experiments/run_model404_component_recall_oof_v4.sh",
    "experiments/queue_model404_component_recall_v4.sh",
    "experiments/model404_preregistration_lock_v4.py",
    "experiments/model404_component_recall_train.py",
    "experiments/model404_component_recall_calibrate.py",
    "experiments/model404_component_recall_loss.py",
    "experiments/score_model404_component_recall_oof.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/checkpoint.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/dataset.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/losses.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/model.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/palette.py",
    "model397_reproduction_bundle/model392/evaluate_image_mean.py",
    "model397_reproduction_bundle/scripts/score_model397_oof.py",
    "model397_reproduction_bundle/scripts/write_oof_manifest.py",
    "experiments/model396_fixed_architecture_oof.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    lock_path = root / LOCK_REL
    previous_path = root / PREVIOUS_LOCK_REL
    if args.create:
        if lock_path.exists():
            raise FileExistsError(f"refusing to replace existing receipt: {lock_path}")
        if not previous_path.is_file():
            raise FileNotFoundError(previous_path)
        files = {}
        for relative in FILES:
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            files[relative] = sha256(path)
        payload = {
            "experiment": "Model404 v6 Al3Ni + Eutectic-Si tiny-component recall with parent non-regression gate",
            "revision": 6,
            "locked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "files_sha256": files,
            "supersedes": {
                "path": str(PREVIOUS_LOCK_REL),
                "sha256": sha256(previous_path),
                "calibration_or_training_started": False,
            },
            "valid_read_before_lock": False,
            "test_read_before_lock": False,
            "training_started_before_lock": False,
        }
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(f"created {lock_path}")
        return

    if not lock_path.is_file():
        raise FileNotFoundError(lock_path)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if (payload.get("experiment") != "Model404 v6 Al3Ni + Eutectic-Si tiny-component recall with parent non-regression gate"
            or payload.get("revision") != 6):
        raise ValueError("unexpected Model404 v6 lock identity")
    previous = payload.get("supersedes", {})
    if (previous.get("path") != str(PREVIOUS_LOCK_REL)
            or previous.get("sha256") != sha256(previous_path)
            or previous.get("calibration_or_training_started") is not False):
        raise ValueError("superseded v5 receipt is missing, altered, or not marked pre-run")
    expected = payload.get("files_sha256", {})
    if set(expected) != set(FILES):
        raise ValueError("Model404 v6 source inventory mismatch")
    for relative, digest in expected.items():
        path = root / relative
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Model404 v6 source hash mismatch: {relative}")
    print(f"verified {len(expected)} locked Model404 v6 sources")


if __name__ == "__main__":
    main()
