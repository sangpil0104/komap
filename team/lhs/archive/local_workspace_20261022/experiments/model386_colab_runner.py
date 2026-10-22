#!/usr/bin/env python3
"""One-off Colab CPU runner for the exploratory Model386 folds1-2 pilot."""

from __future__ import annotations

import json
import sys
import tarfile
import hashlib
from pathlib import Path

sys.path.insert(0, "/content")

from model386_step4_eutectic_distance_rate_pilot import analyze


ROOT = Path("/content/model386_f12_inputs_deref")
ARCHIVE = Path("/content/model386_f12_argmax_bundle_deref.tar.gz")
OUTPUT = Path("/content/model386_step4_eutectic_distance_rate_pilot.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, mode="r:gz") as tar:
        for member in tar.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe archive member: {member.name}")
        tar.extractall(ROOT, filter="data")

    try:
        result = analyze(ROOT, ROOT, folds=(1, 2), draws=20_000)
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        for fold in (1, 2):
            record = manifest["folds"][str(fold)]
            archive_path = ROOT / record["prediction_archive"]
            if sha256(archive_path) != record["argmax_predictions_sha256"]:
                raise ValueError(f"fold{fold} argmax archive checksum mismatch")
    except Exception as exc:
        print(json.dumps({
            "diagnostic_error_type": type(exc).__name__,
            "diagnostic_error": str(exc),
        }))
        return
    result["input_manifest"] = manifest
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
