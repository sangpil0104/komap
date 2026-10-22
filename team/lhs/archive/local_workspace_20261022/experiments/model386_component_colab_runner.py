#!/usr/bin/env python3
"""Run the exploratory Model386 component audit on a temporary Colab CPU."""

from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, "/content")

from model386_step4_interior_component_audit import analyze


ROOT = Path("/content/model386_component_inputs")
ARCHIVE = Path("/content/model386_f12_argmax_bundle.tar.gz")
OUTPUT = Path("/content/model386_interior_component_audit.json")


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
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    for fold in (1, 2):
        record = manifest["folds"][str(fold)]
        path = ROOT / record["prediction_archive"]
        if sha256(path) != record["argmax_predictions_sha256"]:
            raise ValueError(f"fold{fold} argmax archive checksum mismatch")
    result = analyze(ROOT, ROOT)
    result["input_manifest"] = manifest
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("per_image", "per_component")}, indent=2))


if __name__ == "__main__":
    main()
