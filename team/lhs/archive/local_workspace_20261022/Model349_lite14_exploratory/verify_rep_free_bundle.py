#!/usr/bin/env python3
"""Check the 15-checkpoint lightweight Model349 bundle after packaging."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--expected-count", type=int, default=15)
    args = parser.parse_args()
    root = args.bundle.resolve()
    manifest = json.loads((root / "provenance/weights_manifest.json").read_text())
    listed = {entry["path"] for entry in manifest}
    actual = {str(path.relative_to(root)) for path in (root / "weights").rglob("*.pt")}
    errors = []
    if len(manifest) != args.expected_count or listed != actual:
        errors.append(f"Expected exactly {args.expected_count} listed and present checkpoints")
    for entry in manifest:
        path = root / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            errors.append(f"Checkpoint mismatch: {entry['path']}")
    args_files = list((root / "weights").rglob("args.json"))
    if len(args_files) != 8:
        errors.append(f"Expected eight B4 fold configuration files, got {len(args_files)}")
    for relative in ("src", "configs/model349_lite.yaml", "configs/calibration.json", "infer_lite.py"):
        if not (root / relative).exists():
            errors.append(f"Missing {relative}")
    report = {"bundle": str(root), "checkpoint_count": len(manifest),
              "b4_args_count": len(args_files), "all_sha256_ok": not errors, "errors": errors}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
