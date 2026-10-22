#!/usr/bin/env python3
"""Descriptive size-bin GT-component audit for the locked Model407 OOF run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import model406_tiny_component_confusion_audit as m406_audit
import model407_component_fp_lock as m407_lock


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--control-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Model407 secondary audit: {output}")
    m407_lock.verify_oof_lock(root)
    result = m406_audit.audit(
        root, args.candidate_root.resolve(), args.control_root.resolve()
    )
    result["experiment"] = "Model407 native-size GT-component confusion secondary audit"
    result["model407_source_lock_sha256"] = sha256(root / m407_lock.SOURCE_LOCK)
    result["model407_oof_lock_sha256"] = sha256(root / m407_lock.OOF_LOCK)
    result["reference_control"] = "Model406 v3 fixed CE+Dice+boundary control OOF"
    result["official_valid_read"] = False
    result["test_read"] = False
    result["created_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({
        "result": str(output),
        "images": 70,
        "specimen_groups": 59,
        "area_bins": list(result["area_bins_native_pixels"]),
        "descriptive_only": True,
        "official_valid_read": False,
        "test_read": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
