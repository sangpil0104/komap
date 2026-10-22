#!/usr/bin/env python3
"""Descriptive native-size tiny-GT confusion audit for locked Model408 OOF."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import model406_tiny_component_confusion_audit as base_audit
import model408_tiny_confusion_lock as m408lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--control-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Model408 secondary audit: {output}")
    m408lock.verify_oof_lock(root)
    result = base_audit.audit(
        root, args.candidate_root.resolve(), args.control_root.resolve()
    )
    result["experiment"] = "Model408 native-size GT-component confusion secondary audit"
    result["model408_source_lock_sha256"] = m408lock.sha256(root / m408lock.SOURCE_LOCK)
    result["model408_oof_lock_sha256"] = m408lock.sha256(root / m408lock.OOF_LOCK)
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
        "descriptive_only": True,
        "official_valid_read": False,
        "test_read": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
