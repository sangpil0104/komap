#!/usr/bin/env python3
"""Exploratory recovery for a nested-key bug in the locked Model388 E2 audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import model388_component_sampler_audit as locked


def add_legacy_nested_access_shape(payload: dict[str, Any]) -> dict[str, Any]:
    """Expose each cell summary under the nested key expected by the locked aggregator."""
    if not isinstance(payload.get("cells"), dict):
        raise ValueError("component-bin payload has no cells mapping")
    fixed_cells: dict[str, dict[str, Any]] = {}
    for cell, summary in payload["cells"].items():
        if not isinstance(summary, dict) or "cells" in summary:
            raise ValueError(f"unexpected component-bin summary for {cell}")
        item = dict(summary)
        item["cells"] = {cell: dict(summary)}
        fixed_cells[cell] = item
    return {**payload, "cells": fixed_cells}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(
    cv_root: Path,
    model382_root: Path,
    model382_lock: Path,
    model388_lock: Path,
    folds: tuple[int, int],
    draws: int,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Model388 recovery result: {output}")
    original = locked.component_bin_oracle_gains

    def compatibility_adapter(target, prediction):
        return add_legacy_nested_access_shape(original(target, prediction))

    locked.component_bin_oracle_gains = compatibility_adapter
    try:
        result = locked.headroom_audit(
            cv_root, model382_root, folds, model382_lock, model388_lock, draws
        )
    finally:
        locked.component_bin_oracle_gains = original

    locked_script = Path(locked.__file__).resolve()
    result["recovery"] = {
        "type": "aggregation-shape compatibility shim",
        "locked_evaluator_sha256": _sha256(locked_script),
        "correction": (
            "The locked aggregator indexes each already-extracted cell summary as row['cells'][cell]. "
            "The shim adds that expected nested view to a shallow copy of each cell summary. "
            "Component labels, oracle predictions, metrics, bootstrap draws, thresholds, and gates are unchanged."
        ),
        "interpretation": "exploratory after partial train-OOF score exposure; not confirmatory",
        "official_valid_or_test_read": False,
        "training_started": False,
    }
    result["recovery"]["recovery_code_sha256"] = _sha256(Path(__file__).resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--model382-root", type=Path, required=True)
    parser.add_argument("--model382-lock", type=Path, required=True)
    parser.add_argument("--model388-lock", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs=2, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=locked.BOOTSTRAP_DRAWS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folds = tuple(sorted(args.folds))
    if folds not in ((1, 2), (3, 4)):
        parser.error("run folds (1,2) exploratory or (3,4) primary")
    if args.bootstrap_draws < 100:
        parser.error("--bootstrap-draws must be >=100")
    result = run(
        args.cv_root,
        args.model382_root,
        args.model382_lock,
        args.model388_lock,
        folds,
        args.bootstrap_draws,
        args.output,
    )
    print(json.dumps({
        "model": result["model"],
        "mode": result["mode"],
        "folds": result["folds"],
        "metric": result["metric"],
        "recovery_code_sha256": result["recovery"]["recovery_code_sha256"],
        "interpretation": result["recovery"]["interpretation"],
        "official_valid_or_test_read": False,
        "training_started": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
