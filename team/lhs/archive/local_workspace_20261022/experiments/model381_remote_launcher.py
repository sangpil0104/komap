#!/usr/bin/env python3
"""Launch the fixed CPU-only Model381 grouped-OOF boundary diagnostic."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path("/content/komap_model1")
SCRIPT = ROOT / "model377_directional/experiments/model381_boundary_error_diagnostic.py"
OUTPUT = ROOT / "outputs/model381_boundary_error_diagnostic/results.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to replace existing Model381 result: {OUTPUT}")
    roots = {
        "model374_hrnet_group_oof",
        "model376_boundary_ce_group_oof",
        "model377_directional_group_oof",
    }
    missing = [
        str(ROOT / "outputs" / folder / f"fold{fold}_oof/valid_probabilities.npz")
        for folder in roots for fold in range(1, 5)
        if not (ROOT / "outputs" / folder / f"fold{fold}_oof/valid_probabilities.npz").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Model381 OOF input archives missing: {missing}")
    sys.argv = [
        str(SCRIPT),
        "--cv-root", str(ROOT / "data_model322_group_cv"),
        "--model374-root", str(ROOT / "outputs/model374_hrnet_group_oof"),
        "--model376-root", str(ROOT / "outputs/model376_boundary_ce_group_oof"),
        "--model377-root", str(ROOT / "outputs/model377_directional_group_oof"),
        "--draws", "20000",
        "--output", str(OUTPUT),
    ]
    runpy.run_path(str(SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
