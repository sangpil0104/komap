#!/usr/bin/env python3
"""Launch the fixed, pre-registered Model379 OOF evaluation on Colab CPU."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path("/content/komap_model1")
SCRIPT = ROOT / "model377_directional/experiments/model379_uniform_oof_ensemble.py"
OUTPUT = ROOT / "outputs/model379_uniform_oof_ensemble/results.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to replace existing Model379 result: {OUTPUT}")
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
