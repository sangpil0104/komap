#!/usr/bin/env python3
"""Run the already-pre-registered Model377 train-OOF evaluator after artifact repair."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path("/content/komap_model1")
SCRIPT = ROOT / "model377_directional/experiments/model377_oof_evaluation.py"
OUTPUT = ROOT / "outputs/model377_directional_group_oof/model377_oof_evaluation.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to replace existing OOF result: {OUTPUT}")
    roots = {
        "374": ROOT / "outputs/model374_hrnet_group_oof",
        "376": ROOT / "outputs/model376_boundary_ce_group_oof",
        "377": ROOT / "outputs/model377_directional_group_oof",
    }
    expected = [root / f"fold{fold}_oof/valid_probabilities.npz"
                for root in roots.values() for fold in range(1, 5)]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"OOF evaluation inputs still missing: {missing}")
    sys.argv = [
        str(SCRIPT),
        "--cv-root", str(ROOT / "data_model322_group_cv"),
        "--model374-root", str(roots["374"]),
        "--model376-root", str(roots["376"]),
        "--model377-root", str(roots["377"]),
        "--draws", "20000",
        "--output", str(OUTPUT),
    ]
    runpy.run_path(str(SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
