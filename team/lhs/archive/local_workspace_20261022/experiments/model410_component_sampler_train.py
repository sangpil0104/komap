#!/usr/bin/env python3
"""Run the fixed Model392 fine-tuner with Model410's paired crop sampler."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "model336_hrnet_widthnorm_candidate"
TRAIN_ENTRY = ROOT / "model397_reproduction_bundle/model392/train.py"
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model410_component_uniform_sampler import install_sampler  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--arm", choices=("control", "candidate"), required=True)
    own, forwarded = parser.parse_known_args()
    if "--arm" in forwarded:
        raise ValueError("--arm must be supplied once")
    output_dir = None
    for index, value in enumerate(forwarded):
        if value == "--output-dir" and index + 1 < len(forwarded):
            output_dir = Path(forwarded[index + 1])
            break
        if value.startswith("--output-dir="):
            output_dir = Path(value.split("=", 1)[1])
            break
    if output_dir is None:
        raise ValueError("forwarded Model392 arguments must include --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    arm_record = output_dir / "model410_sampler_arm.json"
    record = {
        "experiment": "Model410 small-component-balanced crop sampling",
        "arm": own.arm,
        "area_bins_modal_width_equivalent_pixels": [[1, 4], [4, 16], [16, 64]],
        "normalized_area_formula": "raw_component_area*(target_width/image_width)^2",
        "positive_sample_probability": 0.5,
        "classes": ["Al3Ni", "Eutectic Si"],
        "training_argv_without_resume_flag": [arg for arg in forwarded if arg != "--resume"],
        "valid_read": False,
        "test_read": False,
    }
    if arm_record.exists():
        if "--resume" not in forwarded:
            raise FileExistsError(f"refusing existing Model410 arm receipt: {arm_record}")
        if json.loads(arm_record.read_text(encoding="utf-8")) != record:
            raise ValueError("resume arm does not match the saved Model410 receipt")
    else:
        if any(output_dir.iterdir()):
            raise FileExistsError(f"refusing nonempty Model410 output: {output_dir}")
        with arm_record.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

    install_sampler(own.arm)
    sys.argv = [str(TRAIN_ENTRY), *forwarded]
    runpy.run_path(str(TRAIN_ENTRY), run_name="__main__")


if __name__ == "__main__":
    main()
