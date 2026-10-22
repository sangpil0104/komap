#!/usr/bin/env python3
"""Train-mask-only audit of phase-boundary pixel thickness vs image width."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import re
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO_ROOT / "experiments/results/model391_physical_scale_mask_audit_lock.json"
PROTOCOL_PATH = "experiments/model391_physical_scale_mask_audit.md"
SCRIPT_PATH = "experiments/model391_physical_scale_mask_audit.py"
CLASS_NAMES = ("Al", "Al3Ni", "Primary_Si", "Eutectic_Si")
PALETTE = np.asarray(
    ((153, 127, 76), (25, 76, 153), (76, 178, 76), (204, 204, 204)),
    dtype=np.uint8,
)
GROUP_RE = re.compile(r"^(A\d+)_")
STEP_RE = re.compile(r"_step([1-4])_")
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 391
EXPECTED_IMAGES = 70
EXPECTED_GROUPS = 59


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_lock(lock_path: Path, train_zip: Path) -> dict[str, Any]:
    receipt = json.loads(lock_path.read_text(encoding="utf-8"))
    if receipt.get("experiment") != "Model391 train-mask physical-scale feasibility audit":
        raise ValueError("unexpected Model391 lock identity")
    for relative, expected in receipt["artifacts_sha256"].items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Model391 source/protocol lock mismatch: {relative}")
    expected_input = receipt["input_sha256"].get("train.zip")
    if not expected_input or sha256(train_zip) != expected_input:
        raise ValueError("train.zip SHA-256 differs from the locked input")
    return receipt


def decode_mask(data: bytes, label: str) -> np.ndarray:
    rgb = np.asarray(Image.open(BytesIO(data)).convert("RGB"), dtype=np.uint8)
    mask = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        mask[np.all(rgb == color, axis=2)] = class_id
    if np.any(mask == 255):
        raise ValueError(f"unknown mask RGB values in {label}")
    return mask


def run_lengths(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = mask.shape
    left = np.ones((height, width), dtype=np.uint16)
    right = np.ones_like(left)
    up = np.ones_like(left)
    down = np.ones_like(left)
    for col in range(1, width):
        left[:, col] = np.where(mask[:, col] == mask[:, col - 1], left[:, col - 1] + 1, 1)
    for col in range(width - 2, -1, -1):
        right[:, col] = np.where(mask[:, col] == mask[:, col + 1], right[:, col + 1] + 1, 1)
    for row in range(1, height):
        up[row, :] = np.where(mask[row, :] == mask[row - 1, :], up[row - 1, :] + 1, 1)
    for row in range(height - 2, -1, -1):
        down[row, :] = np.where(mask[row, :] == mask[row + 1, :], down[row + 1, :] + 1, 1)
    return left, right, up, down


def collect_boundary_widths(mask: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
    left_run, right_run, up_run, down_run = run_lengths(mask)
    collected: list[list[np.ndarray]] = [[] for _ in CLASS_NAMES]
    counts = [0 for _ in CLASS_NAMES]
    interfaces = (
        (mask[:, :-1], mask[:, 1:], left_run[:, :-1], right_run[:, 1:]),
        (mask[:-1, :], mask[1:, :], up_run[:-1, :], down_run[1:, :]),
    )
    for a, b, run_a, run_b in interfaces:
        different = a != b
        for class_id in range(len(CLASS_NAMES)):
            side_a = different & (a == class_id)
            side_b = different & (b == class_id)
            values = np.concatenate((run_a[side_a], run_b[side_b])).astype(np.uint16)
            if values.size:
                collected[class_id].append(values)
                counts[class_id] += int(values.size)
    merged = [np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint16)
              for chunks in collected]
    return merged, counts


def parse_pairs(archive: ZipFile) -> tuple[dict[str, str], dict[str, str]]:
    names = archive.namelist()
    image_prefix = "train/images/"
    mask_prefix = "train/masks/"
    images = {
        path[len(image_prefix):-len("_image.png")]: path
        for path in names
        if path.startswith(image_prefix) and path.endswith("_image.png")
    }
    masks = {
        path[len(mask_prefix):-len("_mask.png")]: path
        for path in names
        if path.startswith(mask_prefix) and path.endswith("_mask.png")
    }
    if len(images) != EXPECTED_IMAGES or len(masks) != EXPECTED_IMAGES:
        raise ValueError(f"expected 70 train image/mask pairs, found {len(images)}/{len(masks)}")
    if images.keys() != masks.keys():
        raise ValueError("train image and mask identifiers do not match exactly")
    return images, masks


def slope_with_step_control(rows: list[dict[str, Any]], class_id: int) -> float | None:
    usable = [row for row in rows if row["median_run_px"][class_id] is not None]
    if len(usable) < 8 or len({row["width"] for row in usable}) < 2:
        return None
    log_width = np.log(np.asarray([row["width"] for row in usable], dtype=np.float64))
    y = np.log(np.asarray([row["median_run_px"][class_id] for row in usable], dtype=np.float64))
    steps = np.asarray([row["step"] for row in usable], dtype=np.int8)
    columns = [np.ones(len(usable)), log_width]
    columns.extend((steps == step).astype(np.float64) for step in (2, 3, 4))
    design = np.column_stack(columns)
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return None
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coefficients[1])


def bootstrap_slope(rows: list[dict[str, Any]], class_id: int) -> dict[str, Any]:
    groups = sorted({row["group"] for row in rows})
    point = slope_with_step_control(rows, class_id)
    if point is None:
        return {"slope": None, "group_bootstrap_95ci": None, "valid_draws": 0}
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["median_run_px"][class_id] is not None:
            by_group[row["group"]].append(row)
    rng = np.random.default_rng(BOOTSTRAP_SEED + class_id)
    slopes = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    valid = 0
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = rng.integers(0, len(groups), size=len(groups))
        draw_rows = [row for index in sampled for row in by_group[groups[int(index)]]]
        slope = slope_with_step_control(draw_rows, class_id)
        if slope is not None and np.isfinite(slope):
            slopes[valid] = slope
            valid += 1
    if valid < BOOTSTRAP_DRAWS * 0.95:
        raise ValueError(f"too few non-singular group-bootstrap fits for {CLASS_NAMES[class_id]}: {valid}")
    low, high = np.percentile(slopes[:valid], [2.5, 97.5])
    return {"slope": point, "group_bootstrap_95ci": [float(low), float(high)],
            "valid_draws": valid}


def run_audit(train_zip: Path, lock_path: Path = LOCK_PATH) -> dict[str, Any]:
    receipt = verify_lock(lock_path, train_zip)
    locked_at = datetime.fromisoformat(receipt["locked_at_utc"].replace("Z", "+00:00"))
    if locked_at.tzinfo is None:
        raise ValueError("Model391 lock timestamp must include a timezone")

    records: list[dict[str, Any]] = []
    with ZipFile(train_zip) as archive:
        image_paths, mask_paths = parse_pairs(archive)
        for image_id in sorted(image_paths):
            group_match = GROUP_RE.match(image_id)
            step_match = STEP_RE.search(image_id)
            if not group_match or not step_match:
                raise ValueError(f"cannot parse specimen group/step from {image_id}")
            with Image.open(BytesIO(archive.read(image_paths[image_id]))) as image:
                width, height = image.size
            mask = decode_mask(archive.read(mask_paths[image_id]), mask_paths[image_id])
            if mask.shape != (height, width):
                raise ValueError(f"image/mask shape mismatch for {image_id}")
            run_widths, run_counts = collect_boundary_widths(mask)
            medians = [float(np.median(values)) if values.size else None for values in run_widths]
            records.append({
                "group": group_match.group(1),
                "step": int(step_match.group(1)),
                "width": width,
                "height": height,
                "median_run_px": medians,
                "boundary_side_samples": run_counts,
            })

    group_count = len({row["group"] for row in records})
    if len(records) != EXPECTED_IMAGES or group_count != EXPECTED_GROUPS:
        raise ValueError(f"expected 70 images/59 groups; got {len(records)}/{group_count}")
    width_strata: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        width_strata[row["width"]].append(row)
    if len(width_strata) < 2:
        raise ValueError("train images do not contain multiple native widths")

    summaries = {}
    for class_id, name in enumerate(CLASS_NAMES):
        estimate = bootstrap_slope(records, class_id)
        usable = [row for row in records if row["median_run_px"][class_id] is not None]
        estimate.update({
            "images": len(usable),
            "groups": len({row["group"] for row in usable}),
            "median_run_px_by_native_width": {
                str(width): float(np.median([
                    row["median_run_px"][class_id] for row in rows
                    if row["median_run_px"][class_id] is not None
                ]))
                for width, rows in sorted(width_strata.items())
                if any(row["median_run_px"][class_id] is not None for row in rows)
            },
        })
        summaries[name] = estimate

    non_al = [summaries[name] for name in CLASS_NAMES[1:]]
    supported = sum(
        item["slope"] is not None
        and 0.5 <= item["slope"] <= 1.5
        and item["group_bootstrap_95ci"][0] > 0.25
        for item in non_al
    )
    rejected = all(
        item["group_bootstrap_95ci"] is not None
        and item["group_bootstrap_95ci"][1] < 0.5
        for item in non_al
    )
    decision = (
        "supports_followup_hypothesis" if supported >= 2
        else "defer_width_normalization" if rejected
        else "inconclusive"
    )
    return {
        "experiment": "Model391 train-mask physical-scale feasibility audit",
        "scope": "train.zip image/mask pairs only; no prediction, OOF, valid, or test artifacts",
        "metric": "image-wise median axial same-class boundary run length in native pixels",
        "regression": "log(median run length) ~ log(native image width) + categorical step fixed effects",
        "bootstrap": {
            "unit": "specimen group, retaining all images in each sampled group",
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "minimum_valid_fraction": 0.95,
        },
        "images": len(records),
        "specimen_groups": group_count,
        "resolution_strata": [
            {"native_width_px": width, "images": len(rows),
             "specimen_groups": len({row["group"] for row in rows})}
            for width, rows in sorted(width_strata.items())
        ],
        "class_slopes": summaries,
        "pre_registered_decision": decision,
        "supported_non_al_phase_count": supported,
        "interpretation_guard": (
            "Pixel-run scaling is not physical calibration: field-of-view and acquisition metadata "
            "are unavailable, and phase morphology may vary by specimen/step. A positive result is "
            "only a gate for a later paired OOF experiment, not evidence of mIoU gain."
        ),
        "locked_input_sha256": receipt["input_sha256"]["train.zip"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-zip", type=Path, default=REPO_ROOT / "train.zip")
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "experiments/results/model391_physical_scale_mask_audit.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {args.output}")
    result = run_audit(args.train_zip, args.lock)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
