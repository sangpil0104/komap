#!/usr/bin/env python3
"""One-shot, outer-train-mask-only support audit for 8-way interior directions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, maximum_filter


PROTOCOL_REL = "experiments/model392_direction_target_support_preregistration.md"
SCRIPT_REL = "experiments/model392_direction_target_support.py"
LOCK_REL = "experiments/results/model392_direction_target_support_lock.json"
OUTPUT_REL = "experiments/results/model392_direction_target_support.json"
GROUP_RE = re.compile(r"^(A\d+)")
CLASS_NAMES = ("Al", "Al3Ni", "Primary_Si", "Eutectic_Si")
CANDIDATE_CLASSES = (1, 2)
CLEARANCES = (2, 4, 8)
MAX_OFFSETS = (4, 8, 16, 32)
DIRECTION_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
PALETTE = np.asarray(
    ((153, 127, 76), (25, 76, 153), (76, 178, 76), (204, 204, 204)),
    dtype=np.uint8,
)
EXPECTED_IMAGES = 70
EXPECTED_GROUPS = 59
RUNTIME_LOCK_SHA256 = "68e7f2d524e338f24d9e88472d1463aadc69d5cb4daf7a56e0d5997edc3ebc98"
SEQUENCE_LOCK_SHA256 = "3df753de47ab636390ef806237018cd3c302636c0b59cc1dd2ea5236532a551e"
CONFIG = {
    "candidate_classes": {str(c): CLASS_NAMES[c] for c in CANDIDATE_CLASSES},
    "boundary_band": "3x3 neighborhood contains more than one of four classes",
    "interior_clearance_px": list(CLEARANCES),
    "maximum_target_offset_px": list(MAX_OFFSETS),
    "directions_clockwise_from_north": list(DIRECTION_NAMES),
    "folds": [1, 2, 3, 4],
    "scope": "outer-train masks only; no OOF predictions or official valid/test",
}
SOURCE_RELS = (
    PROTOCOL_REL,
    SCRIPT_REL,
    "model336_hrnet_widthnorm_candidate/komap_seg/palette.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/losses.py",
    "experiments/results/model392_widthnorm_runtime_lock.json",
    "experiments/results/model392_sequence_closure_lock.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def group_of(stem: str) -> str:
    match = GROUP_RE.match(stem)
    if not match:
        raise ValueError(f"cannot parse specimen group from image ID {stem}")
    return match.group(1)


def mask_to_ids(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    ids = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        ids[np.all(rgb == color, axis=2)] = class_id
    if np.any(ids == 255):
        raise ValueError(f"unknown mask color in {path}")
    return ids


def boundary_band(mask: np.ndarray) -> np.ndarray:
    present = np.zeros((4, *mask.shape), dtype=bool)
    for class_id in range(4):
        present[class_id] = maximum_filter(
            (mask == class_id).astype(np.uint8),
            size=3,
            mode="constant",
            cval=0,
        ) > 0
    return present.sum(axis=0) > 1


def direction_bins(dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    angle = np.arctan2(dx.astype(np.float64), -dy.astype(np.float64))
    return np.floor((angle + np.pi / 8) / (np.pi / 4)).astype(np.int8) % 8


def target_support(mask: np.ndarray, class_id: int) -> dict[str, Any]:
    class_pixels = mask == class_id
    sources = class_pixels & boundary_band(mask)
    # Pad with non-class pixels so image borders count as the exterior.
    inside_distance = distance_transform_edt(np.pad(class_pixels, 1))[1:-1, 1:-1]
    source_y, source_x = np.where(sources)
    result: dict[str, Any] = {
        "class_name": CLASS_NAMES[class_id],
        "boundary_source_pixels": int(len(source_y)),
        "interior_seed_pixels_by_clearance": {},
        "support_by_clearance_and_max_offset": {},
    }
    if len(source_y) == 0:
        for clearance in CLEARANCES:
            result["interior_seed_pixels_by_clearance"][str(clearance)] = 0
            result["support_by_clearance_and_max_offset"][str(clearance)] = {
                str(radius): {
                    "supported_source_pixels": 0,
                    "support_fraction": 0.0,
                    "direction_counts": {name: 0 for name in DIRECTION_NAMES},
                }
                for radius in MAX_OFFSETS
            }
        return result

    for clearance in CLEARANCES:
        seeds = class_pixels & (inside_distance >= clearance)
        seed_count = int(seeds.sum())
        result["interior_seed_pixels_by_clearance"][str(clearance)] = seed_count
        by_offset = {}
        if seed_count:
            target_distance, nearest = distance_transform_edt(
                ~seeds, return_indices=True
            )
            target_y = nearest[0, source_y, source_x]
            target_x = nearest[1, source_y, source_x]
            bins = direction_bins(target_y - source_y, target_x - source_x)
            source_distances = target_distance[source_y, source_x]
            for radius in MAX_OFFSETS:
                supported = source_distances <= radius
                counts = np.bincount(bins[supported], minlength=8)
                by_offset[str(radius)] = {
                    "supported_source_pixels": int(supported.sum()),
                    "support_fraction": float(supported.mean()),
                    "direction_counts": {
                        name: int(counts[index])
                        for index, name in enumerate(DIRECTION_NAMES)
                    },
                }
        else:
            for radius in MAX_OFFSETS:
                by_offset[str(radius)] = {
                    "supported_source_pixels": 0,
                    "support_fraction": 0.0,
                    "direction_counts": {name: 0 for name in DIRECTION_NAMES},
                }
        result["support_by_clearance_and_max_offset"][str(clearance)] = by_offset
    return result


def fold_files(root: Path, fold: int, split: str) -> tuple[dict[str, Path], dict[str, Path]]:
    base = root / "data_model322_group_cv" / f"{split}_fold{fold}"
    images = {
        p.name.removesuffix("_image.png"): p
        for p in (base / "images").glob("*_image.png")
    }
    masks = {
        p.name.removesuffix("_mask.png"): p
        for p in (base / "masks").glob("*_mask.png")
    }
    if not images or set(images) != set(masks):
        raise ValueError(f"{split} fold {fold}: image/mask IDs do not pair exactly")
    return images, masks


def snapshot_inputs(root: Path) -> dict[str, Any]:
    source_hashes = {}
    for relative in SOURCE_RELS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"missing/symlinked source {relative}")
        source_hashes[relative] = sha256(path)
    runtime = root / "experiments/results/model392_widthnorm_runtime_lock.json"
    sequence = root / "experiments/results/model392_sequence_closure_lock.json"
    if sha256(runtime) != RUNTIME_LOCK_SHA256 or sha256(sequence) != SEQUENCE_LOCK_SHA256:
        raise ValueError("Model392 runtime/sequence locks differ from the reviewed closure")

    mask_hashes: dict[str, str] = {}
    folds = {}
    all_oof_ids: set[str] = set()
    all_oof_groups: set[str] = set()
    assigned_groups: dict[str, int] = {}
    for fold in (1, 2, 3, 4):
        train_images, train_masks = fold_files(root, fold, "train")
        held_images, held_masks = fold_files(root, fold, "oof")
        if set(train_masks) & set(held_masks):
            raise ValueError(f"fold {fold}: train/OOF IDs overlap")
        train_groups = {group_of(stem) for stem in train_masks}
        held_groups = {group_of(stem) for stem in held_masks}
        if train_groups & held_groups:
            raise ValueError(f"fold {fold}: train/OOF specimen groups overlap")
        for stem in held_masks:
            group = group_of(stem)
            if group in assigned_groups and assigned_groups[group] != fold:
                raise ValueError(f"OOF group {group} appears in multiple folds")
            assigned_groups[group] = fold
            all_oof_ids.add(stem)
            all_oof_groups.add(group)
        for path in train_masks.values():
            mask_hashes[str(path.relative_to(root))] = sha256(path)
        folds[str(fold)] = {
            "train_images": len(train_images),
            "train_groups": len(train_groups),
            "heldout_images": len(held_images),
            "heldout_groups": len(held_groups),
            "group_overlap": [],
            "train_ids": sorted(train_masks),
            "heldout_ids": sorted(held_masks),
        }
    if len(all_oof_ids) != EXPECTED_IMAGES or len(all_oof_groups) != EXPECTED_GROUPS:
        raise ValueError("fixed folds do not contain exactly 70 images and 59 groups")
    return {
        "project_root": str(root.resolve()),
        "config": CONFIG,
        "source_sha256": source_hashes,
        "outer_train_mask_sha256": mask_hashes,
        "folds": folds,
        "oof_images_for_fold_disjointness_check": len(all_oof_ids),
        "oof_groups_for_fold_disjointness_check": len(all_oof_groups),
        "scope": "outer-train masks only; no OOF predictions or official valid/test",
    }


def make_lock(root: Path, lock_path: Path) -> dict[str, Any]:
    if lock_path.exists():
        raise FileExistsError(f"refusing to overwrite lock {lock_path}")
    output = root / OUTPUT_REL
    if output.exists() or Path(str(output) + ".claim").exists():
        raise FileExistsError("support audit already has an output/claim")
    body = {
        "experiment": "Model392 direction-target support audit",
        "revision": 1,
        "locked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "snapshot": snapshot_inputs(root),
    }
    body["lock_sha256"] = canonical_sha256(body)
    write_exclusive(lock_path, body)
    return body


def verify_lock(root: Path, path: Path) -> dict[str, Any]:
    locked = json.loads(path.read_text(encoding="utf-8"))
    digest = locked.get("lock_sha256")
    body = {key: value for key, value in locked.items() if key != "lock_sha256"}
    if digest != canonical_sha256(body):
        raise ValueError("support audit lock content hash is invalid")
    if locked.get("snapshot") != snapshot_inputs(root):
        raise ValueError("outer-train masks/splits/source changed since lock")
    return locked


def run_audit(root: Path, lock: dict[str, Any], output: Path) -> dict[str, Any]:
    if output.exists() or Path(str(output) + ".claim").exists():
        raise FileExistsError("refusing to overwrite support result/claim")
    out_root = root / "outputs/model392_widthnorm"
    lock_handle = (out_root / ".model392_oof.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_handle.close()
        raise RuntimeError("Model392 runner active; refusing concurrent audit") from error
    claim = Path(str(output) + ".claim")
    write_exclusive(claim, {
        "lock_sha256": lock["lock_sha256"],
        "claimed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "policy": "one-shot; an interrupted claimed attempt is not rerun",
    })

    results = {}
    for fold in (1, 2, 3, 4):
        _, masks = fold_files(root, fold, "train")
        fold_groups = sorted({group_of(stem) for stem in masks})
        accumulated: dict[str, dict[str, Any]] = {}
        for stem, path in sorted(masks.items()):
            mask = mask_to_ids(path)
            for class_id in CANDIDATE_CLASSES:
                image_result = target_support(mask, class_id)
                key = str(class_id)
                cell = accumulated.setdefault(key, {
                    "class_name": CLASS_NAMES[class_id],
                    "images_with_boundary_pixels": 0,
                    "boundary_source_pixels": 0,
                    "interior_seed_pixels_by_clearance": {
                        str(c): 0 for c in CLEARANCES
                    },
                    "support_by_clearance_and_max_offset": {
                        str(c): {
                            str(r): {
                                "supported_source_pixels": 0,
                                "direction_counts": {name: 0 for name in DIRECTION_NAMES},
                                "groups_by_direction": {name: set() for name in DIRECTION_NAMES},
                            }
                            for r in MAX_OFFSETS
                        }
                        for c in CLEARANCES
                    },
                })
                source_count = image_result["boundary_source_pixels"]
                cell["images_with_boundary_pixels"] += int(source_count > 0)
                cell["boundary_source_pixels"] += source_count
                for clearance in CLEARANCES:
                    ck = str(clearance)
                    cell["interior_seed_pixels_by_clearance"][ck] += (
                        image_result["interior_seed_pixels_by_clearance"][ck]
                    )
                    for radius in MAX_OFFSETS:
                        rk = str(radius)
                        image_cell = image_result["support_by_clearance_and_max_offset"][ck][rk]
                        target_cell = cell["support_by_clearance_and_max_offset"][ck][rk]
                        target_cell["supported_source_pixels"] += image_cell["supported_source_pixels"]
                        for direction, count in image_cell["direction_counts"].items():
                            target_cell["direction_counts"][direction] += count
                            if count:
                                target_cell["groups_by_direction"][direction].add(group_of(stem))
        for class_id in CANDIDATE_CLASSES:
            cell = accumulated[str(class_id)]
            for clearance in CLEARANCES:
                for radius in MAX_OFFSETS:
                    item = cell["support_by_clearance_and_max_offset"][str(clearance)][str(radius)]
                    item["support_fraction"] = (
                        item["supported_source_pixels"] / cell["boundary_source_pixels"]
                        if cell["boundary_source_pixels"] else 0.0
                    )
                    item["groups_by_direction"] = {
                        direction: len(groups)
                        for direction, groups in item["groups_by_direction"].items()
                    }
        results[str(fold)] = {
            "train_images": len(masks),
            "train_groups": len(fold_groups),
            "classes": accumulated,
        }
        print(json.dumps({"stage": "support_audit", "fold": fold,
                          "train_images": len(masks), "train_groups": len(fold_groups)}),
              flush=True)

    result = {
        "experiment": "Model392 direction-target support audit",
        "revision": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "outer-train masks only; no OOF predictions or official valid/test",
        "training_or_miou_computed": False,
        "lock_sha256": lock["lock_sha256"],
        "config": CONFIG,
        "primary_cells": {
            "interior_clearance_px": 4,
            "maximum_target_offset_px": 16,
        },
        "fold_results": results,
        "interpretation_guard": (
            "Geometric target availability only; not evidence that a model learns "
            "directions or improves semantic mIoU."
        ),
    }
    temporary = Path(str(output) + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing existing temp result {temporary}")
    write_exclusive(temporary, result)
    os.replace(temporary, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--create-lock", action="store_true")
    parser.add_argument("--verify-lock", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--lock", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if sum((args.create_lock, args.verify_lock, args.run)) != 1:
        parser.error("choose exactly one of --create-lock, --verify-lock, or --run")
    root = args.project_root.resolve()
    lock_path = args.lock or (root / LOCK_REL)
    output = args.output or (root / OUTPUT_REL)
    if args.create_lock:
        lock = make_lock(root, lock_path)
        print(json.dumps({"status": "LOCKED", "lock_sha256": lock["lock_sha256"],
                          "folds": 4, "candidate_classes": ["Al3Ni", "Primary_Si"]}, indent=2))
    elif args.verify_lock:
        lock = verify_lock(root, lock_path)
        print(json.dumps({"status": "LOCK_VERIFIED",
                          "lock_sha256": lock["lock_sha256"]}, indent=2))
    else:
        lock = verify_lock(root, lock_path)
        result = run_audit(root, lock, output)
        print(json.dumps({"status": "COMPLETE", "output": str(output),
                          "primary_cells": result["primary_cells"]}, indent=2))


if __name__ == "__main__":
    main()
