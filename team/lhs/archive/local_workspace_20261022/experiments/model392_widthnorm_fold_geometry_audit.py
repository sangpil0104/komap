#!/usr/bin/env python3
"""Train/held-out image-metadata-only audit of Model392 fold geometry."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from PIL import Image

GROUP_RE = re.compile(r"^(A\d+)_")
PATCH_SIZE = 512
BASE_SCALE = 2.0
BASE_OVERLAP = 170


def images_by_stem(directory: Path) -> dict[str, int]:
    paths = sorted(directory.glob("*_image.png"))
    if not paths:
        raise FileNotFoundError(f"no image PNGs found in {directory}")
    result = {}
    for path in paths:
        stem = path.name.removesuffix("_image.png")
        match = GROUP_RE.match(stem)
        if not match:
            raise ValueError(f"cannot identify specimen group from {path.name}")
        with Image.open(path) as image:
            result[stem] = int(image.width)
    if len(result) != len(paths):
        raise ValueError(f"duplicate image stems in {directory}")
    return result


def target_mode(widths: list[int]) -> int:
    counts = Counter(widths)
    highest = max(counts.values())
    return min(width for width, count in counts.items() if count == highest)


def audit_fold(cv_root: Path, fold: int) -> dict[str, Any]:
    train = images_by_stem(cv_root / f"train_fold{fold}" / "images")
    heldout = images_by_stem(cv_root / f"oof_fold{fold}" / "images")
    train_groups = {GROUP_RE.match(stem).group(1) for stem in train}
    heldout_groups = {GROUP_RE.match(stem).group(1) for stem in heldout}
    overlap = sorted(train_groups & heldout_groups)
    if overlap:
        raise ValueError(f"specimen leakage across fold {fold}: {overlap}")

    target = target_mode(list(train.values()))
    train_counts = Counter(train.values())
    heldout_counts = Counter(heldout.values())
    widths = sorted(set(train_counts) | set(heldout_counts))
    geometry = {}
    for width in widths:
        scale = BASE_SCALE * target / width
        crop = round(PATCH_SIZE / scale)
        base_tile = round(PATCH_SIZE / BASE_SCALE)
        overlap_px = min(crop - 1, round(BASE_OVERLAP * crop / base_tile))
        geometry[str(width)] = {
            "train_images": train_counts.get(width, 0),
            "heldout_images": heldout_counts.get(width, 0),
            "effective_scale": scale,
            "native_crop_and_tile_px": crop,
            "inference_overlap_px": overlap_px,
            "overlap_to_tile_ratio": overlap_px / crop,
        }

    canonical = "\n".join(
        f"{split}/{stem}_image.png\t{width}"
        for split, mapping in (("train", train), ("heldout", heldout))
        for stem, width in sorted(mapping.items())
    ).encode()
    return {
        "fold": fold,
        "train_images": len(train),
        "heldout_images": len(heldout),
        "train_groups": len(train_groups),
        "heldout_groups": len(heldout_groups),
        "group_overlap": overlap,
        "training_width_counts": {str(k): train_counts[k] for k in sorted(train_counts)},
        "heldout_width_counts": {str(k): heldout_counts[k] for k in sorted(heldout_counts)},
        "target_width_from_training_mode": target,
        "heldout_target_width_control_images": heldout_counts.get(target, 0),
        "geometry_by_native_width": geometry,
        "image_id_width_manifest_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    cv_root = args.cv_root.resolve()
    folds = {str(fold): audit_fold(cv_root, fold) for fold in range(1, 5)}
    result = {
        "experiment": "Model392 fold geometry audit",
        "scope": "fixed train grouped split image filenames and native widths only",
        "excluded": ["masks/labels", "OOF probabilities/scores", "official valid", "test"],
        "geometry": {
            "patch_size": PATCH_SIZE,
            "base_input_scale": BASE_SCALE,
            "base_overlap": BASE_OVERLAP,
            "target_rule": "per-fold train image width mode; ties choose smaller width",
        },
        "folds": folds,
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite geometry audit result: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "target_width_by_fold": {fold: item["target_width_from_training_mode"]
                                 for fold, item in folds.items()},
        "scope": result["scope"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
