#!/usr/bin/env python3
"""Train-mask-only preflight for Model410's width-normalized area bins."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "model336_hrnet_widthnorm_candidate"
import sys
sys.path.insert(0, str(SOURCE_ROOT))

from komap_seg.dataset import (  # noqa: E402
    crop_size_for_width,
    find_pairs,
    load_class_mask,
    load_image_rgb,
    modal_image_width,
)


TARGETS = {1: "Al3Ni", 3: "Eutectic Si"}
AREA_BINS = ((1, 4), (4, 16), (16, 64))


def audit_fold(fold: int) -> dict:
    split = ROOT / "outputs/model403_local_supcon/splits/model392_cv" / f"seed1_fold{fold}"
    images_dir = split / "train/images"
    masks_dir = split / "train/masks"
    pairs = find_pairs(images_dir, masks_dir)
    target_width = modal_image_width(images_dir)
    counts = {class_id: Counter() for class_id in TARGETS}
    supported_images = {class_id: Counter() for class_id in TARGETS}
    widths = Counter()
    padding_needed = 0

    for pair in pairs:
        image = load_image_rgb(pair["image"])
        mask = load_class_mask(pair["mask"])
        if image.shape[:2] != mask.shape:
            raise ValueError(f"image/mask dimensions differ for {pair['stem']}")
        height, width = mask.shape
        widths[width] += 1
        crop_size = crop_size_for_width(512, 2.0, target_width, width)
        padding_needed += int(height < crop_size or width < crop_size)
        area_factor = (target_width / width) ** 2

        for class_id in TARGETS:
            binary = np.ascontiguousarray(mask == class_id, dtype=np.uint8)
            component_count, _, stats, _ = cv2.connectedComponentsWithStats(
                binary, connectivity=8, ltype=cv2.CV_32S
            )
            image_bins = set()
            for area_value in stats[1:component_count, cv2.CC_STAT_AREA]:
                normalized_area = int(area_value) * area_factor
                for bin_id, (lower, upper) in enumerate(AREA_BINS):
                    if lower <= normalized_area < upper:
                        counts[class_id][bin_id] += 1
                        image_bins.add(bin_id)
                        break
            for bin_id in image_bins:
                supported_images[class_id][bin_id] += 1

    result = {
        "images": len(pairs),
        "target_width": target_width,
        "image_width_counts": {str(width): n for width, n in sorted(widths.items())},
        "crop_size_range": [
            min(crop_size_for_width(512, 2.0, target_width, width) for width in widths),
            max(crop_size_for_width(512, 2.0, target_width, width) for width in widths),
        ],
        "images_needing_reflect_padding": padding_needed,
        "area_formula": "raw_component_area*(target_width/image_width)^2",
        "area_bins_modal_width_equivalent_pixels": [list(item) for item in AREA_BINS],
        "classes": {
            TARGETS[class_id]: {
                "component_counts_by_bin": [counts[class_id][i] for i in range(len(AREA_BINS))],
                "image_support_by_bin": [supported_images[class_id][i] for i in range(len(AREA_BINS))],
            }
            for class_id in TARGETS
        },
    }
    if padding_needed:
        raise ValueError(f"unexpected reflection padding in fold {fold}: {padding_needed} images")
    for class_id, class_name in TARGETS.items():
        if any(counts[class_id][i] == 0 for i in range(len(AREA_BINS))):
            raise ValueError(f"empty normalized size bin for {class_name}, fold {fold}")
    return result


def main() -> None:
    result = {
        "experiment": "Model410 width-normalized component-bin preflight",
        "scope": "outer-training images and masks only; no held-out or official valid/test data",
        "connectivity": 8,
        "folds": {str(fold): audit_fold(fold) for fold in range(1, 5)},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
