#!/usr/bin/env python3
"""Train-mask-only audit of Model406's width-normalized component size bins."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image


PALETTE = np.asarray(
    ((153, 127, 76), (25, 76, 153), (76, 178, 76), (204, 204, 204)),
    dtype=np.uint8,
)
TARGETS = {1: "Al3Ni", 3: "Eutectic_Si"}
NATIVE_BINS = ((1, 64), (64, 256), (256, None))
FIXED_AUGMENTED_BINS = (4, 256)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_from_rgb(rgb: np.ndarray) -> np.ndarray:
    distances = (
        rgb[:, :, None, :].astype(np.int32)
        - PALETTE[None, None, :, :].astype(np.int32)
    ) ** 2
    return distances.sum(axis=-1).argmin(axis=-1).astype(np.uint8)


def area_band(area: int) -> str:
    if area == 1:
        return "1px"
    if area <= 3:
        return "2-3px"
    if area <= 15:
        return "4-15px"
    if area < 64:
        return "16-63px"
    if area < 256:
        return "64-255px"
    return ">=256px"


def audit(train_zip: Path, target_width: int = 1230,
          nominal_input_scale: float = 2.0) -> dict:
    with ZipFile(train_zip) as archive:
        names = archive.namelist()
        images = {
            name.removeprefix("train/images/").removesuffix("_image.png"): name
            for name in names
            if name.startswith("train/images/") and name.endswith("_image.png")
        }
        masks = {
            name.removeprefix("train/masks/").removesuffix("_mask.png"): name
            for name in names
            if name.startswith("train/masks/") and name.endswith("_mask.png")
        }
        if len(images) != 70 or set(images) != set(masks):
            raise ValueError("expected exactly 70 paired training images and masks")

        widths = Counter()
        component_sizes = {class_id: Counter() for class_id in TARGETS}
        per_width: dict[tuple[int, int], Counter] = defaultdict(Counter)
        for image_id in sorted(images):
            with Image.open(BytesIO(archive.read(images[image_id]))) as image_file:
                width = image_file.width
            with Image.open(BytesIO(archive.read(masks[image_id]))) as mask_file:
                mask_rgb = np.asarray(mask_file.convert("RGB"))
            if mask_rgb.shape[1] != width:
                raise ValueError(f"image/mask width mismatch for {image_id}")
            widths[width] += 1
            scale = nominal_input_scale * target_width / width
            class_ids = ids_from_rgb(mask_rgb)
            for class_id in TARGETS:
                count, _, stats, _ = cv2.connectedComponentsWithStats(
                    np.ascontiguousarray(class_ids == class_id, dtype=np.uint8),
                    connectivity=8,
                    ltype=cv2.CV_32S,
                )
                areas = stats[1:count, cv2.CC_STAT_AREA]
                for raw_area in areas:
                    area = int(raw_area)
                    component_sizes[class_id][area_band(area)] += 1
                    if 1 <= area < 64 and FIXED_AUGMENTED_BINS[0] <= area * scale**2 < FIXED_AUGMENTED_BINS[1]:
                        per_width[(width, class_id)]["native_1_63_captured"] += 1
                    if 64 <= area < 256 and FIXED_AUGMENTED_BINS[0] <= area * scale**2 < FIXED_AUGMENTED_BINS[1]:
                        per_width[(width, class_id)]["native_64_255_included"] += 1
                    native_bin = (
                        "native_1_63" if area < 64 else
                        "native_64_255" if area < 256 else "native_ge256"
                    )
                    per_width[(width, class_id)][native_bin] += 1

        width_summary = {}
        for width in sorted(widths):
            scale = nominal_input_scale * target_width / width
            key = str(width)
            width_summary[key] = {
                "images": widths[width],
                "effective_input_scale": scale,
                "fixed_augmented_area_bin_native_approx": [
                    FIXED_AUGMENTED_BINS[0] / scale**2,
                    FIXED_AUGMENTED_BINS[1] / scale**2,
                ],
                "classes": {},
            }
            for class_id, name in TARGETS.items():
                counts = per_width[(width, class_id)]
                tiny = counts["native_1_63"]
                width_summary[key]["classes"][name] = {
                    "native_1_63_components": tiny,
                    "native_1_63_components_inside_fixed_bins": counts["native_1_63_captured"],
                    "native_1_63_capture_percent": (
                        100.0 * counts["native_1_63_captured"] / tiny if tiny else None
                    ),
                    "native_64_255_components": counts["native_64_255"],
                    "native_64_255_components_inside_fixed_bins": counts["native_64_255_included"],
                }

    return {
        "audit": "Model406 nominal native-component size-bin mapping",
        "scope": "70 train.zip images and masks only; no OOF predictions or official valid/test",
        "train_zip": str(train_zip.resolve()),
        "train_zip_sha256": sha256(train_zip),
        "target_width": target_width,
        "nominal_input_scale": nominal_input_scale,
        "fixed_augmented_area_bins": list(FIXED_AUGMENTED_BINS),
        "connectivity": 8,
        "image_width_counts": {str(k): v for k, v in sorted(widths.items())},
        "component_size_histogram": {
            TARGETS[c]: dict(component_sizes[c]) for c in TARGETS
        },
        "by_image_width": width_summary,
        "caveat": (
            "Nominal area mapping only; full-image native components are not realized crop exposure. "
            "Cropping, border exclusion, interpolation and geometric augmentation alter actual targets."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-zip", type=Path, required=True)
    parser.add_argument("--target-width", type=int, default=1230)
    parser.add_argument("--input-scale", type=float, default=2.0)
    args = parser.parse_args()
    print(json.dumps(audit(args.train_zip, args.target_width, args.input_scale),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
