#!/usr/bin/env python3
"""Model410's train-only small-component-balanced crop sampler.

The default dataset and all shared project code remain unchanged. The paired
trainer injects this subclass only for Model410; ``ARM`` is fixed before the
training script constructs its dataset.
"""

from __future__ import annotations

import random
import math
from typing import Optional

import cv2
import numpy as np

from komap_seg.dataset import KomapPatchDataset


AREA_BINS: tuple[tuple[int, int], ...] = ((1, 4), (4, 16), (16, 64))
TARGET_CLASSES: tuple[int, ...] = (1, 3)


def tiny_components_by_class(
    mask: np.ndarray,
    *,
    normalized_area_factor: float = 1.0,
) -> dict[int, dict[int, list[np.ndarray]]]:
    """Group component pixels by area after mapping to the fold's modal width."""
    if mask.ndim != 2:
        raise ValueError(f"expected a 2-D class mask, got shape {mask.shape}")
    if not math.isfinite(normalized_area_factor) or normalized_area_factor <= 0:
        raise ValueError("normalized_area_factor must be finite and positive")
    output: dict[int, dict[int, list[np.ndarray]]] = {
        class_id: {bin_id: [] for bin_id in range(len(AREA_BINS))}
        for class_id in TARGET_CLASSES
    }
    for class_id in TARGET_CLASSES:
        binary = np.ascontiguousarray(mask == class_id, dtype=np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8, ltype=cv2.CV_32S
        )
        for component_id in range(1, count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            normalized_area = area * normalized_area_factor
            bin_id = next(
                (idx for idx, (low, high) in enumerate(AREA_BINS)
                 if low <= normalized_area < high),
                None,
            )
            if bin_id is None:
                continue
            x = int(stats[component_id, cv2.CC_STAT_LEFT])
            y = int(stats[component_id, cv2.CC_STAT_TOP])
            width = int(stats[component_id, cv2.CC_STAT_WIDTH])
            height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            local_y, local_x = np.where(
                labels[y:y + height, x:x + width] == component_id
            )
            coords = np.stack((local_y + y, local_x + x), axis=1).astype(
                np.int32, copy=False
            )
            if coords.shape[0] != area:
                raise RuntimeError("connected-component pixel count mismatch")
            output[class_id][bin_id].append(coords)
    return output


def choose_tiny_component_anchor(
    components: dict[int, dict[int, list[np.ndarray]]],
    rng: random.Random,
) -> Optional[tuple[int, int, int, int, int]]:
    """Choose class, size bin, component, and an interior pixel uniformly.

    Classes present in the current image are equally likely. Within a class,
    nonempty size bins are equally likely, then connected components within a
    bin are equally likely. The returned tuple is ``(class, bin, index, y, x)``.
    """
    classes = [
        class_id for class_id in TARGET_CLASSES
        if any(components[class_id][bin_id] for bin_id in range(len(AREA_BINS)))
    ]
    if not classes:
        return None
    class_id = rng.choice(classes)
    bins = [
        bin_id for bin_id in range(len(AREA_BINS))
        if components[class_id][bin_id]
    ]
    bin_id = rng.choice(bins)
    component_index = rng.randrange(len(components[class_id][bin_id]))
    pixels = components[class_id][bin_id][component_index]
    y, x = pixels[rng.randrange(len(pixels))]
    return class_id, bin_id, component_index, int(y), int(x)


class ComponentBalancedPatchDataset(KomapPatchDataset):
    """Matched default sampler control or component-balanced treatment."""

    ARM = "control"

    def __init__(self, *args, **kwargs):
        if self.ARM not in {"control", "candidate"}:
            raise ValueError(f"invalid Model410 sampler arm: {self.ARM}")
        super().__init__(*args, **kwargs)
        self.sampler_arm = self.ARM
        self.tiny_components = (
            [
                tiny_components_by_class(
                    mask,
                    normalized_area_factor=(
                        (self.target_width / self.images[img_idx].shape[1]) ** 2
                        if self.target_width > 0 else 1.0
                    ),
                )
                for img_idx, mask in enumerate(self.class_masks)
            ]
            if self.sampler_arm == "candidate" else []
        )

    @staticmethod
    def _uniform_crop(h: int, w: int, ps: int, rng: random.Random) -> tuple[int, int]:
        return rng.randint(0, h - ps), rng.randint(0, w - ps)

    @staticmethod
    def _anchor_crop(
        y: int, x: int, h: int, w: int, ps: int, rng: random.Random
    ) -> tuple[int, int]:
        jitter = ps // 4
        top = y - ps // 2 + rng.randint(-jitter, jitter)
        left = x - ps // 2 + rng.randint(-jitter, jitter)
        return int(min(max(top, 0), h - ps)), int(min(max(left, 0), w - ps))

    def _sample_top_left(
        self,
        img_idx: int,
        h: int,
        w: int,
        rng: Optional[random.Random] = None,
        crop_size: Optional[int] = None,
    ) -> tuple[int, int]:
        if self.sampler_arm == "control":
            return super()._sample_top_left(img_idx, h, w, rng, crop_size)

        ps = self.crop_size if crop_size is None else crop_size
        rng = self.rng if rng is None else rng
        if rng.random() >= self.positive_sample_prob:
            return self._uniform_crop(h, w, ps, rng)

        anchor = choose_tiny_component_anchor(self.tiny_components[img_idx], rng)
        if anchor is not None:
            _, _, _, y, x = anchor
            return self._anchor_crop(y, x, h, w, ps, rng)

        # Preserve the existing rare-pixel positive sampler on images without
        # an eligible modal-width-equivalent target component; do not consume
        # a second branch draw.
        coords = self.minority_coords[img_idx]
        if not coords:
            return self._uniform_crop(h, w, ps, rng)
        keys = list(coords.keys())
        weights = [1.0] * len(keys)
        class_id = rng.choices(keys, weights=weights, k=1)[0]
        points = coords[class_id]
        y, x = points[rng.randrange(len(points))]
        return self._anchor_crop(int(y), int(x), h, w, ps, rng)


def install_sampler(arm: str) -> None:
    """Inject this dataset class into the unchanged Model392 training entrypoint."""
    if arm not in {"control", "candidate"}:
        raise ValueError(f"unknown Model410 arm: {arm}")
    import komap_seg.dataset as dataset_module

    ComponentBalancedPatchDataset.ARM = arm
    dataset_module.KomapPatchDataset = ComponentBalancedPatchDataset
