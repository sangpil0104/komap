#!/usr/bin/env python3
"""Synthetic-only regression tests for the Model388 sampler treatment."""

from __future__ import annotations

import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model336_hrnet_component_candidate"))

from komap_seg.dataset import KomapPatchDataset  # noqa: E402
from komap_seg.palette import PALETTE  # noqa: E402


def write_fixture(root: Path, *, eligible: bool) -> tuple[Path, Path]:
    images = root / "images"
    masks = root / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    class_ids = np.zeros((600, 600), dtype=np.uint8)
    if eligible:
        # Separated 8-connected components in four size bins.
        for top, left, side in ((10, 10, 8), (10, 35, 10), (10, 65, 16), (10, 105, 20)):
            class_ids[top : top + side, left : left + side] = 1
        for top, left, side in ((300, 10, 10), (300, 40, 15)):
            class_ids[top : top + side, left : left + side] = 3
        class_ids[80:86, 10:16] = 1  # 36 px tiny component
        class_ids[80:85, 40:45] = 3  # 25 px tiny component
    else:
        class_ids[10:16, 10:16] = 1
        class_ids[30:35, 30:35] = 3
    image = np.full((600, 600, 3), 128, dtype=np.uint8)
    rgb_mask = PALETTE[class_ids]
    Image.fromarray(image).save(images / "A001_step1_image.png")
    Image.fromarray(rgb_mask).save(masks / "A001_step1_mask.png")
    return images, masks


def make_dataset(images: Path, masks: Path, seed: int, sampler: str) -> KomapPatchDataset:
    return KomapPatchDataset(
        images,
        masks,
        patch_size=512,
        input_scale=2.0,
        samples_per_image=12,
        seed=seed,
        minority_sampler=sampler,
        transform=lambda **kwargs: kwargs,
    )


def test_legacy_fallback_is_bitwise() -> None:
    with tempfile.TemporaryDirectory(prefix="model395-fallback-") as temp:
        images, masks = write_fixture(Path(temp), eligible=False)
        legacy = make_dataset(images, masks, 395, "legacy")
        candidate = make_dataset(images, masks, 395, "model388")
        assert legacy.rng.getstate() == candidate.rng.getstate()
        for _ in range(5000):
            assert legacy._sample_top_left(0, 600, 600) == candidate._sample_top_left(0, 600, 600)
        assert legacy.rng.getstate() == candidate.rng.getstate()


def test_component_and_pixel_uniformity() -> None:
    with tempfile.TemporaryDirectory(prefix="model395-components-") as temp:
        images, masks = write_fixture(Path(temp), eligible=True)
        dataset = make_dataset(images, masks, 395, "model388")
        class_mask = dataset.class_masks[0]
        component_info: dict[int, dict[int, int]] = {}
        labels_by_class: dict[int, np.ndarray] = {}
        component_pixels: dict[tuple[int, int], dict[tuple[int, int], int]] = {}
        for cls in (1, 3):
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                (class_mask == cls).astype(np.uint8), connectivity=8, ltype=cv2.CV_32S
            )
            del count
            labels_by_class[cls] = labels
            component_info[cls] = {
                label: int(stats[label, cv2.CC_STAT_AREA])
                for label in range(1, len(stats))
                if int(stats[label, cv2.CC_STAT_AREA]) >= 64
            }
            for label, area in component_info[cls].items():
                ys, xs = np.where(labels == label)
                assert len(ys) == area
                component_pixels[(cls, label)] = {
                    (int(y), int(x)): index for index, (y, x) in enumerate(zip(ys, xs))
                }

        selected_components: dict[int, Counter[int]] = {1: Counter(), 3: Counter()}
        selected_pixels: dict[tuple[int, int], np.ndarray] = {
            key: np.zeros(component_info[key[0]][key[1]], dtype=np.int64)
            for key in component_pixels
        }
        rng = random.Random(2026)
        positive = Counter()
        draws = 120_000
        for _ in range(draws):
            top, left, details = dataset._sample_top_left(
                0, 600, 600, rng=rng, return_details=True
            )
            if not details["positive"]:
                continue
            cls = details["class_id"]
            y, x = details["anchor_y"], details["anchor_x"]
            assert top <= y < top + dataset.crop_size
            assert left <= x < left + dataset.crop_size
            positive[cls] += 1
            label = int(labels_by_class[cls][y, x])
            area = component_info[cls].get(label, 0)
            assert area >= 64, f"selected tiny component: class={cls}, area={area}"
            selected_components[cls][area] += 1
            pixel_index = component_pixels[(cls, label)][(y, x)]
            selected_pixels[(cls, label)][pixel_index] += 1

        assert 0.49 <= sum(positive.values()) / draws <= 0.51
        assert 0.485 <= positive[1] / sum(positive.values()) <= 0.515
        assert 0.485 <= positive[3] / sum(positive.values()) <= 0.515
        for cls in (1, 3):
            areas = list(component_info[cls].values())
            expected = positive[cls] / len(areas)
            for area in areas:
                assert abs(selected_components[cls][area] - expected) <= 0.08 * expected
        for key, counts in selected_pixels.items():
            mean = float(counts.mean())
            # The largest fixture component receives only ~19 draws/pixel;
            # allow finite-sample Poisson variation while rejecting gross bias.
            assert mean > 0 and float(counts.std() / mean) < 0.35, key


def main() -> None:
    test_legacy_fallback_is_bitwise()
    test_component_and_pixel_uniformity()
    print("Model395 component sampler synthetic smoke: PASS")


if __name__ == "__main__":
    main()
