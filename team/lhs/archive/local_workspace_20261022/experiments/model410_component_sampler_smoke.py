#!/usr/bin/env python3
"""CPU-only deterministic tests for Model410's component sampler."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model336_hrnet_widthnorm_candidate"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model410_component_uniform_sampler import (  # noqa: E402
    AREA_BINS,
    ComponentBalancedPatchDataset,
    choose_tiny_component_anchor,
    tiny_components_by_class,
)
from komap_seg.dataset import KomapPatchDataset  # noqa: E402


def main() -> None:
    mask = np.zeros((96, 128), dtype=np.uint8)
    # Disjoint 8-connected components in three fixed normalized-area bins plus one
    # large component that must never be chosen by the treatment sampler.
    mask[4, 4] = 1
    mask[10:12, 10:12] = 1
    mask[20:24, 20:24] = 1
    mask[40:50, 40:50] = 1
    mask[60:72, 60:72] = 1  # area 144, excluded
    mask[5:7, 100:102] = 3
    components = tiny_components_by_class(mask)
    assert [len(components[1][i]) for i in range(3)] == [1, 1, 1]
    assert [len(components[3][i]) for i in range(3)] == [0, 1, 0]
    assert tuple(AREA_BINS) == ((1, 4), (4, 16), (16, 64))

    # The same modal-width-equivalent area should land in the same bin despite
    # different source-image widths (the real folds contain widths 675–1440).
    narrow_mask = np.zeros((24, 24), dtype=np.uint8)
    narrow_mask[3, 3:5] = 1  # 2 native px at width 675
    wide_mask = np.zeros((24, 24), dtype=np.uint8)
    wide_mask[3:6, 3:6] = 1  # 9 native px at width 1440
    narrow = tiny_components_by_class(
        narrow_mask, normalized_area_factor=(1278 / 675) ** 2
    )
    wide = tiny_components_by_class(
        wide_mask, normalized_area_factor=(1278 / 1440) ** 2
    )
    assert len(narrow[1][1]) == len(wide[1][1]) == 1

    first_rng = random.Random(410)
    second_rng = random.Random(410)
    first = [choose_tiny_component_anchor(components, first_rng) for _ in range(3000)]
    second = [choose_tiny_component_anchor(components, second_rng) for _ in range(3000)]
    assert first == second, "same seed must reproduce all component choices"
    class_counts = {class_id: sum(item[0] == class_id for item in first) for class_id in (1, 3)}
    assert abs(class_counts[1] - class_counts[3]) < 250, class_counts
    bin_counts = {
        bin_id: sum(item[0] == 1 and item[1] == bin_id for item in first)
        for bin_id in range(3)
    }
    assert max(bin_counts.values()) - min(bin_counts.values()) < 120, bin_counts
    for class_id, bin_id, component_index, y, x in first:
        pixels = components[class_id][bin_id][component_index]
        low, high = AREA_BINS[bin_id]
        assert low <= len(pixels) < high
        assert np.any((pixels[:, 0] == y) & (pixels[:, 1] == x))

    # The control's method dispatches to the original base implementation
    # byte-for-byte; identical per-item RNGs must yield identical crop origins.
    base = KomapPatchDataset.__new__(KomapPatchDataset)
    base.crop_size = 32
    base.positive_sample_prob = 0.5
    base.rng = random.Random(99)
    base.minority_coords = [{1: np.asarray([[40, 40], [41, 42]]), 3: np.asarray([[70, 70]])}]
    control = ComponentBalancedPatchDataset.__new__(ComponentBalancedPatchDataset)
    control.sampler_arm = "control"
    control.crop_size = base.crop_size
    control.positive_sample_prob = base.positive_sample_prob
    control.rng = random.Random(99)
    control.minority_coords = base.minority_coords
    base_xy = base._sample_top_left(0, 128, 128, random.Random(410), 32)
    control_xy = control._sample_top_left(0, 128, 128, random.Random(410), 32)
    assert base_xy == control_xy, (base_xy, control_xy)

    candidate = ComponentBalancedPatchDataset.__new__(ComponentBalancedPatchDataset)
    candidate.sampler_arm = "candidate"
    candidate.crop_size = 32
    candidate.positive_sample_prob = 1.0
    candidate.tiny_components = [components]
    candidate.minority_coords = [{1: np.argwhere(mask == 1), 3: np.argwhere(mask == 3)}]
    for seed in range(100):
        top, left = candidate._sample_top_left(
            0, mask.shape[0], mask.shape[1], random.Random(seed), 32
        )
        assert 0 <= top <= mask.shape[0] - 32
        assert 0 <= left <= mask.shape[1] - 32
        crop = mask[top:top + 32, left:left + 32]
        assert np.any((crop == 1) | (crop == 3)), "positive component anchor was cropped out"
    print(
        "Model410 width-normalized component sampler smoke passed: "
        f"class_counts={class_counts}, size_bin_counts={bin_counts}"
    )


if __name__ == "__main__":
    main()
