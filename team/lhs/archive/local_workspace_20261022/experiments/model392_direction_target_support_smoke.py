#!/usr/bin/env python3
"""Synthetic tests for geometric support of same-class direction targets."""

from __future__ import annotations

import unittest

import numpy as np

from model392_direction_target_support import (
    CLASS_NAMES,
    DIRECTION_NAMES,
    boundary_band,
    direction_bins,
    target_support,
)


class DirectionSupportSmoke(unittest.TestCase):
    def test_boundary_band_detects_two_classes(self):
        mask = np.zeros((9, 9), dtype=np.uint8)
        mask[:, 4:] = 1
        band = boundary_band(mask)
        self.assertTrue(band[:, 3:5].any())
        self.assertFalse(band[0, 0])

    def test_cardinal_direction_bins(self):
        dy = np.asarray([-1, -1, 0, 1, 1, 1, 0, -1])
        dx = np.asarray([0, 1, 1, 1, 0, -1, -1, -1])
        bins = direction_bins(dy, dx)
        self.assertEqual([DIRECTION_NAMES[int(b)] for b in bins],
                         ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])

    def test_nearest_interior_direction_has_support(self):
        mask = np.zeros((13, 13), dtype=np.uint8)
        mask[3:10, 3:10] = 1
        result = target_support(mask, 1)
        self.assertEqual(result["class_name"], CLASS_NAMES[1])
        primary = result["support_by_clearance_and_max_offset"]["4"]["16"]
        self.assertGreater(primary["supported_source_pixels"], 0)
        self.assertGreater(primary["direction_counts"]["S"], 0)
        self.assertLessEqual(primary["support_fraction"], 1.0)

    def test_absent_class_is_reported_without_error(self):
        result = target_support(np.zeros((8, 8), dtype=np.uint8), 2)
        self.assertEqual(result["boundary_source_pixels"], 0)
        self.assertEqual(result["interior_seed_pixels_by_clearance"]["4"], 0)
        self.assertEqual(
            result["support_by_clearance_and_max_offset"]["4"]["16"]["supported_source_pixels"],
            0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
