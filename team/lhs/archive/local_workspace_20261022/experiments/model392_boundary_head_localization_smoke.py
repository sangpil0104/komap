#!/usr/bin/env python3
"""Synthetic checks for the Model392 boundary-head localization probe."""

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from model392_boundary_head_localization import (
    average_precision,
    boundary_tta_batch,
    predict_boundary_map,
    summarize_records,
    tile_starts,
)


class ToyBoundaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()
        self.segmentation_head = nn.Conv2d(3, 4, kernel_size=1)
        self.boundary_head = nn.Conv2d(3, 1, kernel_size=3, padding=1)


def toy_forward_with_boundary(model, batch):
    features = model.encoder(batch)
    decoded = model.decoder(features)
    return model.segmentation_head(decoded), model.boundary_head(decoded)


class LocalizationSmoke(unittest.TestCase):
    def test_tile_starts_cover_full_length(self):
        for length, tile, stride in ((1200, 256, 192), (512, 512, 384), (100, 256, 192)):
            starts = tile_starts(length, tile, stride)
            self.assertEqual(starts[0], 0)
            self.assertGreaterEqual(starts[-1] + tile, max(length, tile))
            self.assertTrue(all(b - a <= tile for a, b in zip(starts, starts[1:])))

    def test_average_precision_threshold_free_ordering(self):
        target = np.asarray([[1, 1, 0, 0]], dtype=bool)
        score = np.asarray([[0.9, 0.8, 0.2, 0.1]], dtype=np.float32)
        self.assertAlmostEqual(average_precision(target, score), 1.0)
        self.assertGreater(
            average_precision(target, score),
            average_precision(target, score[:, ::-1]),
        )

    def test_flip4_boundary_forward_shape_and_finiteness(self):
        torch.manual_seed(392081)
        model = ToyBoundaryModel().eval()
        batch = torch.randn(2, 3, 32, 32)
        result = boundary_tta_batch(model, batch, toy_forward_with_boundary)
        self.assertEqual(tuple(result.shape), (2, 1, 32, 32))
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(((result >= 0) & (result <= 1)).all())

    def test_tiled_map_shape_and_complete_coverage(self):
        torch.manual_seed(392081)
        model = ToyBoundaryModel().eval()
        image = torch.randn(3, 45, 49)
        scores, coverage, tile_count = predict_boundary_map(
            model=model,
            image=image,
            patch_size=32,
            input_scale=1.0,
            device=torch.device("cpu"),
            batch_size=2,
            overlap_ratio=0.25,
            forward_with_boundary=toy_forward_with_boundary,
        )
        self.assertEqual(scores.shape, (45, 49))
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(coverage, 1.0)
        self.assertGreater(tile_count, 1)

    def test_group_summary_and_fixed_decision_schema(self):
        rows = []
        for fold in (1, 2, 3, 4):
            for seed in (1, 2):
                for i in range(3):
                    rows.append({
                        "image_id": f"A{fold * 10 + i}_image",
                        "group": f"A{fold * 10 + i}",
                        "fold": fold,
                        "seed": seed,
                        "average_precision": 0.8,
                        "prevalence": 0.2,
                    })
        summary = summarize_records(rows)
        self.assertEqual(summary["images"], 12)
        self.assertEqual(summary["specimen_groups"], 12)
        self.assertEqual(summary["decision"], "LOCALIZATION_SIGNAL")
        self.assertEqual(len(summary["ap_minus_prevalence_group_bootstrap_95ci_pp"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
