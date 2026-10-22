#!/usr/bin/env python3
"""Synthetic-only tests for Model393 mask morphology descriptors."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from model393_synthetic_mask_feasibility_audit import (
    PALETTE,
    adjacency_counts,
    component_summary,
    decode_mask,
    distance_width_summary,
    two_point,
)


def main() -> None:
    labels = np.zeros((8, 12), dtype=np.uint8)
    labels[:, 6:] = 3
    rgb = PALETTE[labels]
    encoded = BytesIO()
    Image.fromarray(rgb).save(encoded, format="PNG")
    decoded = decode_mask(encoded.getvalue(), "synthetic")
    assert np.array_equal(decoded, labels)

    components = component_summary(labels == 3)
    assert components["components"] == 1
    assert components["area_px"] == [48.0]

    adjacency = adjacency_counts(labels)
    assert adjacency[0, 3] == adjacency[3, 0] == 8
    assert int(adjacency.sum()) == 16

    width = distance_width_summary(labels == 3)
    assert width["pixels"] == 48
    assert width["width_proxy_px_q"][0] > 0

    correlations = two_point(labels == 3)
    assert correlations["x1"] > correlations["x8"]
    assert correlations["y1"] == 0.5
    print("Model393 synthetic morphology smoke: PASS (no train/valid/test data read)")


if __name__ == "__main__":
    main()
