#!/usr/bin/env python3
"""Synthetic-only scale equivariance test; never opens KoMaP data."""

from __future__ import annotations

import cv2
import numpy as np

from model394_image_scale_mask_association import apparent_wavelength


def main() -> None:
    size = 640
    period = 16.0
    x = np.arange(size, dtype=np.float32)[None, :]
    stripe = 127.0 + 80.0 * np.cos(2.0 * np.pi * x / period)
    gray = np.repeat(stripe, size, axis=0)
    rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.uint8)

    estimates = {}
    for scale in (0.5, 1.0, 2.0):
        if scale == 1.0:
            image = rgb
        else:
            interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            image = cv2.resize(
                rgb,
                None,
                fx=scale,
                fy=scale,
                interpolation=interpolation,
            )
        estimates[scale] = apparent_wavelength(image)
        expected = period * scale
        relative_error = abs(estimates[scale] - expected) / expected
        assert relative_error <= 0.10, (scale, estimates[scale], expected)

    assert abs(estimates[0.5] / estimates[1.0] - 0.5) <= 0.15
    assert abs(estimates[2.0] / estimates[1.0] - 2.0) <= 0.15
    print("Model394 synthetic-only apparent-scale smoke: PASS")
    print("estimated wavelengths:", {k: round(v, 3) for k, v in estimates.items()})


if __name__ == "__main__":
    main()
