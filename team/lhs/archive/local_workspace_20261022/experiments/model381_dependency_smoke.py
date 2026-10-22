#!/usr/bin/env python3
"""Check OpenCV precise Euclidean distance transform for Model381."""

from __future__ import annotations

import json

import cv2
import numpy as np


def main() -> None:
    source = np.ones((9, 9), dtype=np.uint8)
    source[4, 4] = 0
    distance = cv2.distanceTransform(source, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    checks = {
        "opencv_version": cv2.__version__,
        "distance_one": float(distance[4, 5]),
        "distance_sqrt2": float(distance[5, 5]),
        "distance_two": float(distance[4, 6]),
        "distance_sqrt5": float(distance[5, 6]),
    }
    if not (np.isclose(checks["distance_one"], 1.0)
            and np.isclose(checks["distance_sqrt2"], np.sqrt(2.0))
            and np.isclose(checks["distance_two"], 2.0)
            and np.isclose(checks["distance_sqrt5"], np.sqrt(5.0))):
        raise AssertionError(f"OpenCV distance transform is not precise: {checks}")
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    main()
