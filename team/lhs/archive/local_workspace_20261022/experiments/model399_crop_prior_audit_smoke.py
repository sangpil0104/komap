#!/usr/bin/env python3
"""Small helper checks; run this with the dedicated komap environment."""

import numpy as np

from model399_crop_prior_audit import class_hist, rect_sum, shares


def main() -> None:
    mask = np.asarray([[0, 1, 1], [2, 2, 3], [0, 3, 3]], dtype=np.uint8)
    hist = class_hist(mask)
    assert hist.tolist() == [2, 2, 2, 3]
    assert np.isclose(sum(shares(hist)), 1.0)
    integral = np.pad(mask == 3, ((1, 0), (1, 0))).astype(np.int32).cumsum(0).cumsum(1)
    assert rect_sum(integral, 1, 1, 2) == 3
    print("Model399 crop-prior audit smoke: PASS")


if __name__ == "__main__":
    main()
