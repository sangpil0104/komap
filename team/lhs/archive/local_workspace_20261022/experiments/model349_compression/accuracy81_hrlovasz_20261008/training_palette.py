"""RGB mask color <-> class index mapping.

Class order is fixed; it matches the competition mask palette (see README).
"""

import numpy as np

CLASS_NAMES = ["Al", "Al3Ni", "Primary_Si", "Eutectic_Si"]

PALETTE = np.array(
    [
        (153, 127, 76),  # 0 Al
        (25, 76, 153),  # 1 Al3Ni
        (76, 178, 76),  # 2 Primary Si
        (204, 204, 204),  # 3 Eutectic Si
    ],
    dtype=np.uint8,
)

NUM_CLASSES = len(CLASS_NAMES)


def rgb_to_class(mask_rgb: np.ndarray) -> np.ndarray:
    """Convert an (H, W, 3) uint8 RGB mask into an (H, W) uint8 class-index mask.

    Uses nearest-color matching against PALETTE so it stays robust to any
    off-palette pixels (e.g. resize interpolation artifacts), even though
    the raw dataset masks only contain the exact palette colors.
    """
    flat = mask_rgb.reshape(-1, 3).astype(np.int32)
    palette = PALETTE.astype(np.int32)
    # (N, num_classes) squared distance to each palette color
    dists = ((flat[:, None, :] - palette[None, :, :]) ** 2).sum(axis=-1)
    assert np.all(dists.min(axis=1) == 0), "unknown training mask palette color"
    class_idx = dists.argmin(axis=1).astype(np.uint8)
    return class_idx.reshape(mask_rgb.shape[:2])


def class_to_rgb(class_mask: np.ndarray) -> np.ndarray:
    """Convert an (H, W) class-index mask back into an (H, W, 3) uint8 RGB image."""
    return PALETTE[class_mask]
