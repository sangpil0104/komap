from pathlib import Path
import numpy as np
from PIL import Image

PALETTE = np.asarray(
    [(76, 178, 76), (25, 76, 153), (204, 204, 204), (153, 127, 76)],
    dtype=np.uint8,
)


def decode_mask(path: str | Path) -> np.ndarray:
    """RGB/RGBA palette PNG -> int64 [H,W]. Unknown colors raise."""
    with Image.open(path) as image:
        if "A" in image.getbands():
            alpha = np.asarray(image.getchannel("A"))
            if np.any(alpha != 255):
                raise ValueError(f"Non-opaque annotation: {path}")
        rgb = np.asarray(image.convert("RGB"))
    labels = np.full(rgb.shape[:2], 255, dtype=np.int64)
    for class_id, color in enumerate(PALETTE):
        labels[np.all(rgb == color, axis=-1)] = class_id
    unknown = labels == 255
    if unknown.any():
        examples = np.unique(rgb[unknown], axis=0)[:8].tolist()
        raise ValueError(f"Unknown mask colors in {path}: {examples}")
    return labels


def save_mask(labels: np.ndarray, path: str | Path, rgba: bool = True) -> None:
    """Integer [H,W] -> exact-palette lossless PNG."""
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Expected a two-dimensional integer label array")
    if labels.size == 0 or labels.min() < 0 or labels.max() >= len(PALETTE):
        raise ValueError("Label values must be in [0,3]; ignore is not exportable")
    rgb = PALETTE[labels]
    output = rgb
    if rgba:
        alpha = np.full((*labels.shape, 1), 255, dtype=np.uint8)
        output = np.concatenate([rgb, alpha], axis=-1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output).save(path, format="PNG")
