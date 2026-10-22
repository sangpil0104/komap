# R ensemble inference dependency.

# Unused training/search entrypoints are intentionally omitted.

from __future__ import annotations


import numpy as np

def cleanup_primary(prediction, logits, threshold=128):
    from scipy import ndimage
    mask = prediction == 1
    labels, count = ndimage.label(mask)
    if count == 0:
        return prediction
    sizes = np.bincount(labels.reshape(-1))
    small = mask & (sizes[labels] <= threshold)
    if not small.any():
        return prediction
    other = logits.copy(); other[1] = -np.inf
    output = prediction.copy(); output[small] = other[:, small].argmax(axis=0).astype(np.uint8)
    return output
