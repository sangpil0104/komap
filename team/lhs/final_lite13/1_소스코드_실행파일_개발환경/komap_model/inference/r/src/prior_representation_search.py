# R ensemble inference dependency.

# Unused training/search entrypoints are intentionally omitted.

from __future__ import annotations


import numpy as np

def percentile_image(gray):
    flat = gray.reshape(-1)
    order = np.argsort(flat, kind="stable")
    result = np.empty(flat.shape, dtype=np.float32)
    result[order] = np.linspace(0.0, 1.0, len(flat), dtype=np.float32)
    return result.reshape(gray.shape)
