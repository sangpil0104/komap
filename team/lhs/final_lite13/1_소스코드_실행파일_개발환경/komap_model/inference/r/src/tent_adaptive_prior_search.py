# R ensemble inference dependency.

# Unused training/search entrypoints are intentionally omitted.

from __future__ import annotations


import numpy as np

def gaussian_prior(z, stats):
    return np.asarray([-0.5 * ((z - mean) / std) ** 2 - np.log(std)
                       for mean, std in stats], dtype=np.float32)
