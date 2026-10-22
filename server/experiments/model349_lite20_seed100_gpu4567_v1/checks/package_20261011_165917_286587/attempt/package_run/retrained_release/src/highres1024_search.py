# Model349 inference dependency; extracted from the recorded source snapshot.

# Unused training/search entrypoints are intentionally omitted.

from __future__ import annotations


import numpy as np

MATRIX = np.asarray([
    [0.35, 0.40, 0.25], [0.25, 0.35, 0.40],
    [0.50, 0.30, 0.20], [0.25, 0.55, 0.20],
], dtype=np.float32)

def choose_source(mean: float, args):
    if 159.0 <= mean < 161.0:
        return args.source_pairwise
    if 161.3 <= mean < 162.0:
        return args.source_ema
    if 174.6 <= mean < 175.1:
        return args.source_model109
    if 152.3 <= mean < 152.7:
        return args.source_seed456
    if 155.2 <= mean < 155.7:
        return args.source_ema
    return args.source_restart51
