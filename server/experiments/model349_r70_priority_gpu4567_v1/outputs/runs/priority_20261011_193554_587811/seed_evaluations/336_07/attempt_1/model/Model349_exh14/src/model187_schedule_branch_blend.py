# Model349 inference dependency; extracted from the recorded source snapshot.

# Unused training/search entrypoints are intentionally omitted.

from __future__ import annotations


import numpy as np

def undo_model181_primary_blend(item: dict, gamma: float = 0.40) -> np.ndarray:
    """Recover the pre-auxiliary probability from model-182's mixed map."""
    mixed = item["mixed"]
    extra = item["extra"]
    primary = mixed[1]
    scale = (1.0 - gamma + gamma * extra[1]) / (1.0 - gamma + gamma * primary)
    base = mixed * scale[None]
    base[1] = 1.0 - base[0] - base[2] - base[3]
    return base / np.maximum(base.sum(0, keepdims=True), 1e-7)
