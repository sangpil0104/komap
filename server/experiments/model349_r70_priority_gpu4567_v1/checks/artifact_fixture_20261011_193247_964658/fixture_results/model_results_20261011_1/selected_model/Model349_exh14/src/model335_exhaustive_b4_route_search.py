# Model349 inference dependency; extracted from the recorded source snapshot.

# Unused training/search entrypoints are intentionally omitted.

from __future__ import annotations


import numpy as np

from model187_schedule_branch_blend import undo_model181_primary_blend

from model198_fast_branch_search import bilateral_weights

from model198_fast_branch_search import predict

from model198_fast_branch_search import replace_classes

def make_mixed(row: dict, setting: dict) -> np.ndarray:
    mixed = row["base"]
    # Preserve Model270's established replacement order and renormalization.
    mixed = replace_classes(
        mixed, row["sources"][setting["primary_source"]], (1,),
        setting["primary_gamma"])
    a_source, a_gamma = setting["al3ni_source"], setting["al3ni_gamma"]
    e_source, e_gamma = setting["eutectic_source"], setting["eutectic_gamma"]
    if a_source == e_source and a_gamma == e_gamma:
        mixed = replace_classes(mixed, row["sources"][a_source], (0, 3), a_gamma)
    else:
        mixed = replace_classes(mixed, row["sources"][a_source], (0,), a_gamma)
        mixed = replace_classes(mixed, row["sources"][e_source], (3,), e_gamma)
    al_gamma = float(setting.get("al_gamma", 0.0))
    if al_gamma:
        source = row["sources"][setting["al_source"]]
        mixed[2] = (1.0 - al_gamma) * mixed[2] + al_gamma * source[2]
        mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)
    return mixed
