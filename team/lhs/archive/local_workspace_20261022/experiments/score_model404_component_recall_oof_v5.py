#!/usr/bin/env python3
"""Scoring-only fix for the pretraining Model404 v4 paired OOF evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_model404_component_recall_oof_v4 as v4


base_hit_rate_delta_bootstrap = v4.base_hit_rate_delta_bootstrap


def fixed_hit_rate_delta_bootstrap(candidate, control, samples, bin_name):
    result = base_hit_rate_delta_bootstrap(candidate, control, samples, bin_name)
    class_name = next(
        name for name in v4.TARGET_CLASSES.values()
        if bin_name.startswith(name + "_")
    )
    size_name = bin_name[len(class_name) + 1:]
    interval = v4.AREA_BINS[size_name]
    result["hit_definition"] = (
        f"at least 50% of native-grid GT component pixels predicted as {class_name}; "
        f"area [{interval[0]},{interval[1]})"
    )
    return result


v4.hit_rate_delta_bootstrap = fixed_hit_rate_delta_bootstrap
base_score = v4.score


def score(*args, **kwargs):
    result = base_score(*args, **kwargs)
    result["experiment"] = "Model404 v5 Al3Ni + Eutectic-Si component recall fine-tuning"
    result["scorer_revision"] = "v5 fixes target-class label formatting; training objective unchanged"
    return result


v4.score = score


if __name__ == "__main__":
    v4.main()
