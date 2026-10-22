#!/usr/bin/env python3
"""Add a parent-model non-regression gate to the corrected v5 scorer."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_model404_component_recall_oof_v5 as v5  # noqa: E402


base_score = v5.score


def score(*args, **kwargs):
    result = base_score(*args, **kwargs)
    parent_delta = result["paired_candidate_vs_model392_parent"]
    checks = result["preregistered_gate"]["checks"]
    checks["candidate_vs_model392_parent_point_delta_nonnegative"] = (
        parent_delta["delta_image_mean_pp"] >= 0.0
    )
    result["preregistered_gate"]["pass"] = all(checks.values())
    result["parent_nonregression_gate"] = {
        "reference": "fixed Model392 seed1 OOF, the candidate initialization parent",
        "candidate_delta_pp": parent_delta["delta_image_mean_pp"],
        "paired_group_bootstrap_95ci_pp": parent_delta["paired_group_bootstrap_95ci_pp"],
        "required_point_delta_pp": ">= 0.0",
        "pass": checks["candidate_vs_model392_parent_point_delta_nonnegative"],
    }
    result["experiment"] = "Model404 v6 Al3Ni + Eutectic-Si tiny-component recall with parent guard"
    result["scorer_revision"] = (
        "v6 adds a non-regression gate versus the fixed Model392 seed1 OOF parent; "
        "training objective and treatment-control gates are unchanged"
    )
    return result


v5.v4.score = score


if __name__ == "__main__":
    v5.v4.main()
