#!/usr/bin/env python3
"""Run the synthetic evaluator smoke test against the copied Colab source."""

from __future__ import annotations

import sys
from pathlib import Path


REMOTE_EXPERIMENTS = Path(
    "/content/komap_model1/model377_directional/experiments"
)
sys.path.insert(0, str(REMOTE_EXPERIMENTS))
import model377_oof_evaluation_smoke  # noqa: E402

model377_oof_evaluation_smoke.main()
