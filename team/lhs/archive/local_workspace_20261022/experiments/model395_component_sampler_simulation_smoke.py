#!/usr/bin/env python3
"""Synthetic-only test for worker-faithful crop exposure simulation."""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from model395_component_sampler_simulation import simulate_fold  # noqa: E402
from model395_component_sampler_smoke import write_fixture  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="model395-sim-smoke-") as temp:
        fold_root = Path(temp) / "train_fold3"
        images, masks = write_fixture(fold_root, eligible=True)
        shutil.copy2(images / "A001_step1_image.png", images / "A002_step2_image.png")
        shutil.copy2(masks / "A001_step1_mask.png", masks / "A002_step2_mask.png")
        first = simulate_fold(Path(temp), 3, epochs=10)
        second = simulate_fold(Path(temp), 3, epochs=10)
        assert first["arms"] == second["arms"]
        assert first["candidate_to_legacy_ratios"] == second["candidate_to_legacy_ratios"]
        assert first["arms"]["legacy"]["epochs"] == 10
        assert first["arms"]["model388"]["workers"] == 4
        assert first["arms"]["model388"]["persistent_workers"] is False
        assert first["arms"]["model388"]["worker_private_rng_reseeded_by_train_worker_init"] is False
        assert first["arms"]["model388"]["images"] == 2
        print("Model395 realized-dose simulation synthetic smoke: PASS")


if __name__ == "__main__":
    main()
