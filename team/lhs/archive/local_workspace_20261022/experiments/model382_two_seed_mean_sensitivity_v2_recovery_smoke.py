#!/usr/bin/env python3
"""Synthetic-only test for the narrowly scoped v2 checkpoint-path repair."""

from pathlib import Path

from model382_two_seed_mean_sensitivity_v2_recovery import normalize_checkpoint_path


def main() -> None:
    wrong = Path("/workspace/komap_model1/outputs/model382/seedseed1/fold3/last.pt")
    assert normalize_checkpoint_path(wrong) == Path(
        "/workspace/komap_model1/outputs/model382/seed1/fold3/last.pt"
    )
    assert normalize_checkpoint_path(Path("/tmp/seed1/fold3/last.pt")) == Path(
        "/tmp/seed1/fold3/last.pt"
    )
    assert normalize_checkpoint_path(Path("/tmp/seedseed2/fold1/last.pt")) == Path(
        "/tmp/seed2/fold1/last.pt"
    )
    print("Model382 path-recovery synthetic smoke: PASS")


if __name__ == "__main__":
    main()
