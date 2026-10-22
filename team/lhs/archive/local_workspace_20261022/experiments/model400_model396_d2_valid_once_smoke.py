#!/usr/bin/env python3
"""Dataset-free safety/metric smoke tests for the Model400 one-shot scorer."""

from __future__ import annotations

from pathlib import Path
import tempfile

import model400_model396_d2_valid_once as model400


def main() -> None:
    model400.self_test()
    with tempfile.TemporaryDirectory(prefix="model400-smoke-") as temp:
        root = Path(temp)
        protocol = root / model400.PROTOCOL
        protocol.parent.mkdir(parents=True)
        protocol.write_text("Status: draft only; not authorized to score.\n", encoding="utf-8")
        try:
            model400._protocol_reviewed(root)
        except PermissionError:
            pass
        else:
            raise AssertionError("draft protocol did not block valid access")

        protocol.write_text(
            f"{model400.APPROVED_PROTOCOL_STATUS}\n", encoding="utf-8"
        )
        model400._protocol_reviewed(root)

        claim = root / "one_shot.claim"
        model400._claim(claim, "synthetic-lock", "synthetic-score-stage")
        try:
            model400._claim(claim, "synthetic-lock", "synthetic-score-stage")
        except FileExistsError:
            pass
        else:
            raise AssertionError("one-shot claim allowed a duplicate attempt")

    print("Model400 safety smoke passed; no dataset, checkpoint, valid score, or GPU accessed.")


if __name__ == "__main__":
    main()
