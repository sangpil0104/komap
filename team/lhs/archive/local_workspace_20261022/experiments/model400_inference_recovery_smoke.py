#!/usr/bin/env python3
"""Synthetic-only checks for Model400's partial-inference recovery helper."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

import model400_inference_recovery as recovery


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="model400-recovery-smoke-") as temp:
        root = Path(temp)
        image_id = "synthetic"
        probabilities = np.zeros((4, 3, 5), dtype=np.float32)
        probabilities[2] = 1.0
        map_path = root / "synthetic.npz"
        np.savez_compressed(map_path, **{image_id: probabilities.astype(np.float16)})
        recovered = recovery._read_map(map_path, image_id, (3, 5))
        assert recovered.shape == (4, 3, 5)
        assert np.allclose(recovered[2], 1.0)

        receipt = root / "claim.json"
        recovery._exclusive_json(receipt, {"stage": "synthetic"})
        try:
            recovery._exclusive_json(receipt, {"stage": "synthetic"})
        except FileExistsError:
            pass
        else:
            raise AssertionError("recovery receipt allowed overwrite")

    print("Model400 recovery smoke passed; no dataset, checkpoint, valid score, or GPU accessed.")


if __name__ == "__main__":
    main()
