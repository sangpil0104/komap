"""Synthetic unit smoke for the Model390 paired train-OOF evaluator."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from model390_per_item_rng_oof import (
    BOOTSTRAP_SEED,
    INFERENCE,
    METRIC,
    expected_training_args,
    sha256,
    summarize_final,
    summarize_seed,
    verify_candidate_manifest,
)
from model382_oof_ensemble_compare import class_ious


def make_rows(delta_by_seed: dict[int, float]):
    rows = {1: {}, 2: {}}
    image_index = 0
    base = np.asarray([0.60, 0.50, 0.70, 0.30], dtype=np.float64)
    for group_index in range(1, 60):
        fold = (group_index - 1) % 4 + 1
        image_count = 2 if group_index <= 11 else 1
        for local_index in range(image_count):
            image_index += 1
            name = f"A{group_index}_step{4 if group_index % 4 == 0 else 2}_image{local_index}"
            for seed in (1, 2):
                reference = base.copy()
                candidate = np.clip(base + delta_by_seed[seed], 0.0, 1.0)
                rows[seed][name] = {
                    "group": f"A{group_index}",
                    "fold": fold,
                    "step": 4 if group_index % 4 == 0 else 2,
                    "reference_class_iou": reference,
                    "candidate_class_iou": candidate,
                }
    assert image_index == 70
    return rows


def test_empty_union_zero() -> None:
    probabilities = np.zeros((4, 4, 4), dtype=np.float32)
    probabilities[0] = 1.0
    truth = np.zeros((4, 4), dtype=np.uint8)
    values = class_ious(probabilities, truth)
    assert np.array_equal(values, np.asarray([1.0, 0.0, 0.0, 0.0]))
    assert float(values.mean()) == 0.25


def test_candidate_manifest_verification(tmp: Path) -> None:
    root = tmp / "project"
    run_dir = root / "outputs/model390_per_item_rng/seed1/fold1"
    oof_dir = root / "outputs/model390_per_item_rng/seed1/fold1_oof"
    run_dir.mkdir(parents=True)
    oof_dir.mkdir(parents=True)
    lock_path = root / "model390_lock.json"
    lock_path.write_text(
        json.dumps({"locked_at_utc": "2000-01-01T00:00:00Z"}), encoding="utf-8"
    )
    lock_hash = sha256(lock_path)
    checkpoint = run_dir / "last.pt"
    checkpoint.write_bytes(b"synthetic checkpoint bytes")
    archive = oof_dir / "valid_probabilities.npz"
    np.savez_compressed(archive, sample=np.zeros((4, 2, 2), dtype=np.float16))
    args = expected_training_args(root, run_dir, 1)
    (run_dir / "args.json").write_text(json.dumps(args), encoding="utf-8")
    manifest = {
        "model": 390,
        "seed": 1,
        "fold": 1,
        "epochs": 70,
        "rng_mode": "per_item",
        "metric": METRIC,
        "checkpoint_sha256": sha256(checkpoint),
        "oof_probabilities_sha256": sha256(archive),
        "model390_lock_sha256": lock_hash,
        "inference": INFERENCE,
        "training_args": args,
    }
    (oof_dir / "model390_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    assert verify_candidate_manifest(
        root, run_dir, oof_dir, 1, 1, lock_path, lock_hash
    ) == archive

    os.utime(checkpoint, (946684799, 946684799))
    try:
        verify_candidate_manifest(root, run_dir, oof_dir, 1, 1, lock_path, lock_hash)
    except ValueError as error:
        assert "predates its pre-score lock" in str(error)
    else:
        raise AssertionError("candidate artifact predating the lock was accepted")


def main() -> None:
    draws = 500
    rows = make_rows({1: 0.01, 2: 0.005})
    first = summarize_seed(rows[1], draws)
    assert first["gate_passed"]
    assert np.isclose(first["delta_pp"], 1.0)
    assert np.allclose(first["group_bootstrap_95ci_delta_pp"], [1.0, 1.0])

    final = summarize_final(rows, draws)
    assert final["gate_passed"]
    assert np.isclose(final["equal_weight_delta_pp"], 0.75)
    assert np.allclose(final["equal_weight_group_bootstrap_95ci_delta_pp"], [0.75, 0.75])

    failing = summarize_final(make_rows({1: 0.01, 2: -0.001}), draws)
    assert not failing["gate_passed"]

    test_empty_union_zero()
    with tempfile.TemporaryDirectory(prefix="model390-oof-smoke-") as temp_dir:
        test_candidate_manifest_verification(Path(temp_dir))

    print(f"PASS: image-wise empty-union=0 and class-wise zero fill; bootstrap seed={BOOTSTRAP_SEED}.")
    print("PASS: seed1 continuation gate, two-seed paired gate, deterministic group bootstrap, and negative control.")
    print("PASS: candidate training args, checkpoint/OOF hashes, inference config, and lock-bound manifest validation.")
    print("Scope: synthetic in-memory rows and temporary dummy artifacts only; no KoMaP data or results read.")


if __name__ == "__main__":
    main()
