#!/usr/bin/env python3
"""Synthetic end-to-end smoke test for the fixed Model382 OOF evaluator."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from model374_train_oof_error_diagnostic import PALETTE
from model382_oof_ensemble_compare import LOCKED_ARTIFACTS, class_ious, run, sha256


def probs_for(truth: np.ndarray, group: int, model_code: int) -> np.ndarray:
    prediction = truth.copy()
    y = (group + model_code) % truth.shape[0]
    x = (group * 3 + model_code) % truth.shape[1]
    prediction[y, x] = (int(prediction[y, x]) + 1) % 4
    result = np.full((4, *truth.shape), 0.001, dtype=np.float32)
    for class_id in range(4):
        result[class_id, prediction == class_id] = 0.997
    return result


def main() -> None:
    absent = class_ious(
        np.concatenate((np.ones((1, 4, 4), dtype=np.float32),
                        np.zeros((3, 4, 4), dtype=np.float32)), axis=0),
        np.zeros((4, 4), dtype=np.uint8),
    )
    assert np.allclose(absent, (1.0, 0.0, 0.0, 0.0))
    with tempfile.TemporaryDirectory(prefix="model382-oof-smoke-") as temporary:
        root = Path(temporary)
        cv, m374, m376, m377, m382 = (root / name for name in ("cv", "m374", "m376", "m377", "m382"))
        group_ids = list(range(1, 59)) + [71]
        fold_group_counts = (15, 15, 14, 15)
        duplicate_counts = (3, 3, 3, 2)
        cursor = 0
        for fold, (group_count, duplicate_count) in enumerate(zip(fold_group_counts, duplicate_counts), start=1):
            masks = cv / f"oof_fold{fold}" / "masks"
            masks.mkdir(parents=True)
            archives: dict[str, dict[str, np.ndarray]] = {
                "m374": {}, "m376": {}, "m377": {}, "s1": {}, "s2": {}
            }
            fold_groups = group_ids[cursor:cursor + group_count]
            cursor += group_count
            for group_position, group in enumerate(fold_groups):
                image_count = 2 if group_position < duplicate_count else 1
                for sample in range(1, image_count + 1):
                    step = 4 if group == 71 else sample
                    image_id = f"A{group}_step{step}_sample{sample}"
                    truth = (np.indices((8, 8)).sum(axis=0) % 4).astype(np.uint8)
                    Image.fromarray(PALETTE[truth]).save(masks / f"{image_id}_mask.png")
                    for code, key in enumerate(archives, start=1):
                        archives[key][image_id] = probs_for(truth, group, code)
            for model_root, key in ((m374, "m374"), (m376, "m376"), (m377, "m377")):
                out = model_root / f"fold{fold}_oof"
                out.mkdir(parents=True)
                np.savez_compressed(out / "valid_probabilities.npz", **archives[key])
            for seed, key in ((1, "s1"), (2, "s2")):
                model382_oof = m382 / f"seed{seed}/fold{fold}_oof"
                model382_oof.mkdir(parents=True)
                archive_path = model382_oof / "valid_probabilities.npz"
                np.savez_compressed(archive_path, **archives[key])
                checkpoint = m382 / f"seed{seed}/fold{fold}/last.pt"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(f"synthetic seed={seed}, fold={fold}".encode())
                manifest = {
                    "model": 382, "seed": seed, "fold": fold, "epochs": 70,
                    "metric": "image-wise 4-class mIoU; empty-union=0",
                    "checkpoint_sha256": sha256(checkpoint),
                    "oof_probabilities_sha256": sha256(archive_path),
                    "training_args": {
                        "seed": seed, "epochs": 70, "architecture": "UnetPlusPlus",
                        "encoder_name": "tu-hrnet_w48", "encoder_weights": "imagenet",
                        "patch_size": 512, "input_scale": 2.0, "samples_per_image": 12,
                        "batch_size": 4, "accumulation_steps": 2, "lr": 3e-4,
                        "encoder_lr_mult": 1.0, "weight_decay": 1e-4,
                        "boundary_head_weight": 1.0, "val_every": 0,
                        "init_checkpoint": None,
                    },
                }
                (model382_oof / "model382_manifest.json").write_text(json.dumps(manifest))

        assert cursor == len(group_ids)
        lock_path = root / "pre_result_lock.json"
        repo_root = Path(__file__).resolve().parents[1]
        lock = {
            "locked_at_utc": "2000-01-01T00:00:00Z",
            "locked_at_unix": 1,
            "artifacts_sha256": {
                relative: sha256(repo_root / relative) for relative in LOCKED_ARTIFACTS
            },
        }
        lock_path.write_text(json.dumps(lock))
        result = run(cv, m374, m376, m377, m382, lock_path, draws=200)
        assert result["model_scores"]["A"]["images"] == 70
        assert result["model_scores"]["A"]["specimen_groups"] == 59
        assert result["comparisons"]["C_vs_B"]["paired_fold_stratified_group_bootstrap_95ci_delta_pp"]
        assert np.isfinite(result["comparisons"]["E5_vs_C"]["A71_excluded_delta_pp"])
        assert result["E5_secondary_gate_passed"] in (True, False)

        future_lock_path = root / "future_lock.json"
        lock["locked_at_unix"] = 4_102_444_800.0
        future_lock_path.write_text(json.dumps(lock))
        try:
            run(cv, m374, m376, m377, m382, future_lock_path, draws=200)
        except ValueError as error:
            assert "predates or matches preregistration lock" in str(error)
        else:
            raise AssertionError("OOF archives older than lock must fail closed")
        print({
            "images": 70,
            "groups": 59,
            "arms": sorted(result["arms"]),
            "comparison_count": len(result["comparisons"]),
            "empty_union_policy": "0 verified",
            "A71_excluded_metric_present": True,
            "manifest_checks": "passed",
            "pre_result_timestamp_gate": "pass; stale OOF correctly rejected",
        })


if __name__ == "__main__":
    main()
