#!/usr/bin/env python3
"""Synthetic-only tests for Model392 paired grouped-OOF analysis."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone
import hashlib

import numpy as np
from PIL import Image
import torch

import model392_record_fold as recorder
import model392_prepare_run as prepare_runner
from model382_oof_ensemble_compare import PALETTE, class_ious
from model392_widthnorm_oof import (
    BASE_INFERENCE,
    METRIC,
    claim_confirmatory_analysis,
    expected_training_args,
    paired_group_bootstrap,
    sha256,
    summarize_final_paired,
    summarize_seed,
    verify_candidate_manifest,
)


def make_rows(delta_by_seed: dict[int, float]):
    rows = {1: {}, 2: {}}
    base = np.asarray([0.60, 0.50, 0.70, 0.30], dtype=np.float64)
    image_index = 0
    for group_index in range(1, 60):
        fold = (group_index - 1) % 4 + 1
        image_count = 2 if group_index <= 11 else 1
        for local_index in range(image_count):
            image_index += 1
            step = 4 if group_index % 4 == 0 else 2
            name = f"A{group_index}_step{step}_image{local_index}"
            for run_seed in (1, 2):
                reference = base.copy()
                candidate = np.clip(base + delta_by_seed[run_seed], 0.0, 1.0)
                rows[run_seed][name] = {
                    "group": f"A{group_index}",
                    "fold": fold,
                    "step": step,
                    "native_width": 1278 if group_index % 3 == 0 else 675,
                    "target_width": 1278,
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


def test_pooled_fold_safety_is_required() -> None:
    rows = make_rows({1: 0.0, 2: 0.0})
    deltas_pp = {
        1: {1: 2.0, 2: 0.1, 3: 2.0, 4: -0.2},
        2: {1: 2.0, 2: -0.2, 3: 2.0, 4: 0.1},
    }
    for seed in (1, 2):
        for row in rows[seed].values():
            delta = deltas_pp[seed][row["fold"]] / 100.0
            row["candidate_class_iou"] = np.clip(
                row["reference_class_iou"] + delta, 0.0, 1.0
            )
    summary = summarize_final_paired(rows, draws=1_000)
    for seed in (1, 2):
        assert sum(value > 0.0 for value in summary["per_seed"][seed]["fold_delta_pp"].values()) == 3
        assert summary["per_seed"][seed]["delta_pp"] > 0.0
    assert sum(value > 0.0 for value in summary["equal_weight_fold_delta_pp"].values()) == 2
    assert not summary["gate"]["pooled_three_of_four_folds_improve"]
    assert not summary["gate_passed"]


def test_confirmatory_analysis_single_use_claim() -> None:
    original = __import__("model392_widthnorm_oof").CONFIRMATORY_OUTPUT
    with tempfile.TemporaryDirectory() as tmp:
        fixed_output = Path(tmp) / "fixed_result.json"
        module = __import__("model392_widthnorm_oof")
        module.CONFIRMATORY_OUTPUT = fixed_output
        try:
            claim, digest = claim_confirmatory_analysis(fixed_output, "locked-source-hash")
            assert claim.is_file() and len(digest) == 64
            try:
                claim_confirmatory_analysis(fixed_output, "locked-source-hash")
            except FileExistsError:
                pass
            else:
                raise AssertionError("second confirmatory OOF analysis was not blocked")
            try:
                claim_confirmatory_analysis(Path(tmp) / "alternate.json", "locked-source-hash")
            except ValueError:
                pass
            else:
                raise AssertionError("alternate OOF result path was accepted")

            preexisting = Path(tmp) / "preexisting_result.json"
            preexisting.write_text("{}", encoding="utf-8")
            module.CONFIRMATORY_OUTPUT = preexisting
            try:
                claim_confirmatory_analysis(preexisting, "locked-source-hash")
            except FileExistsError:
                pass
            else:
                raise AssertionError("preexisting OOF result was accepted")

            stale_temp = Path(tmp) / "stale_temp_result.json"
            stale_temp.with_suffix(".json.tmp").touch()
            module.CONFIRMATORY_OUTPUT = stale_temp
            try:
                claim_confirmatory_analysis(stale_temp, "locked-source-hash")
            except FileExistsError:
                pass
            else:
                raise AssertionError("stale OOF result temporary was accepted")
        finally:
            module.CONFIRMATORY_OUTPUT = original


def test_candidate_manifest_verification(tmp: Path) -> None:
    root = tmp / "project"
    data_root = root / "data_model382_seed1_fold1"
    images_dir = data_root / "train" / "images"
    images_dir.mkdir(parents=True)
    for index, width in enumerate((40, 80, 80), start=1):
        Image.fromarray(np.zeros((24, width, 3), dtype=np.uint8)).save(
            images_dir / f"A{index}_step1_sample1_image.png"
        )

    run_dir = root / "outputs/model392/seed1/fold1"
    oof_dir = root / "outputs/model392/seed1/fold1_oof"
    run_dir.mkdir(parents=True)
    oof_dir.mkdir(parents=True)
    lock_path = root / "model392_lock.json"
    lock_path.write_text(
        json.dumps({"locked_at_utc": "2000-01-01T00:00:00Z"}), encoding="utf-8"
    )
    lock_hash = sha256(lock_path)
    checkpoint = run_dir / "last.pt"
    checkpoint.write_bytes(b"synthetic checkpoint bytes")
    archive = oof_dir / "valid_probabilities.npz"
    np.savez_compressed(archive, A1_step1_sample1=np.full((4, 2, 3), 0.25, np.float16))
    args = expected_training_args(root, run_dir, 1, 1)
    (run_dir / "args.json").write_text(json.dumps(args), encoding="utf-8")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    args_hash = hashlib.sha256(
        json.dumps(args, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    run_receipt = {
        "model": 392,
        "seed": 1,
        "fold": 1,
        "model392_lock_sha256": lock_hash,
        "initialized_at_utc": now,
        "training_args_sha256": args_hash,
        "attempts": [{
            "started_at_utc": now,
            "resume_from_epoch": 0,
            "previous_checkpoint_sha256": None,
        }],
    }
    run_receipt_path = run_dir / "model392_run_receipt.json"
    run_receipt_path.write_text(json.dumps(run_receipt), encoding="utf-8")
    inference = {
        **BASE_INFERENCE,
        "target_width": 80,
        "scale_rule": "base_input_scale * target_width / native image width",
        "normalize_width": True,
    }
    manifest = {
        "model": 392,
        "seed": 1,
        "fold": 1,
        "epochs": 70,
        "rng_mode": "legacy",
        "metric": METRIC,
        "target_width": 80,
        "checkpoint_sha256": sha256(checkpoint),
        "oof_probabilities_sha256": sha256(archive),
        "model392_lock_sha256": lock_hash,
        "run_receipt_sha256": sha256(run_receipt_path),
        "resume_attempts": run_receipt["attempts"],
        "inference": inference,
        "training_args": args,
    }
    (oof_dir / "model392_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    assert verify_candidate_manifest(
        root, run_dir, oof_dir, 1, 1, lock_path, lock_hash
    ) == archive

    wrong_manifest = dict(manifest, target_width=81)
    (oof_dir / "model392_manifest.json").write_text(
        json.dumps(wrong_manifest), encoding="utf-8"
    )
    try:
        verify_candidate_manifest(root, run_dir, oof_dir, 1, 1, lock_path, lock_hash)
    except ValueError as error:
        assert "manifest mismatch" in str(error)
    else:
        raise AssertionError("manifest with incorrect target width was accepted")

    os.utime(checkpoint, (946684799, 946684799))
    try:
        verify_candidate_manifest(root, run_dir, oof_dir, 1, 1, lock_path, lock_hash)
    except ValueError as error:
        assert "predates its pre-score lock" in str(error)
    else:
        raise AssertionError("candidate artifact predating the lock was accepted")


def test_fold_recorder(tmp: Path) -> None:
    tmp.mkdir(parents=True, exist_ok=True)
    root = (tmp / "synthetic_project").resolve()
    lock = tmp / "synthetic_lock.json"
    lock.write_text("{}", encoding="utf-8")
    cv_root = root / "data_model322_group_cv"
    train_fold = cv_root / "train_fold1"
    heldout_fold = cv_root / "oof_fold1"
    train_images = train_fold / "images"
    masks_dir = heldout_fold / "masks"
    train_images.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    for index, width in enumerate((40, 80, 80), start=1):
        Image.fromarray(np.zeros((24, width, 3), dtype=np.uint8)).save(
            train_images / f"A{index}_step1_sample1_image.png"
        )

    expected_ids = ("A91_step1_sample1", "A92_step2_sample1")
    for image_id in expected_ids:
        rgb = np.broadcast_to(PALETTE[0], (2, 3, 3)).copy()
        Image.fromarray(rgb).save(masks_dir / f"{image_id}_mask.png")

    data_root = root / "data_model382_seed1_fold1"
    data_root.mkdir()
    os.symlink(train_fold, data_root / "train")
    os.symlink(heldout_fold, data_root / "valid")
    run_dir = root / "outputs/model392_widthnorm/seed1/fold1"
    oof_dir = root / "outputs/model392_widthnorm/seed1/fold1_oof"
    run_dir.mkdir(parents=True)
    oof_dir.mkdir(parents=True)
    args = expected_training_args(root, run_dir, 1, 1)
    (run_dir / "args.json").write_text(json.dumps(args), encoding="utf-8")
    args_digest = hashlib.sha256(
        json.dumps(args, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    (run_dir / "model392_run_receipt.json").write_text(json.dumps({
        "model": 392,
        "seed": 1,
        "fold": 1,
        "model392_lock_sha256": sha256(lock),
        "initialized_at_utc": now,
        "training_args_sha256": args_digest,
        "attempts": [{
            "started_at_utc": now,
            "resume_from_epoch": 0,
            "previous_checkpoint_sha256": None,
        }],
    }), encoding="utf-8")
    torch.save({"epoch": 70}, run_dir / "train_state.pt")
    torch.save({
        "preprocessing_config": {
            "patch_size": 512,
            "input_scale": 2.0,
            "target_width": 80,
            "normalize_width": True,
        }
    }, run_dir / "last.pt")
    (run_dir / "train.log").write_text("done. final model: synthetic\n", encoding="utf-8")
    probabilities = np.full((4, 2, 3), 0.25, dtype=np.float16)
    np.savez_compressed(oof_dir / "valid_probabilities.npz", **{
        image_id: probabilities for image_id in expected_ids
    })
    (oof_dir / "valid_summary.json").write_text(json.dumps({
        "split": "valid",
        "images": len(expected_ids),
        "image_miou_empty_zero": 0.25,
        "inference": {
            "patch_size": 512,
            "base_input_scale": 2.0,
            "target_width": 80,
            "overlap": 170,
            "overlap_rule": "preserve overlap/native-tile ratio",
            "tta": "4-way horizontal/vertical flips",
            "tta_mode": "flip4",
            "tta_enabled": True,
            "downsample_antialias": False,
            "batch_size": 2,
            "device": "cuda:0",
        },
    }), encoding="utf-8")
    old_verifier = recorder.verify_model392_lock
    recorder.verify_model392_lock = lambda _: {"locked_at_utc": "2000-01-01T00:00:00Z"}
    try:
        result = recorder.record_fold(SimpleNamespace(
            project_root=root,
            lock=lock,
            seed=1,
            fold=1,
        ))
    finally:
        recorder.verify_model392_lock = old_verifier
    assert result["target_width"] == 80
    manifest = json.loads((oof_dir / "model392_manifest.json").read_text())
    assert manifest["training_args"] == args
    assert manifest["oof_probabilities_sha256"] == sha256(oof_dir / "valid_probabilities.npz")


def test_run_receipt_resume_history(tmp: Path) -> None:
    tmp.mkdir(parents=True, exist_ok=True)
    root = (tmp / "project").resolve()
    cv_root = root / "data_model322_group_cv"
    train = cv_root / "train_fold1"
    heldout = cv_root / "oof_fold1"
    train_images = train / "images"
    heldout_images = heldout / "images"
    train_images.mkdir(parents=True)
    heldout_images.mkdir(parents=True)
    for index, width in enumerate((40, 80, 80), start=1):
        Image.fromarray(np.zeros((16, width, 3), dtype=np.uint8)).save(
            train_images / f"A{index}_step1_sample1_image.png"
        )
    Image.fromarray(np.zeros((16, 80, 3), dtype=np.uint8)).save(
        heldout_images / "A91_step1_sample1_image.png"
    )
    data_root = root / "data_model382_seed1_fold1"
    data_root.mkdir()
    os.symlink(train, data_root / "train")
    os.symlink(heldout, data_root / "valid")
    lock = tmp / "synthetic_prepare_lock.json"
    lock.write_text("{}", encoding="utf-8")

    old_verifier = prepare_runner.verify_model392_lock
    prepare_runner.verify_model392_lock = lambda _: {"locked_at_utc": "2000-01-01T00:00:00Z"}
    try:
        first = prepare_runner.prepare(SimpleNamespace(
            project_root=root, lock=lock, seed=1, fold=1
        ))
        run_dir = root / "outputs/model392_widthnorm/seed1/fold1"
        args = expected_training_args(root, run_dir, 1, 1)
        (run_dir / "args.json").write_text(json.dumps(args), encoding="utf-8")
        second = prepare_runner.prepare(SimpleNamespace(
            project_root=root, lock=lock, seed=1, fold=1
        ))
        assert second["attempts"][0]["resume_from_epoch"] == 0
        assert second["attempts"][1]["resume_from_epoch"] == 0
        torch.save({"epoch": 3}, run_dir / "train_state.pt")
        (run_dir / "last.pt").write_bytes(b"synthetic epoch-3 checkpoint")
        third = prepare_runner.prepare(SimpleNamespace(
            project_root=root, lock=lock, seed=1, fold=1
        ))
    finally:
        prepare_runner.verify_model392_lock = old_verifier
    assert first["attempts"][0]["resume_from_epoch"] == 0
    assert third["attempts"][-1]["resume_from_epoch"] == 3
    assert third["attempts"][-1]["previous_checkpoint_sha256"] == sha256(run_dir / "last.pt")


def test_seed_gates_and_paired_bootstrap() -> None:
    rows = make_rows({1: 0.01, 2: 0.005})
    first = summarize_seed(rows[1], draws=500)
    assert first["gate_passed"]
    assert np.isclose(first["delta_pp"], 1.0)
    assert np.allclose(first["group_bootstrap_95ci_delta_pp"], [1.0, 1.0])

    final = summarize_final_paired(rows, draws=500)
    assert final["gate_passed"]
    assert np.isclose(final["equal_weight_delta_pp"], 0.75)
    assert np.allclose(final["equal_weight_group_bootstrap_95ci_delta_pp"], [0.75, 0.75])
    assert "resample each fold's specimen groups once" in final["bootstrap_pairing"]
    assert final["width_strata_descriptive_only"]["native_width_equals_fold_train_mode"]["images"] > 0
    assert paired_group_bootstrap(rows, draws=50) == paired_group_bootstrap(rows, draws=50)

    failing = summarize_final_paired(make_rows({1: 0.01, 2: -0.001}), draws=500)
    assert not failing["gate_passed"]

    heterogeneous = make_rows({1: 0.0, 2: 0.0})
    for name in heterogeneous[1]:
        group_index = int(heterogeneous[1][name]["group"][1:])
        deltas = {1: 0.02 if group_index % 2 == 0 else -0.015,
                  2: 0.01 if group_index % 2 == 0 else -0.020}
        for run_seed in (1, 2):
            row = heterogeneous[run_seed][name]
            row["candidate_class_iou"] = np.clip(
                row["reference_class_iou"] + deltas[run_seed], 0.0, 1.0
            )
    heterogeneous_ci = paired_group_bootstrap(heterogeneous, draws=500)
    assert heterogeneous_ci[0] < heterogeneous_ci[1]
    assert heterogeneous_ci == paired_group_bootstrap(heterogeneous, draws=500)


def main() -> None:
    test_empty_union_zero()
    test_pooled_fold_safety_is_required()
    test_confirmatory_analysis_single_use_claim()
    test_seed_gates_and_paired_bootstrap()
    with tempfile.TemporaryDirectory(prefix="model392-oof-smoke-") as temp_dir:
        temp = Path(temp_dir)
        test_candidate_manifest_verification(temp / "manifest")
        test_fold_recorder(temp / "recorder")
        test_run_receipt_resume_history(temp / "resume")
    print("PASS: image-wise four-class IoU uses empty-union=0.")
    print("PASS: per-seed safety, equal-weight two-seed gate, heterogeneous paired group bootstrap, and width diagnostics.")
    print("PASS: train-only modal width, checkpoint/OOF hashes, lock timestamp checks, fold manifests, and resume receipts.")
    print("Scope: synthetic rows, temporary images, and dummy artifacts only; no KoMaP scores or GPU used.")


if __name__ == "__main__":
    main()
