#!/usr/bin/env python3
"""Synthetic-only tests for Model392 one-shot valid evaluator guardrails."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

import model392_prepare_full_train as full_train_preparer
from model392_official_valid_once import (
    METRIC,
    full_train_args,
    image_mask_stems,
    official_image_miou,
    claim_one_shot_attempt,
    verify_full_train_checkpoint,
    verify_oof_gate,
)


def test_empty_union_policy() -> None:
    truth = np.zeros((2, 2), dtype=np.uint8)
    probabilities = np.zeros((4, 2, 2), dtype=np.float32)
    probabilities[0] = 1.0
    score, classes = official_image_miou(probabilities, truth)
    assert score == 0.25
    assert classes == [1.0, 0.0, 0.0, 0.0]

    probabilities[1, 0, 0] = 1.0
    probabilities[0, 0, 0] = 0.0
    score, classes = official_image_miou(probabilities, truth)
    assert score == 0.1875
    assert classes == [0.75, 0.0, 0.0, 0.0]


def test_gate_guard() -> None:
    good = {
        "model": 392,
        "stage": "two_seed_confirmatory",
        "metric": METRIC,
        "model392_lock_sha256": "lock-hash",
        "valid_evaluation_authorized": True,
        "summary": {"gate_passed": True},
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gate.json"
        path.write_text(json.dumps(good), encoding="utf-8")
        assert verify_oof_gate(path, "lock-hash")["model"] == 392
        for update in (
            {"valid_evaluation_authorized": False},
            {"model392_lock_sha256": "different"},
            {"summary": {"gate_passed": False}},
            {"stage": "seed1_interim"},
        ):
            bad = dict(good)
            bad.update(update)
            path.write_text(json.dumps(bad), encoding="utf-8")
            try:
                verify_oof_gate(path, "lock-hash")
            except ValueError:
                pass
            else:
                raise AssertionError(f"invalid OOF gate was accepted: {update}")


def test_one_shot_claim_is_exclusive() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        claim = Path(tmp) / "valid_once.claim"
        record = {"model": 392, "stage": "official_valid_once_claimed"}
        digest = claim_one_shot_attempt(claim, record)
        assert len(digest) == 64 and claim.is_file()
        try:
            claim_one_shot_attempt(claim, record)
        except FileExistsError:
            pass
        else:
            raise AssertionError("second official-valid attempt was not blocked")


def test_valid_ids_and_config() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "images").mkdir()
        (root / "masks").mkdir()
        (root / "images/A1_step1_image.png").touch()
        (root / "masks/A1_step1_mask.png").touch()
        assert image_mask_stems(root)[0] == ["A1_step1"]
        (root / "masks/A2_step1_mask.png").touch()
        try:
            image_mask_stems(root)
        except ValueError:
            pass
        else:
            raise AssertionError("an orphan mask was accepted")

    args = full_train_args(Path("/project"), 2, 1278)
    assert args["seed"] == 2 and args["target_width"] == 1278
    assert args["val_every"] == 0 and args["normalize_width"] is True
    assert args["data_root"] == "/project/data_model392_full"


def test_full_train_preparation_receipt() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "project"
        train = root / "data/train"
        images = train / "images"
        masks = train / "masks"
        images.mkdir(parents=True)
        masks.mkdir(parents=True)
        for index in range(70):
            stem = f"A{index + 1}_step1_sample1"
            Image.fromarray(np.zeros((8, 16, 3), dtype=np.uint8)).save(
                images / f"{stem}_image.png"
            )
            (masks / f"{stem}_mask.png").touch()
        lock_path = root / "model392_lock.json"
        lock_path.write_text("synthetic lock", encoding="utf-8")
        lock_hash = full_train_preparer.sha256(lock_path)
        gate_path = root / "experiments/results/model392_widthnorm_two_seed_confirmatory.json"
        gate_path.parent.mkdir(parents=True)
        gate_path.write_text(json.dumps({
            "model": 392,
            "stage": "two_seed_confirmatory",
            "metric": METRIC,
            "model392_lock_sha256": lock_hash,
            "valid_evaluation_authorized": True,
            "summary": {"gate_passed": True},
        }), encoding="utf-8")
        original_verifier = full_train_preparer.verify_model392_lock
        full_train_preparer.verify_model392_lock = lambda _path: {}
        try:
            args = type("Args", (), {
                "project_root": root,
                "model392_lock": lock_path,
                "seed": 1,
            })()
            full_train_preparer.prepare(args)
            receipt_path = root / "outputs/model392_widthnorm/full_train/seed1/model392_full_train_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            assert receipt["stage"] == "full_train"
            assert receipt["attempts"][0]["resume_from_epoch"] == 0
            assert (root / "data_model392_full/train").resolve() == train.resolve()
            full_train_preparer.prepare(args)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            assert len(receipt["attempts"]) == 2
            run_dir = receipt_path.parent
            expected = full_train_args(root.resolve(), 1, 16)
            (run_dir / "args.json").write_text(json.dumps(expected), encoding="utf-8")
            torch_checkpoint = {
                "model_config": {
                    "architecture": "UnetPlusPlus",
                    "encoder_name": "tu-hrnet_w48",
                    "boundary_head": True,
                },
                "preprocessing_config": {
                    "patch_size": 512,
                    "input_scale": 2.0,
                    "target_width": 16,
                    "normalize_width": True,
                },
            }
            import torch
            torch.save(torch_checkpoint, run_dir / "last.pt")
            torch.save({"epoch": 70, "args": expected}, run_dir / "train_state.pt")
            (run_dir / "train.log").write_text("done. final model: last.pt\n", encoding="utf-8")
            checkpoint, checked_args = verify_full_train_checkpoint(
                root, 1, 16, lock_hash, full_train_preparer.sha256(gate_path), gate_path.stat().st_mtime
            )
            assert checkpoint.is_file() and checked_args == expected
            completed_attempts = len(receipt["attempts"])
            (run_dir / "train.log").write_text("interrupted after final checkpoint\n", encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                full_train_preparer.prepare(args)
            status = json.loads(output.getvalue())
            assert status["needs_finalize"] is True
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            assert len(receipt["attempts"]) == completed_attempts

            (run_dir / "train.log").write_text("done. final model: last.pt\n", encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                full_train_preparer.prepare(args)
            status = json.loads(output.getvalue())
            assert status["already_complete"] is True
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            assert len(receipt["attempts"]) == completed_attempts
        finally:
            full_train_preparer.verify_model392_lock = original_verifier


def main() -> None:
    test_empty_union_policy()
    test_gate_guard()
    test_one_shot_claim_is_exclusive()
    test_valid_ids_and_config()
    test_full_train_preparation_receipt()
    print("Model392 official-valid one-shot synthetic smoke: PASS")


if __name__ == "__main__":
    main()
