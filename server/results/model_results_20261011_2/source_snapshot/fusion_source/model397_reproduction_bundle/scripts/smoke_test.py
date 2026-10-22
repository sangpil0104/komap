#!/usr/bin/env python3
"""CPU-only synthetic checks for the Model397 OOF blend scorer."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_oof_splits import ROOT as BUNDLE_ROOT
from verify_training_args import EXPECTED
from write_oof_manifest import create_or_verify, verify_manifest
from score_model397_oof import (
    CLASS_NAMES,
    FOLD_FILE,
    PALETTE,
    bootstrap_indices,
    class_ious,
    compare,
    group_and_step,
    summarize,
)


class Model397Smoke(unittest.TestCase):
    def test_fixed_fold_inventory(self):
        config = json.loads(FOLD_FILE.read_text(encoding="utf-8"))
        heldout = [group for fold in config["folds"] for group in fold["heldout_groups"]]
        self.assertEqual(len(config["folds"]), 4)
        self.assertEqual(len(heldout), 59)
        self.assertEqual(len(set(heldout)), 59)

    def test_synthetic_split_creation_and_idempotence(self):
        config = json.loads(FOLD_FILE.read_text(encoding="utf-8"))
        extra_groups = set()
        for entry in config["folds"]:
            count = 3 if int(entry["fold"]) < 4 else 2
            extra_groups.update(entry["heldout_groups"][:count])
        with tempfile.TemporaryDirectory(prefix="model397-smoke-") as temporary:
            root = Path(temporary)
            data = root / "data"
            image_dir, mask_dir = data / "train/images", data / "train/masks"
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            all_groups = sorted(group for entry in config["folds"] for group in entry["heldout_groups"])
            for group in all_groups:
                stems = [f"{group}_step1_sample1"]
                if group in extra_groups:
                    stems.append(f"{group}_step1_sample2")
                for stem in stems:
                    Image.new("RGB", (2, 2), tuple(int(x) for x in PALETTE[0])).save(
                        image_dir / f"{stem}_image.png"
                    )
                    Image.new("RGB", (2, 2), tuple(int(x) for x in PALETTE[0])).save(
                        mask_dir / f"{stem}_mask.png"
                    )
            output = root / "splits"
            command = [
                sys.executable, str(BUNDLE_ROOT / "scripts/prepare_oof_splits.py"),
                "--data-root", str(data), "--output-root", str(output),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            subprocess.run(command, check=True, capture_output=True, text=True)
            for fold, expected in ((1, 18), (2, 18), (3, 17), (4, 17)):
                files = list((output / "model329_cv" / f"oof_fold{fold}/images").glob("*_image.png"))
                self.assertEqual(len(files), expected)
                self.assertTrue(all(path.is_symlink() for path in files))

    def test_saved_recipe_verifier(self):
        with tempfile.TemporaryDirectory(prefix="model397-args-smoke-") as temporary:
            root = Path(temporary)
            b4_run = root / "model329/fold1"
            b4_run.mkdir(parents=True)
            b4_args = dict(EXPECTED["model329"])
            b4_args.update({
                "train_splits": "train_fold1",
                "data": str((root / "splits/model329_cv").resolve()),
                "out": str(b4_run.resolve()),
            })
            b4_path = b4_run / "args.json"
            b4_path.write_text(json.dumps(b4_args), encoding="utf-8")
            subprocess.run([
                sys.executable, str(BUNDLE_ROOT / "scripts/verify_training_args.py"),
                "--family", "model329", "--args-file", str(b4_path), "--fold", "1",
            ], check=True, capture_output=True, text=True)

            hr_run = root / "model392/seed1/fold1"
            hr_run.mkdir(parents=True)
            hr_args = dict(EXPECTED["model392"])
            hr_args.update({
                "seed": 1,
                "data_root": str((root / "splits/model392_cv/seed1_fold1").resolve()),
                "output_dir": str(hr_run.resolve()),
                "target_width": 1230,
            })
            hr_path = hr_run / "args.json"
            hr_path.write_text(json.dumps(hr_args), encoding="utf-8")
            subprocess.run([
                sys.executable, str(BUNDLE_ROOT / "scripts/verify_training_args.py"),
                "--family", "model392", "--args-file", str(hr_path),
                "--seed", "1", "--fold", "1",
            ], check=True, capture_output=True, text=True)

    def test_oof_manifest_binds_checkpoint_and_probabilities(self):
        config = json.loads(FOLD_FILE.read_text(encoding="utf-8"))
        groups = config["folds"][0]["heldout_groups"]
        image_ids = [f"{group}_step1_sample1" for group in groups]
        image_ids.extend(f"{group}_step1_sample2" for group in groups[:3])
        with tempfile.TemporaryDirectory(prefix="model397-manifest-smoke-") as temporary:
            output = Path(temporary) / "output"
            b4_data = output / "splits/model329_cv/oof_fold1/images"
            hr_data = output / "splits/model392_cv/seed1_fold1/valid/images"
            b4_data.mkdir(parents=True)
            hr_data.mkdir(parents=True)
            for image_id in image_ids:
                (b4_data / f"{image_id}_image.png").touch()
                (hr_data / f"{image_id}_image.png").touch()
            b4_run = output / "model329/fold1"
            hr_run = output / "model392/seed1/fold1"
            b4_run.mkdir(parents=True)
            hr_run.mkdir(parents=True)
            for run in (b4_run, hr_run):
                (run / "args.json").write_text('{"synthetic":true}', encoding="utf-8")
                (run / "last.pt").write_bytes(b"synthetic-checkpoint")
            b4_pred = output / "model329/fold1_oof"
            hr_pred = output / "model392/seed1/fold1_oof"
            b4_pred.mkdir(parents=True)
            hr_pred.mkdir(parents=True)
            probability = np.full((4, 2, 2), 0.25, dtype=np.float32)
            np.savez_compressed(b4_pred / "probabilities.npz", **{key: probability for key in image_ids})
            np.savez_compressed(hr_pred / "valid_probabilities.npz", **{key: probability for key in image_ids})
            b4_manifest = create_or_verify(output, "model329", 1, None)
            hr_manifest = create_or_verify(output, "model392", 1, 1)
            self.assertEqual(len(b4_manifest["oof_image_ids"]), 18)
            self.assertEqual(len(hr_manifest["heldout_groups"]), 15)
            verify_manifest(output, "model329", 1, None)
            (b4_run / "last.pt").write_bytes(b"changed-checkpoint")
            with self.assertRaises(ValueError):
                verify_manifest(output, "model329", 1, None)

    def test_group_step_parser(self):
        self.assertEqual(group_and_step("A118_step4_sample2"), ("A118", 4))

    def test_official_empty_union_zero(self):
        gt = np.zeros((2, 2), dtype=np.uint8)
        probs = np.zeros((4, 2, 2), dtype=np.float32)
        probs[1] = 1.0
        np.testing.assert_array_equal(class_ious(probs, gt), np.zeros(4))

    def test_perfect_four_class_iou(self):
        gt = np.asarray([[0, 1], [2, 3]], dtype=np.uint8)
        probs = np.eye(4, dtype=np.float32)[gt].transpose(2, 0, 1)
        np.testing.assert_array_equal(class_ious(probs, gt), np.ones(4))
        result = summarize(np.asarray([class_ious(probs, gt)]))
        self.assertAlmostEqual(result["image_mean_miou_percent"], 100.0)
        self.assertEqual(list(result["class_image_mean_iou_percent"]), CLASS_NAMES)

    def test_group_bootstrap_is_reproducible_and_fold_stratified(self):
        rows = [
            {"fold": fold, "group": f"A{fold}_{group}", "step": 4 if group == 0 else 1}
            for fold in range(1, 5)
            for group in range(2)
        ]
        a = bootstrap_indices(rows, draws=32, seed=397)
        b = bootstrap_indices(rows, draws=32, seed=397)
        self.assertEqual(len(a), 32)
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left, right)
            self.assertEqual(len(left), len(rows))
            for fold in range(1, 5):
                self.assertEqual(sum(rows[i]["fold"] == fold for i in left), 2)

    def test_comparison_zero_delta(self):
        rows = [
            {"fold": fold, "group": f"A{fold}_{group}", "step": 4 if group == 0 else 1}
            for fold in range(1, 5)
            for group in range(2)
        ]
        values = np.full((len(rows), 4), 0.75)
        summary = compare(values, values, rows, bootstrap_indices(rows, draws=16, seed=1))
        self.assertEqual(summary["delta_image_mean_pp"], 0.0)
        self.assertEqual(summary["positive_folds"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
