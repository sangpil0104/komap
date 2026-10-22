import csv
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import core
import reporting
import run


class ThresholdTests(unittest.TestCase):
    def test_dp_matches_exhaustive_optimum(self):
        rng = np.random.default_rng(923)
        for _ in range(10):
            levels = np.sort(rng.choice(256, 8, replace=False))
            counts = rng.integers(1, 30, size=8)
            gray = np.repeat(levels, counts).astype(np.uint8)[None, :]

            def objective(cuts):
                groups = core.intensity_bins(gray, cuts)
                return sum(float(gray[groups == c].sum()) ** 2 / np.count_nonzero(groups == c)
                           for c in range(4))

            exhaustive = max(objective([(float(levels[i - 1]) + float(levels[i])) / 2 for i in indices])
                             for indices in itertools.combinations(range(1, len(levels)), 3))
            actual = objective(core.multi_otsu(gray))
            self.assertAlmostEqual(actual, exhaustive, places=6)

    def test_affine_brightness_adaptation(self):
        first = np.array([[5, 10, 35, 40, 65, 70, 95, 100]], dtype=np.uint8)
        second = first * 2 + 20
        cuts_first = np.asarray(core.multi_otsu(first))
        cuts_second = np.asarray(core.multi_otsu(second))
        np.testing.assert_allclose(cuts_second, cuts_first * 2 + 20)
        np.testing.assert_array_equal(core.intensity_bins(first, cuts_first),
                                      core.intensity_bins(second, cuts_second))

    def test_interval_and_degenerate_image(self):
        gray = np.array([[30, 30, 30]], dtype=np.uint8)
        self.assertIsNone(core.multi_otsu(gray))
        model = {"fallback_thresholds": [30, 100, 180], "bin_to_class": [2, 1, 0, 3]}
        pred, info = core.predict(np.repeat(gray[..., None], 3, axis=2), model)
        self.assertTrue(np.all(pred == 2))
        self.assertEqual(info["threshold_source"], "train_fallback")
        self.assertEqual(model["fallback_thresholds"], [30, 100, 180])
        np.testing.assert_array_equal(core.intensity_bins(np.array([[30, 31, 100, 101, 180, 181]]),
                                                         [30, 100, 180]), [[0, 1, 1, 2, 2, 3]])

    def test_phase_mapping_permutation(self):
        mapping = [2, 0, 3, 1]
        cm = np.zeros((4, 4), dtype=int)
        for bin_id, class_id in enumerate(mapping):
            cm[class_id, bin_id] = 10 + bin_id
        best, candidates = core.choose_mapping([cm, cm * 7])
        self.assertEqual(len(candidates), 24)
        self.assertEqual(best["bin_to_class"], mapping)
        self.assertEqual(best["train_image_miou_one"], 100)

    def test_fit_rejects_validation_before_any_label_access(self):
        with patch.object(core, "read_target") as target_reader:
            with self.assertRaisesRegex(ValueError, "restricted"):
                core.fit([{"source_split": "valid"}])
            target_reader.assert_not_called()

    def test_empty_classes_and_image_mean(self):
        first = np.zeros((4, 4), dtype=int)
        first[0, 0] = 1000
        second = np.zeros((4, 4), dtype=int)
        second[0, 1] = 1
        result = core.summarize([core.image_metrics(first), core.image_metrics(second)])
        self.assertEqual(result["image_miou_one"], 75)
        self.assertEqual(result["image_miou_zero"], 12.5)
        self.assertEqual(result["image_miou_exclude"], 50)
        self.assertEqual(result["class_0_iou_one"], 50)
        self.assertIsNone(result["class_2_iou_exclude"])
        self.assertNotAlmostEqual(core.class_ious(first + second).mean() * 100, 75)
        self.assertEqual(core.class_ious(second)[1], 0)  # Absent in GT but falsely predicted.

    def test_error_panel_uses_gt_and_preserves_pixels(self):
        target = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        pred = np.array([[1, 1], [0, 3]], dtype=np.uint8)
        errors = reporting.error_rgb(target, pred)
        np.testing.assert_array_equal(errors[0, 0], core.PALETTE[0])
        np.testing.assert_array_equal(errors[1, 0], [204, 204, 204])
        np.testing.assert_array_equal(errors[0, 1], [255, 255, 255])
        original = np.zeros((2, 2, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            check = reporting.comparison(Path(temporary) / "panel.png", original, target, pred, "test", 50)
            self.assertEqual(check["panels_verified"], 4)

    def test_output_reservation_preserves_existing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = run.fresh_result(root)
            (first / "keep.txt").write_text("preserve", encoding="utf-8")
            second = run.fresh_result(root)
            self.assertEqual(int(second.name.rsplit("_", 1)[1]), int(first.name.rsplit("_", 1)[1]) + 1)
            self.assertEqual((first / "keep.txt").read_text(), "preserve")
            run.write_json(first / "record.json", {"keep": True})
            with self.assertRaises(FileExistsError):
                run.write_json(first / "record.json", {"keep": False})


class SyntheticWorkflowTest(unittest.TestCase):
    def test_cli_suite_and_unlabeled_prediction(self):
        # All data, fitted fixtures, and reports live in this disposable directory.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            bins = np.tile(np.repeat(np.arange(4), 4), (8, 1)).astype(np.uint8)
            mapping = np.array([2, 1, 3, 0], dtype=np.uint8)
            for index in range(100):
                split = "train" if index < 70 else "valid" if index < 90 else "test"
                gray = np.array([20, 80, 150, 220], dtype=np.uint8)[bins]
                # Give every original a distinct hash without changing its class structure.
                row_id, col_id = divmod(index, 16)
                gray[row_id, col_id] += 1
                image_path = root / f"image_{index}.png"
                mask_path = root / f"mask_{index}.png"
                Image.fromarray(gray).convert("RGB").save(image_path)
                if split != "test":
                    Image.fromarray(core.PALETTE[mapping[bins]]).save(mask_path)
                rows.append({"image_id": f"synthetic_{index}", "source_split": split,
                             "image_path": str(image_path), "mask_path": str(mask_path) if split != "test" else "",
                             "width": 16, "height": 8, "image_sha256": core.sha256(image_path),
                             "mask_sha256": core.sha256(mask_path) if split != "test" else ""})
            manifest = root / "manifest.csv"
            with open(manifest, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            base = [sys.executable, str(HERE / "run.py")]
            process = subprocess.run(base + ["preflight", "--manifest", str(manifest),
                                     "--output", str(root / "preflight.json")],
                                     text=True, capture_output=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertEqual(json.loads((root / "preflight.json").read_text())["checked_masks"], 90)
            process = subprocess.run(base + ["suite", "--manifest", str(manifest),
                                     "--runs-root", str(root / "runs"),
                                     "--results-root", str(root / "results")],
                                     text=True, capture_output=True, timeout=90)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertIn("total_estimate~", process.stdout)
            self.assertIn("DONE | total_elapsed=", process.stdout)
            model_path = next((root / "runs").glob("*/model.json"))
            model = core.load_model(model_path)
            self.assertEqual(model["bin_to_class"], mapping.tolist())
            self.assertEqual(model["train_image_count"], 70)
            self.assertEqual(len(model["train_images"]), 70)
            self.assertEqual({r["image_id"] for r in model["train_images"]}, {f"synthetic_{i}" for i in range(70)})
            result = next((root / "results").iterdir())
            summary = json.loads((result / "summary.json").read_text())
            self.assertEqual(summary["image_count"], 20)
            self.assertEqual(summary["image_miou_one"], 100)
            self.assertEqual(len(list((result / "valid_mask_comparison").glob("*.png"))), 20)
            self.assertTrue((result / f"{result.name}.md").is_file())
            self.assertTrue((result / "sources_at_creation" / "core.py").is_file())
            # A new image has no accompanying mask. Predict must not request one.
            original_only = root / "original_only.png"
            Image.new("RGB", (5, 7), (100, 100, 100)).save(original_only)
            process = subprocess.run(base + ["predict", "--model", str(model_path),
                                     "--input", str(original_only), "--output-root", str(root / "inference")],
                                     text=True, capture_output=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            prediction_record = next((root / "inference").glob("*/predictions.json"))
            self.assertEqual(json.loads(prediction_record.read_text())[0]["threshold_source"], "train_fallback")


if __name__ == "__main__":
    unittest.main()
