"""Integration checks; no training or optimizer updates."""
import ast
import copy
import json
import pickle
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image

import components326
import model338
from runtime import ROOT, SOURCE, PALETTE, common, metric, pairs, read_json, export_results, comparison
from run import gpu_list


class Integration(unittest.TestCase):
    def test_empty_class_is_zero_and_images_equal_weight(self):
        target = np.array([[0]*9 + [1]*9 + [3]*9], dtype=np.uint8)
        pred = target.copy()
        pred[0, 0], pred[0, 9], pred[0, 18] = 1, 3, 0
        result = metric(pred, target)
        self.assertAlmostEqual(result["image_miou"], .6)
        self.assertEqual(result["class_iou"]["Primary Si"], 0)
        self.assertAlmostEqual(metric(target, target)["image_miou"], .75)

    def test_original_training_components_preserved(self):
        original = ast.parse((SOURCE / "experiments/model326_hybrid_fullstats/KoMaP/train.py").read_text())
        adapted = ast.parse((ROOT / "components326.py").read_text())
        for name in ("CropDS", "worker_init", "dice_loss"):
            left = next(n for n in original.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
            right = next(n for n in adapted.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
            self.assertEqual(ast.dump(left), ast.dump(right))
        left = next(n for n in original.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "aug" for t in n.targets))
        configure = next(n for n in adapted.body if isinstance(n, ast.FunctionDef) and n.name == "configure")
        right = next(n for n in configure.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "aug" for t in n.targets))
        self.assertEqual(ast.dump(left), ast.dump(right))

    def test_actual_training_crop(self):
        config = read_json(ROOT / "configs/model326.json")
        components326.configure(config)
        ds = components326.CropDS(pairs(split="train")[:1], 2)
        x, target = ds[0]
        self.assertEqual(tuple(x.shape), (3, 1536, 1536))
        self.assertEqual(tuple(target.shape), (1536, 1536))
        self.assertTrue(np.isfinite(x.numpy()).all())
        self.assertTrue(np.isin(target.numpy(), range(4)).all())
        full = common.resize_gray(ds.imgs[0], 3).astype(np.float32) / 255
        self.assertAlmostEqual(ds.image_stats[0][0], float(full.mean()))

    def test_gpu_ids_and_mapping(self):
        self.assertEqual(gpu_list("0,1,2,3,4,5"), list(range(6)))
        for invalid in ("6", "0,0", "-1", ""):
            with self.assertRaises(ValueError):
                gpu_list(invalid)
        labels = np.arange(4)
        np.testing.assert_array_equal(model338.MAIN_LABEL_TO_TEAM[model338.TEAM_LABEL_TO_MAIN[labels]], labels)
        team = np.array([.1, .2, .3, .4])[:, None, None]
        np.testing.assert_array_equal(team[model338.TEAM_TO_MAIN, 0, 0], [.2, .3, .1, .4])

    def test_orchestration_and_duplicate_lock_without_training(self):
        import fcntl
        import run as entry
        from runtime import write_json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "source").mkdir()
            config = read_json(ROOT / "configs/model326.json")
            config["iters"] = 2
            write_json(root / "configs/model326.json", config)
            write_json(root / "configs/folds.json", {"completed_folds": []})
            launched = []

            class FakeProcess:
                def __init__(self, command, **kwargs):
                    self.pid = 12345678
                    self.returncode = 0
                    launched.append((command, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
                    if "_fold326" in command:
                        fold = int(command[command.index("--fold")+1])
                        output = Path(command[command.index("--run")+1]) / f"fold{fold}"
                        write_json(output / "progress.json", {"fold": fold, "step": 2, "phase": "complete"})

                def poll(self):
                    return self.returncode

                def wait(self, timeout=None):
                    return self.returncode

            args = SimpleNamespace(data=ROOT.parents[1] / "dataset", gpus="2,4", execute=True)
            with mock.patch.object(entry, "ROOT", root), \
                 mock.patch.object(entry, "audit", return_value={"folds": []}), \
                 mock.patch.object(entry.subprocess, "Popen", FakeProcess), \
                 mock.patch.object(entry.subprocess, "check_output", return_value="fake test environment\n"), \
                 mock.patch.object(entry.time, "sleep"), \
                 mock.patch.object(entry.signal, "signal"), \
                 mock.patch("model326.aggregate_oof") as aggregate:
                entry.train(args)
                self.assertEqual([gpu for _, gpu in launched], ["2", "4", "2", "4", "2"])
                self.assertEqual(aggregate.call_count, 1)
                run_dir = Path(read_json(root / "outputs/latest_model326.json")["run"])
                self.assertEqual(read_json(run_dir / "status.json")["phase"], "complete")
                self.assertTrue((run_dir / "code_snapshot/configs/model326.json").is_file())
                with (root / "outputs/model326.lock").open("a+") as held:
                    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with self.assertRaisesRegex(RuntimeError, "Another Model326 run"):
                        entry.train(args)
                self.assertEqual(len(launched), 5)

    def test_archive_rejects_duplicates_missing_and_invalid_probabilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.npz"
            a = np.full((4, 3, 5), .25, dtype=np.float32)
            np.savez(path, A1=a, A1_image=a)
            with self.assertRaises(ValueError):
                model338.load_archive(path, {"A1": (3, 5)})
            np.savez(path, A1=a)
            with self.assertRaises(ValueError):
                model338.load_archive(path, {"A1": (3, 5), "A2": (3, 5)})
            with self.assertRaises(ValueError):
                model338.validate_probs(a * 2, (3, 5), "invalid")

    def test_error_panel_uses_ground_truth_colors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = np.zeros((480, 640), dtype=np.uint8)
            target[100:200, 100:200] = 2
            pred = target.copy()
            pred[150, 150] = 1
            original = np.full((480, 640, 3), 90, dtype=np.uint8)
            Image.fromarray(original).save(root / "image.png")
            comparison(root / "image.png", pred, target, .5, root / "panel.png")
            canvas = np.array(Image.open(root / "panel.png"))
            self.assertEqual(canvas.shape, (1088, 1280, 3))
            np.testing.assert_array_equal(canvas[150 + 64, 150 + 640], PALETTE[2])
            np.testing.assert_array_equal(canvas[64, 640], [255, 255, 255])
            np.testing.assert_array_equal(canvas[544 + 64 + 150, 150], PALETTE[1])
            np.testing.assert_array_equal(canvas[544 + 64 + 150, 640 + 150], PALETTE[2])

    def test_fixed_route_end_to_end_and_reordered_entries(self):
        rng = np.random.default_rng(47)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ("images", "masks"):
                (root / "valid" / folder).mkdir(parents=True)
            entries, archives = [], {"269": {}, "322": {}, "326": {}}
            for index in range(20):
                key = f"A{index+1}_step1_sample1"
                gray = rng.integers(0, 256, (24, 32), dtype=np.uint8)
                target = rng.integers(0, 4, gray.shape, dtype=np.uint8)
                Image.fromarray(gray).save(root / "valid/images" / f"{key}_image.png")
                Image.fromarray(PALETTE[target]).save(root / "valid/masks" / f"{key}_mask.png")
                g = gray.astype(np.float32)
                a = rng.random((4, *gray.shape), dtype=np.float32)
                a /= a.sum(0)
                item = {"target": model338.TEAM_LABEL_TO_MAIN[target], "z": (g-g.mean())/g.std(),
                        "q": g/255, "mixed": a, "extra": np.full_like(a, .25),
                        "prior_z_stats": [(0, 1)]*4, "prior_q_stats": [(.5, .3)]*4,
                        "params": {"beta": .5, "al_bias": .1, "eut_bias": -.1}}
                entries.append(item)
                for source, archive in archives.items():
                    p = rng.random(a.shape, dtype=np.float32)
                    p /= p.sum(0)
                    archive[key + ("_image" if source == "322" else "")] = p
            config = copy.deepcopy(read_json(ROOT / "configs/model338.json"))
            config["artifacts"]["entries"] = str(root / "entries.pkl")
            with (root / "entries.pkl").open("wb") as handle:
                pickle.dump(entries[::-1], handle)
            for source, archive in archives.items():
                path = root / f"{source}.npz"
                np.savez(path, **archive)
                config["artifacts"][source] = str(path)
            predictions = model338.infer(config, root)
            self.assertEqual(len(predictions), 20)
            # Verify input channel reordering and exact upstream route on one independent row.
            make_mixed, undo, weights, predict = model338.route_functions()
            key = "A1_step1_sample1"
            sources = {"269": archives["269"][key],
                       "322": archives["322"][key+"_image"][[1, 2, 0, 3]],
                       "326": archives["326"][key][[1, 2, 0, 3]]}
            sources["b4mean"] = (sources["322"] + sources["326"]) * .5
            original = predict(entries[0], make_mixed({"base": undo(entries[0]), "sources": sources}, config["setting"]),
                               config["setting"], weights(entries[0]["z"]), exact=True)
            np.testing.assert_array_equal(predictions[key], model338.MAIN_LABEL_TO_TEAM[original])
            # A repeated cache must fail rather than silently pair images by list position.
            with self.assertRaises(ValueError):
                model338.align_entries([entries[0]]*20, pairs(root, "valid"))
            result = export_results("SYNTHETIC TEST", predictions, root, results_root=root / "results")
            self.assertEqual(len(list((result / "valid_mask_comparison").glob("*.png"))), 20)
            summary = json.loads((result / "metrics.json").read_text())
            self.assertEqual(summary["empty_union_policy"], "zero")
            self.assertNotIn("global_miou", summary)
            self.assertAlmostEqual(summary["image_miou"], np.mean([r["image_miou"] for r in summary["images"]]))
            result2 = export_results("SYNTHETIC TEST", predictions, root, results_root=root / "results")
            self.assertNotEqual(result, result2)
            self.assertTrue((result / f"{result.name}.md").is_file())


if __name__ == "__main__":
    unittest.main()
