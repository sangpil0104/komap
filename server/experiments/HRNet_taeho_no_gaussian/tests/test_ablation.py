"""Check the controlled augmentation change and concurrent experiment launcher."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "HRNet_taeho"
sys.path.insert(0, str(ROOT))
import runtime
from komap_seg.dataset import KomapPatchDataset, build_train_transform
from komap_seg.palette import PALETTE


def test_only_noise_blur_removed_from_source_and_config():
    before = json.loads((BASE / "configs/default.json").read_text())
    after = json.loads((ROOT / "configs/default.json").read_text())
    assert after.pop("model_name") == "HRNet_taeho_no_gaussian"
    before.pop("model_name")
    assert before == after
    modified = []
    for row in json.loads((ROOT / "source_manifest.json").read_text()):
        source = (BASE / "source" / row["path"]).read_text()
        ablation = (ROOT / "source" / row["path"]).read_text()
        if source != ablation:
            modified.append(row["path"])
            expected = source.replace("            A.GaussNoise(std_range=(0.02, 0.08), p=0.2),\n", "")
            expected = expected.replace("            A.GaussianBlur(blur_limit=(3, 5), p=0.15),\n", "")
            assert ablation == expected
    assert modified == ["komap_seg/dataset.py"]


def test_actual_augmentation_pipeline_and_dataset(tmp_path):
    spec = importlib.util.spec_from_file_location("baseline_dataset", BASE / "source/komap_seg/dataset.py")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    expected = baseline.build_train_transform().to_dict()
    ops = expected["transform"]["transforms"]
    removed = {"GaussNoise", "GaussianBlur"}
    expected["transform"]["transforms"] = [op for op in ops if op["__class_fullname__"] not in removed]
    assert build_train_transform().to_dict() == expected
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, (300, 320, 3), dtype=np.uint8)
    labels = np.indices((300, 320)).sum(0).astype(np.uint8) % 4
    Image.fromarray(image).save(tmp_path / "images/check_image.png")
    Image.fromarray(PALETTE[labels]).save(tmp_path / "masks/check_mask.png")
    dataset = KomapPatchDataset(tmp_path / "images", tmp_path / "masks", seed=0)
    for index in range(3):
        x, y = dataset[index]
        assert x.shape == (3, 512, 512)
        assert y.shape == (512, 512)
        assert torch.isfinite(x).all()
        assert y.dtype == torch.int64 and set(y.unique().tolist()) <= {0, 1, 2, 3}


def test_concurrent_result_reservations_are_distinct(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "REPO", tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _: runtime.reserve_result("0928"), range(8)))
    assert len(set(paths)) == 8
    assert all(path.is_dir() for path in paths)


def test_pair_launcher_routes_gpus_without_training(tmp_path):
    base, variant = tmp_path / "HRNet_taeho", tmp_path / ROOT.name
    base.mkdir()
    variant.mkdir()
    launcher = variant / "start_pair_nohup.sh"
    launcher.write_bytes((ROOT / "start_pair_nohup.sh").read_bytes())
    # Stub launchers record the intended GPU; they never start Python or training.
    for name, folder in [("baseline", base), ("ablation", variant)]:
        stub = folder / "start_nohup.sh"
        stub.write_text(f'#!/usr/bin/env bash\nprintf "{name}:%s\\n" "$HRNET_GPU" >> "$PAIR_TEST_LOG"\n')
        stub.chmod(0o755)
    log = tmp_path / "calls.txt"
    env = dict(os.environ, BASELINE_GPU="2", ABLATION_GPU="3", PAIR_TEST_LOG=str(log))
    subprocess.run(["bash", str(launcher), "--dry-run"], env=env, check=True, capture_output=True)
    assert not log.exists()
    subprocess.run(["bash", str(launcher)], env=env, check=True, capture_output=True)
    assert log.read_text().splitlines() == ["baseline:2", "ablation:3"]
    original = log.read_text()
    env["ABLATION_GPU"] = "2"
    denied = subprocess.run(["bash", str(launcher)], env=env, capture_output=True)
    assert denied.returncode == 2 and log.read_text() == original
