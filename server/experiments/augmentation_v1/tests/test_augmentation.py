import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from alphase.data.dataset import TrainDataset, load_image, load_target
from alphase.data.splits import select_rows
from alphase.engine import trainer
from auglab.dataset import AugmentedDataset, resize_pair, transform
from auglab.training import TerminalProgress, data_adapter
from auglab.workspace import SOURCE, config, output_path, read_json, run_dir


@pytest.mark.parametrize("stage_index", [0, 1, 2])
def test_control_matches_original_every_tensor(stage_index):
    cfg = config("control")
    rows, _ = select_rows(cfg, "official")
    stage = cfg["train"]["stages"][stage_index]
    original = TrainDataset(rows, cfg, stage, stage_index)
    adapter = AugmentedDataset(rows, cfg, stage, stage_index)
    for index in [0, 1, 7, 23]:
        a, b = original[index], adapter[index]
        assert a["image_id"] == b["image_id"]
        for key in ("detail", "dino", "box", "target"):
            assert torch.equal(a[key], b[key]), (stage_index, index, key)


@pytest.mark.parametrize("name", ["photo_mild", "noise_mild", "scale_mild"])
def test_final_stage_is_identical_to_control(name):
    cfg, control = config(name), config("control")
    rows, _ = select_rows(cfg, "official")
    a = AugmentedDataset(rows, cfg, cfg["train"]["stages"][2], 2)
    b = AugmentedDataset(rows, control, control["train"]["stages"][2], 2)
    for key in ("detail", "target", "box"):
        assert torch.equal(a[3][key], b[3][key])


@pytest.mark.parametrize("name", ["photo_mild", "noise_mild", "scale_mild"])
def test_index_repeat_and_cached_inputs_unchanged(name):
    cfg = config(name)
    rows, _ = select_rows(cfg, "official")
    # Force the newly added transformation to exercise its path.
    if name == "noise_mild":
        cfg["augmentation_extension"]["noise_probability"] = 1
    if name == "scale_mild":
        cfg["augmentation_extension"].update(scale_probability=1, scale_choices=[.9])
    image = load_image(rows[0]["image_path"])
    mask = load_target(rows[0]["mask_path"])
    image_before, mask_before = image.copy(), mask.copy()
    ds = AugmentedDataset(rows[:1], cfg, cfg["train"]["stages"][0])
    a, b = ds[4], ds[4]
    assert torch.equal(a["detail"], b["detail"])
    assert torch.equal(a["target"], b["target"])
    assert torch.isfinite(a["detail"]).all()
    assert set(a["target"].unique().tolist()) <= {0, 1, 2, 3, 255}
    assert np.array_equal(image, image_before) and np.array_equal(mask, mask_before)


def test_d4_preserves_image_mask_alignment():
    cfg = config("control")
    aug = {**cfg["augmentation"], "photometric_probability": 0}
    target = np.arange(15 * 23, dtype=np.uint8).reshape(15, 23) % 4
    for seed in range(24):
        g, m = transform(target.astype(np.float32) / 3, target, np.random.default_rng(seed), aug, {}, {"mode": "patch"})
        np.testing.assert_allclose(g * 3, m, atol=1e-6)


def test_noise_is_image_only_and_has_expected_strength():
    cfg = config("noise_mild")
    aug = {**cfg["augmentation"], "d4": False, "photometric_probability": 0}
    ext = {"noise_probability": 1, "noise_std": [.01, .01]}
    g = np.full((200, 200), .5, np.float32)
    m = np.arange(40000, dtype=np.int64).reshape(200, 200).astype(np.uint8) % 4
    after, target = transform(g, m, np.random.default_rng(1), aug, ext, {"mode": "patch"})
    assert .0095 < float((after-g).std()) < .0105
    assert np.array_equal(target, m)
    assert np.all(g == .5)


def test_scale_masks_keep_categories_and_pixel_alignment():
    m = np.zeros((20, 30), np.uint8)
    m[:, 15:] = 3
    g, target = resize_pair(m.astype(np.float32) / 3, m, 2)
    assert g.shape == target.shape == (40, 60)
    assert set(np.unique(target)) == {0, 3}
    assert np.all(target[:, :30] == 0) and np.all(target[:, 30:] == 3)
    assert np.all(g[:, :29] == 0) and np.all(g[:, 31:] == 1)
    m[0, 0] = 255
    _, target = resize_pair(m.astype(np.float32) / 255, m, 2)
    assert 255 in target


@pytest.mark.parametrize("name", ["control", "photo_mild", "noise_mild", "scale_mild"])
def test_controlled_config_and_smoke_isolation(name):
    cfg = config(name)
    assert sum(s["max_updates"] for s in cfg["train"]["stages"]) == 4500
    assert cfg["model"] == {"name": "cnn_unet"}
    assert cfg["seed"] == 2026 and cfg["train"]["effective_batch_size"] == 8
    assert cfg["train"]["stages"] == config("control")["train"]["stages"]
    assert cfg["loss"] == config("control")["loss"]
    assert cfg["inference"] == config("control")["inference"]
    assert cfg["augmentation"]["scales"] == [1.]
    assert sum(s["max_updates"] for s in config(name, smoke=True)["train"]["stages"]) == 3
    assert run_dir(name) != run_dir(name, smoke=True)


def test_terminal_progress_has_eta_without_html(tmp_path):
    cfg = config("control", smoke=True)
    progress = TerminalProgress(tmp_path, cfg, 8)
    progress.write(0, 1, 1, seconds_per_update=.5, loss=.8)
    status = read_json(tmp_path / "status.json")
    assert status["remaining_seconds"] > 0 and status["percent"] == pytest.approx(100 / 3)
    assert not list(tmp_path.glob("*.html"))


def test_adapter_restores_frozen_engine():
    original = trainer.TrainDataset, trainer.Progress
    with data_adapter():
        assert trainer.TrainDataset is AugmentedDataset
        assert trainer.Progress is TerminalProgress
    assert (trainer.TrainDataset, trainer.Progress) == original
    assert str(Path(trainer.__file__).resolve()).startswith(str(SOURCE))


def test_outputs_reject_traversal():
    with pytest.raises(ValueError):
        output_path("../../../../runs")


def test_group_fold_has_no_prefix_overlap():
    train, valid = select_rows(config("control"), "fold0")
    assert not {r["group_id"] for r in train} & {r["group_id"] for r in valid}
