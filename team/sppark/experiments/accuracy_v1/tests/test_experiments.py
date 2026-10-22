from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pytest
import torch
import yaml

from alphase.losses.segmentation import segmentation_loss
from alphase.models.factory import build_model
from lab.inference import select_images, validate_probability, weighted_average
from lab.training import training_config
from lab.workspace import BASELINE, SOURCE, load_experiment, output_path, read_json, safe_name


def test_frozen_source_is_imported():
    import alphase
    assert Path(alphase.__file__).resolve().is_relative_to(SOURCE)


def remove_identity(cfg):
    cfg = deepcopy(cfg)
    for key in ("experiment", "data"):
        cfg.pop(key)
    return cfg


def test_controlled_ablation_configs():
    base = yaml.safe_load((SOURCE/"configs/cnn_unet.yaml").read_text())
    candidate = load_experiment("fullframe")
    assert [s["max_updates"] for s in candidate["train"]["stages"]] == [2000, 1000, 1500]
    for new, old in zip(candidate["train"]["stages"], base["train"]["stages"]):
        new["max_updates"] = old["max_updates"]
    assert remove_identity(candidate) == remove_identity(base)
    candidate = load_experiment("boundary")
    assert candidate["loss"]["boundary"] == .05
    candidate["loss"]["boundary"] = 0.
    assert remove_identity(candidate) == remove_identity(base)
    for seed in (2027, 2028):
        candidate = load_experiment(f"seed{seed}")
        assert candidate["seed"] == seed
        candidate["seed"] = base["seed"]
        assert remove_identity(candidate) == remove_identity(base)


def test_smoke_does_not_change_real_config():
    smoke = training_config("boundary", smoke=True)
    normal = training_config("boundary")
    assert sum(s["max_updates"] for s in smoke["train"]["stages"]) == 3
    assert sum(s["max_updates"] for s in normal["train"]["stages"]) == 4500
    assert smoke["loss"]["boundary"] == normal["loss"]["boundary"]


def test_probability_average_before_argmax():
    # Hard votes pick class 0 (two votes). Probabilities correctly pick class 1.
    arrays = [np.array(p, np.float32)[:, None, None] for p in ([.51,.49,0,0], [.51,.49,0,0], [0,1,0,0])]
    result = weighted_average(arrays, [1,1,1])
    assert result.argmax(0).item() == 1
    np.testing.assert_allclose(result[:, 0, 0], [.34,.66,0,0], atol=1e-6)
    np.testing.assert_allclose(weighted_average(arrays, [2,2,2]), result)
    np.testing.assert_allclose(weighted_average(arrays, [1e308,1e308,1e308]), result)


@pytest.mark.parametrize("weights", [[0,1], [-1,2], [np.nan,1], [1]])
def test_invalid_ensemble_weights(weights):
    a = np.ones((4,2,3), np.float32)/4
    with pytest.raises(ValueError):
        weighted_average([a,a], weights)


def test_probability_layout_validation():
    a = np.ones((4,2,3), np.float32)/4
    validate_probability(a,2,3)
    with pytest.raises(ValueError):
        validate_probability(a.astype(np.float16),2,3)
    with pytest.raises(ValueError):
        validate_probability(a,3,2)
    with pytest.raises(ValueError):
        validate_probability(a*2,2,3)


def test_output_traversal_rejected():
    with pytest.raises(ValueError):
        output_path("../baseline/run")
    with pytest.raises(ValueError):
        safe_name("../old_run")


def test_heldout_and_all90_isolation():
    import pandas as pd
    table = pd.read_csv(BASELINE/"data/manifest.csv")
    state = dict(train_ids=table[table.source_split == "train"].image_id.tolist(),
                 validation_ids=table[table.source_split == "valid"].image_id.tolist())
    assert len(select_images(state,"valid")) == 20
    assert len(select_images(state,"test")) == 10
    state["train_ids"].append(state["validation_ids"][0])
    with pytest.raises(ValueError, match="overlap"):
        select_images(state,"valid")
    state["validation_ids"] = []
    with pytest.raises(ValueError, match="no held-out"):
        select_images(state,"valid")


def test_boundary_auxiliary_has_finite_gradient():
    torch.set_num_threads(2)
    cfg = load_experiment("boundary")
    model = build_model(cfg)
    x = torch.rand(1,3,64,64)
    target = torch.randint(0,4,(1,64,64))
    target[:, :8] = 255
    loss, _ = segmentation_loss(model(x), target, cfg["loss"])
    loss.backward()
    gradient = model.boundary_head.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert model.classifier.weight.grad is not None


def test_original_checkpoint_palette_and_selected_weights():
    state = torch.load(BASELINE/"run/best.pt",map_location="cpu",weights_only=False)
    assert state["selected_weights"] in ["raw","ema"]
    assert len(state["train_ids"]) == 70 and len(state["validation_ids"]) == 20
    assert abs(state["best_metric"]-78.47287474071697) < 1e-9


@pytest.mark.parametrize("field", ["dataset_hash", "class_names", "palette", "images", "train_ids", "validation_ids", "inference"])
def test_ensemble_rejects_mismatched_provenance(tmp_path, monkeypatch, field):
    import lab.inference as inference
    checkpoints = []
    for i in range(2):
        p = tmp_path/f"{i}.pt"
        p.write_bytes(bytes([i]))
        checkpoints.append(p)
    monkeypatch.setattr(inference, "checkpoint_for", lambda name, *args: checkpoints[int(name[-1])])
    signature = dict(dataset_hash="data", class_names=["a","b","c","d"], palette=[[0,0,0]]*4,
                     images=["same-image"], train_ids=["train"], validation_ids=["valid"], inference={"d4":False})
    def cache(checkpoint, *args):
        value = deepcopy(signature)
        if checkpoint == checkpoints[1]:
            value[field] = "mismatch"
        return tmp_path, [], {"signature":value}
    monkeypatch.setattr(inference,"cache_probabilities",cache)
    config = tmp_path/"ensemble.yaml"
    config.write_text(yaml.safe_dump(dict(name="test",members=[dict(experiment="member0"),dict(experiment="member1")])))
    with pytest.raises(ValueError, match=field):
        inference.ensemble(config)
