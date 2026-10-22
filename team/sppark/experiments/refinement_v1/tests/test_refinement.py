import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from alphase.data.dataset import TrainDataset
from alphase.data.splits import select_rows
from alphase.engine import trainer
from alphase.losses.segmentation import segmentation_loss as original_loss
from alphase.utils import write_json
from refinelab.dataset import RefineDataset, d4_array
from refinelab.evaluation import begin_output, finish_output, metric_summary
from refinelab.losses import segmentation_loss
from refinelab.models import ContextPhaseCNN, ResidualASPP, build_model, validate_parent
from refinelab.training import TerminalProgress, adapter
from refinelab.workspace import PARENT, config, output_path, plan, run_dir

torch.set_num_threads(4)


def test_ten_distinct_conditions_and_fixed_budget():
    entries = plan()
    assert len(entries) == 10
    signatures = []
    for name in entries:
        cfg = config(name)
        assert cfg["refinement"]["parent_checkpoint"] == str(PARENT)
        assert sum(s["max_updates"] for s in cfg["train"]["stages"]) == 2000
        assert cfg["seed"] == 2026
        assert cfg["augmentation"]["gaussian_noise_std"] == [0, 0]
        assert cfg["augmentation"]["scales"] == [1.]
        assert cfg["train"]["stages"][-1]["photometric_probability"] == 0
        assert run_dir(name) != run_dir(name, smoke=True)
        signature = [cfg[k] for k in ["model", "loss", "sampling", "train"]]
        signatures.append(json.dumps(signature, sort_keys=True))
    assert len(set(signatures)) == 10


def test_identity_aspp_reproduces_parent_logits():
    baseline = build_model(config("e00_ft_control")).eval()
    aspp = build_model(config("e05_aspp")).eval()
    x = torch.randn(1, 3, 64, 96)
    with torch.no_grad():
        a, b = baseline(x), aspp(x)
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_aspp_projection_then_branches_receive_gradients():
    module = ResidualASPP(16, 8)
    opt = torch.optim.SGD(module.parameters(), lr=.1)
    x = torch.randn(2, 16, 8, 10)
    assert torch.equal(module(x), x)
    module(x).square().mean().backward()
    assert module.project.weight.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
    opt.step()
    opt.zero_grad()
    module(x).square().mean().backward()
    assert module.branches[0][0].weight.grad.abs().sum() > 0


def test_new_architecture_round_trip(tmp_path):
    cfg = config("e05_aspp")
    a = build_model(cfg, load_pretrained=False).eval()
    torch.save(a.state_dict(), tmp_path / "model.pt")
    b = build_model(cfg, load_pretrained=False).eval()
    b.load_state_dict(torch.load(tmp_path / "model.pt", weights_only=True))
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(a(x)["logits"], b(x)["logits"], rtol=0, atol=0)


def test_weighted_ce_is_real_and_ignores_padding():
    z = torch.randn(1, 4, 3, 3, requires_grad=True)
    target = torch.tensor([[[0, 1, 2], [2, 3, 255], [1, 0, 2]]])
    weights = {**config("e02_weighted")["loss"], "dice": 0, "lovasz": 0}
    got, _ = segmentation_loss({"logits": z}, target, weights)
    valid = target[0] != 255
    expected = F.cross_entropy(z[0, :, valid].T, target[0, valid], weight=torch.tensor(weights["class_weights"]))
    torch.testing.assert_close(got, expected)
    raw = F.cross_entropy(z[0, :, valid].T, target[0, valid])
    assert not torch.isclose(got, raw)
    got.backward()
    assert torch.all(z.grad[0, :, 1, 2] == 0)


def test_unweighted_loss_identical_to_frozen():
    z = torch.randn(1, 4, 6, 6, requires_grad=True)
    target = torch.randint(0, 4, (1, 6, 6))
    cfg = config("e00_ft_control")
    a, _ = segmentation_loss({"logits": z}, target, cfg["loss"])
    b, _ = original_loss({"logits": z}, target, cfg["loss"])
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("stage_index", [0, 1, 2])
def test_control_data_matches_original(stage_index):
    cfg = config("e00_ft_control")
    train, _ = select_rows(cfg, "official")
    stage = cfg["train"]["stages"][stage_index]
    a, b = TrainDataset(train, cfg, stage, stage_index), RefineDataset(train, cfg, stage, stage_index)
    for index in [0, 3, 9]:
        x, y = a[index], b[index]
        assert x["image_id"] == y["image_id"]
        for key in ("detail", "target", "box", "dino"):
            assert torch.equal(x[key], y[key])


def test_hard_maps_d4_alignment():
    target = np.arange(11 * 17).reshape(11, 17) % 4
    hard = target == 2
    for k in range(8):
        assert np.array_equal(d4_array(hard, k), d4_array(target, k) == 2)


def test_mining_rejects_validation_ids(tmp_path):
    from refinelab.mining import verify_mining
    cfg = config("e01_hard")
    cfg["refinement"]["mining_folder"] = str(tmp_path)
    train, valid = select_rows(cfg, "official")
    write_json(tmp_path / "request.json", dict(train_ids=[r["image_id"] for r in train], validation_images_used=[]))
    records = {r["image_id"]: {} for r in train}
    records[valid[0]["image_id"]] = {}
    write_json(tmp_path / "manifest.json", records)
    with pytest.raises(ValueError, match="exactly the official training"):
        verify_mining(cfg)


def test_parent_split_leakage_is_rejected():
    cfg = config("e00_ft_control")
    train, valid = select_rows(cfg, "official")
    state = dict(train_ids=[r["image_id"] for r in train] + [valid[0]["image_id"]],
                 validation_ids=[r["image_id"] for r in valid])
    with pytest.raises(ValueError, match="same official"):
        validate_parent(state, cfg)


def test_terminal_progress_and_restore(tmp_path):
    cfg = config("e00_ft_control", smoke=True)
    p = TerminalProgress(tmp_path, cfg, 8)
    p.write(0, 1, 1, seconds_per_update=.5)
    assert json.loads((tmp_path / "status.json").read_text())["remaining_seconds"] > 0
    assert not list(tmp_path.glob("*.html"))
    old = trainer.TrainDataset, trainer.build_model
    with adapter():
        assert trainer.TrainDataset is RefineDataset
    assert (trainer.TrainDataset, trainer.build_model) == old


def test_metric_definition_is_image_mean_and_global_separately():
    cms = [np.diag([90, 90, 90, 90]), np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])]
    frame = pd.DataFrame(dict(confusion_matrix=[json.dumps(cm.tolist()) for cm in cms], mIoU_exclude=[100, 0]))
    scores = metric_summary(frame)
    assert scores["image_mIoU"] == 50
    assert scores["global_mIoU"] == pytest.approx(100 * 90 / 92)


def test_interrupted_evaluation_is_preserved_and_completed_artifacts_checked(tmp_path):
    p = tmp_path / "d4"
    expected = {"checkpoint": "fixed"}
    assert begin_output(p, expected)
    (p / "partial.txt").write_text("retain")
    assert begin_output(p, expected)
    assert len(list(tmp_path.glob("d4.interrupted-*/partial.txt"))) == 1
    (p / "result.txt").write_text("score")
    finish_output(p, expected)
    assert not begin_output(p, expected)
    (p / "result.txt").write_text("tampered")
    with pytest.raises(ValueError):
        begin_output(p, expected)


def test_output_traversal_rejected():
    with pytest.raises(ValueError):
        output_path("../../../../runs")
