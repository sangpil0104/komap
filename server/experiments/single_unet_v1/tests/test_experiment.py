import copy

import numpy as np
import pytest
import torch

from singlelab.common import config
from singlelab.data import PatchDataset, boundary, boundary_band, crop
from singlelab.inference import d4, predict, tile_starts, undo_d4
from singlelab.loss import segmentation_loss
from singlelab.metrics import confusion, ious, measure, summarize
from singlelab.model import EfficientUNet


def test_fixed_four_class_empty_policies_and_image_weighting():
    target = np.zeros((2, 2), dtype=np.uint8)
    cm = confusion(target, target)
    assert ious(cm, "one").mean() == 1
    assert ious(cm, "zero").mean() == .25
    assert np.nanmean(ious(cm, "exclude")) == 1
    options = config("S0")["evaluation"]
    a = measure(target, target, options)
    b = measure(np.zeros((20, 20), np.uint8), np.ones((20, 20), np.uint8), options)
    result = summarize([a, b], "one")
    assert result["image_miou"] == 75  # Equal image weight, not pixel-weighted.
    assert result["miou_zero"] == 12.5


def test_boundary_ignores_padding_and_covers_both_sides():
    t = np.array([[255, 0, 0, 1, 1]], np.uint8)
    assert boundary(t).tolist() == [[False, False, True, True, False]]
    assert not boundary_band(np.zeros((7, 7), np.uint8), 2).any()
    assert crop(np.ones((2, 3)), -2, -2, 7, 8).shape == (8, 7)


def test_loss_halo_does_not_change_supervision_and_gradients_are_finite():
    cfg = config("S3", scale=2)["loss"]
    z = torch.randn(1, 4, 8, 8, requires_grad=True)
    target = torch.full((1, 8, 8), 255, dtype=torch.long)
    target[:, 2:6, 2:6] = 0
    edge = torch.ones_like(target, dtype=torch.float32)
    loss, _ = segmentation_loss(z, target, edge, cfg)
    loss.backward()
    assert torch.isfinite(z.grad).all()
    assert (z.grad[:, :, :2] == 0).all()
    assert (z.grad[:, :, 2:6, 2:6].abs().sum() > 0)


def test_scale_does_not_change_native_crop_or_target(monkeypatch):
    from singlelab import data
    g = np.arange(80 * 96, dtype=np.float32).reshape(80, 96) / (80 * 96)
    t = (np.indices(g.shape)[1] // 24).astype(np.uint8)
    monkeypatch.setattr(data, "gray", lambda p: g)
    monkeypatch.setattr(data, "labels", lambda p: t)
    samples = []
    for experiment in ("S0", "S1", "S2"):
        cfg = config(experiment)
        cfg["sampling"] = dict(core=32, halo=16)
        samples.append(PatchDataset([dict(image_path="i", mask_path="m")], cfg)[17])
    for scale, (_, target, edge) in enumerate(samples, 1):
        assert torch.equal(target[::scale, ::scale], samples[0][1])
        assert torch.equal(edge[::scale, ::scale], samples[0][2])
    assert samples[2][0].shape == (3, 192, 192)


class Pointwise(torch.nn.Module):
    def forward(self, x):
        return torch.cat((x[:, :1], 1 - x[:, :1], x[:, :1] * 0, x[:, :1] * 0 + .5), 1)


@pytest.mark.parametrize("shape", [(17, 23), (77, 93)])
@pytest.mark.parametrize("scale", [1, 2, 3])
def test_tiling_padding_shift_and_tta_cover_original_image(shape, scale):
    cfg = config("S0")
    cfg["scale"] = scale
    cfg["inference"].update(core=32, halo=16, offset=8, d4=True)
    g = np.random.default_rng(7).random(shape, dtype=np.float32)
    tiled = predict(Pointwise(), g, cfg, torch.device("cpu"))
    cfg["inference"].update(mode="full", pad=16, d4=False)
    full = predict(Pointwise(), g, cfg, torch.device("cpu"))
    np.testing.assert_allclose(tiled, full, atol=1e-6)
    np.testing.assert_allclose(tiled.sum(0), 1, atol=1e-6)
    assert tiled.shape == (4, *shape)


def test_d4_inverse_and_shifted_grid():
    image = np.arange(35).reshape(5, 7)
    for index in range(8):
        np.testing.assert_array_equal(undo_d4(d4(image, index), index), image)
    assert tile_starts(100, 32, 16, 8) == [0, 8, 24, 40, 56, 68]


def test_model_full_resolution_and_frozen_encoder_stats():
    torch.set_num_threads(2)
    model = EfficientUNet(config("S0"), pretrained=False).train()
    assert all(not m.training for m in model.encoder.modules() if isinstance(m, torch.nn.BatchNorm2d))
    x = torch.rand(1, 3, 64, 96)
    output = model(x)
    assert output.shape == (1, 4, 64, 96)
    output.mean().backward()
    assert model.encoder[0][0].weight.grad is not None
    assert torch.isfinite(model.head.weight.grad).all()


def test_s3_requires_explicit_scale_and_changes_only_edge_weight():
    with pytest.raises(ValueError):
        config("S3")
    a, b = config("S1"), config("S3", scale=2)
    b["experiment"] = "S1"
    b["loss"]["boundary_weight"] = 0
    assert a == b


def test_evaluation_report_and_masks_without_training(tmp_path, monkeypatch):
    from PIL import Image
    from singlelab import evaluation, report
    from singlelab.common import PALETTE, read_json
    cfg = config("S0")
    cfg["inference"].update(core=32, halo=16)
    target = (np.indices((48, 64))[1] // 16).astype(np.uint8)
    img = tmp_path / "input.png"
    mask = tmp_path / "truth.png"
    Image.fromarray(target * 60).save(img)
    Image.fromarray(PALETTE[target]).save(mask)
    row = dict(image_id="synthetic", image_path=str(img), mask_path=str(mask))
    monkeypatch.setattr(evaluation, "records", lambda c: {"valid": [row], "test": [row]})
    monkeypatch.setattr(report, "PROJECT", tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    checkpoint = run / "dummy.pt"
    checkpoint.write_bytes(b"Synthetic test placeholder, not trained weights")
    bundle = Pointwise(), cfg, checkpoint, 0
    output = evaluation.evaluate(run, torch.device("cpu"), model_bundle=bundle)
    test_output = evaluation.evaluate(run, torch.device("cpu"), split="test", model_bundle=bundle)
    first = report.make_report([output, test_output])
    before = (first / f"{first.name}.md").read_bytes()
    second = report.make_report([output])
    assert first != second
    assert before == (first / f"{first.name}.md").read_bytes()
    assert read_json(output / "summary.json")["num_images"] == 1
    assert read_json(test_output / "summary.json")["test_has_no_labels"]
    assert "image_miou" not in read_json(test_output / "summary.json")
    with Image.open(output / "masks/synthetic_mask.png") as prediction:
        assert prediction.mode == "RGBA" and prediction.size == (64, 48)
        assert np.all(np.asarray(prediction)[..., 3] == 255)
    assert (first / "01_S0_seed2026/comparisons/synthetic.png").exists()
