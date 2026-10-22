"""Metric and output contracts; synthetic fixtures only, no model training."""

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime
import run
from komap_seg.palette import PALETTE


def test_absent_class_is_one_with_three_08_classes():
    # Classes 0, 1, 3: TP=8, FP=FN=1, hence IoU=0.8 each. Class 2 absent.
    target = np.array([[0] * 9 + [1] * 9 + [3] * 9], dtype=np.uint8)
    pred = np.array([[0] * 8 + [1] + [1] * 8 + [3] + [3] * 8 + [0]], dtype=np.uint8)
    score = runtime.image_metrics(pred, target)
    assert score["class_iou"] == pytest.approx([0.8, 0.8, 1.0, 0.8])
    assert score["image_miou"] == pytest.approx(0.85)


def test_absent_ground_truth_with_false_positive_is_zero():
    score = runtime.image_metrics(np.array([[0, 2]]), np.array([[0, 0]]))
    assert score["class_iou"] == pytest.approx([0.5, 1.0, 0.0, 1.0])
    assert score["image_miou"] == pytest.approx(0.625)


def test_images_have_equal_weight_despite_different_sizes():
    small = runtime.image_metrics(
        np.ones((1, 1), dtype=int), np.zeros((1, 1), dtype=int)
    )
    large = runtime.image_metrics(
        np.zeros((100, 100), dtype=int), np.zeros((100, 100), dtype=int)
    )
    assert (
        runtime.summarize(
            [dict(image_id="small", **small), dict(image_id="large", **large)]
        )["image_miou"]
        == 0.75
    )


def test_strict_palette_and_alpha(tmp_path):
    labels = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    rgb = PALETTE[labels]
    mask = tmp_path / "mask.png"
    Image.fromarray(rgb).save(mask)
    np.testing.assert_array_equal(runtime.strict_mask(mask), labels)
    rgba = np.concatenate([rgb, np.full((2, 2, 1), 255, dtype=np.uint8)], axis=2)
    Image.fromarray(rgba).save(mask)
    np.testing.assert_array_equal(runtime.strict_mask(mask), labels)
    rgba[0, 0, 3] = 0
    Image.fromarray(rgba).save(mask)
    with pytest.raises(ValueError, match="Nonopaque"):
        runtime.strict_mask(mask)
    rgb[0, 0] = [1, 2, 3]
    Image.fromarray(rgb).save(mask)
    with pytest.raises(ValueError, match="Unknown mask colors"):
        runtime.strict_mask(mask)


def test_comparison_layout_and_errors_use_ground_truth(tmp_path):
    h, w = 10, 12
    original = np.full((h, w, 3), 123, dtype=np.uint8)
    target = np.zeros((h, w), dtype=np.uint8)
    target[0, 0] = 2
    pred = np.zeros_like(target)
    out = tmp_path / "comparison.png"
    runtime.comparison(out, original, pred, target, 0.7)
    canvas = np.array(Image.open(out))
    x1, x2 = 16, 16 + w + 24
    y1, y2 = 16 + 76, 16 + 76 + h + 24 + 76
    np.testing.assert_array_equal(canvas[y1 : y1 + h, x1 : x1 + w], original)
    np.testing.assert_array_equal(canvas[y2 : y2 + h, x1 : x1 + w], PALETTE[pred])
    np.testing.assert_array_equal(canvas[y2 : y2 + h, x2 : x2 + w], PALETTE[target])
    np.testing.assert_array_equal(canvas[y1, x2], PALETTE[2])
    np.testing.assert_array_equal(canvas[y1 + 1, x2 + 1], [255, 255, 255])


def test_new_results_do_not_overwrite_old_results(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "REPO", tmp_path)
    first = runtime.reserve_result("0928")
    marker = first / "existing.txt"
    marker.write_text("preserve")
    second = runtime.reserve_result("0928")
    assert first.name == "model_results_0928_1"
    assert second.name == "model_results_0928_2"
    assert marker.read_text() == "preserve"


class ConstantModel(torch.nn.Module):
    def forward(self, x):
        logits = x.new_zeros((x.shape[0], 4, *x.shape[-2:]))
        logits[:, 0] = 5
        return logits


def test_evaluation_exports_native_masks_report_and_comparison(tmp_path, monkeypatch):
    cfg = runtime.load_config()
    cfg.update(
        data_root=str(tmp_path / "data"),
        patch_size=32,
        input_scale=2.0,
        overlap=4,
        device="cpu",
    )
    for sub in ["images", "masks"]:
        (tmp_path / "data/valid" / sub).mkdir(parents=True)
    shape = (19, 27)
    Image.fromarray(np.full((*shape, 3), 100, dtype=np.uint8)).save(
        tmp_path / "data/valid/images/sample_image.png"
    )
    labels = np.zeros(shape, dtype=np.uint8)
    Image.fromarray(PALETTE[labels]).save(tmp_path / "data/valid/masks/sample_mask.png")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint stub; model supplied by test")
    monkeypatch.setattr(run, "load_checkpoint", lambda path, config: ConstantModel())
    result = run.evaluate(cfg, checkpoint, output=tmp_path / "model_results_0928_1")
    np.testing.assert_array_equal(
        runtime.strict_mask(result / "predictions/sample_mask.png"), labels
    )
    report = json.loads((result / "summary.json").read_text())
    assert report["image_miou"] == 1.0
    assert report["inference"]["tta"] == "four_flip"
    assert report["checkpoint_sha256"] == runtime.sha256(checkpoint)
    assert (result / "valid_mask_comparison/sample.png").is_file()
    assert (result / "model_results_0928_1.md").is_file()
    assert (result / "per_image.csv").is_file()
    assert (result / "per_class.csv").is_file()
    assert (
        checkpoint.read_bytes() == b"synthetic checkpoint stub; model supplied by test"
    )
    assert not (checkpoint.parent / "evaluation_latest.json").exists()
    with pytest.raises(FileExistsError):
        run.evaluate(cfg, checkpoint, output=result)
