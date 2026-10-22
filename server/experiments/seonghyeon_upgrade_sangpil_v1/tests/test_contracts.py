from pathlib import Path
import json
import numpy as np
import pytest
import torch
from alphase.contracts import (
    PALETTE,
    rgb_to_ids,
    ids_to_rgb,
    confusion_matrix,
    metrics_from_cm,
    summarize_images,
    d4_forward,
    d4_inverse,
    align_context_features,
    restore_native_probs,
    relative_features,
)


def test_palette_roundtrip():
    ids = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.uint8)
    assert np.array_equal(rgb_to_ids(ids_to_rgb(ids)), ids)


def test_unknown_palette_color_fails():
    image = PALETTE.reshape(1, 4, 3).copy()
    image[0, 0, 0] += 1
    with pytest.raises(ValueError):
        rgb_to_ids(image)


def test_rgba_alpha_and_invalid_id_fail():
    rgba = np.concatenate(
        (PALETTE.reshape(1, 4, 3), np.full((1, 4, 1), 255, dtype=np.uint8)), axis=-1
    )
    assert np.array_equal(rgb_to_ids(rgba), np.arange(4, dtype=np.uint8)[None, :])
    rgba[0, 0, 3] = 1
    with pytest.raises(ValueError):
        rgb_to_ids(rgba)
    with pytest.raises(ValueError):
        ids_to_rgb(np.array([[255]], dtype=np.uint8))


def test_a71_exact_fixture():
    fixture = json.loads(
        (Path(__file__).parents[1] / "input/fixtures/a71_confusion.json").read_text()
    )
    result = metrics_from_cm(np.asarray(fixture["confusion_matrix"], dtype=np.int64))
    assert result["image_miou"] == pytest.approx(0.7113850263, abs=1e-10)
    assert result["pixel_error_rate"] == pytest.approx(0.1263641975, abs=1e-10)
    assert result["error_pixels"] == 40942
    assert result["Al3Ni_to_EutecticSi_count"] == 19939
    assert result["iou"][2] == 1.0
    assert result["precision"][2] is None


def test_empty_union_and_false_positive_are_different():
    target = np.zeros((2, 2), dtype=np.int64)
    correct = target.copy()
    wrong = target.copy()
    wrong[0, 0] = 2
    good = metrics_from_cm(confusion_matrix(target, correct))
    bad = metrics_from_cm(confusion_matrix(target, wrong))
    assert good["iou"][2] == 1
    assert bad["iou"][2] == 0
    assert good["image_miou"] == 1


def test_ignore_and_bad_shapes():
    target = np.array([[0, 255], [1, 255]], dtype=np.int64)
    pred = np.array([[0, 3], [1, 2]], dtype=np.int64)
    assert confusion_matrix(target, pred).sum() == 2
    with pytest.raises(ValueError):
        confusion_matrix(np.full((2, 2), 255, dtype=np.int64), pred)
    with pytest.raises(ValueError):
        confusion_matrix(target, pred[:, :1])


def test_image_global_differ_and_manual_average():
    a = np.diag(np.array([90, 10, 0, 0], dtype=np.int64))
    b = np.array(
        [[500, 500, 0, 0], [0, 100, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int64
    )
    out = summarize_images([a, b])
    assert out["val_image_miou"] == pytest.approx(
        (metrics_from_cm(a)["image_miou"] + metrics_from_cm(b)["image_miou"]) / 2
    )
    assert abs(out["val_image_miou"] - out["val_global_miou"]) > 0.01


@pytest.mark.parametrize("k", range(4))
@pytest.mark.parametrize("flip", [False, True])
def test_d4_inverse_non_square(k, flip):
    x = torch.arange(2 * 3 * 5).view(2, 3, 5)
    assert torch.equal(d4_inverse(d4_forward(x, k, flip), k, flip), x)


def test_context_alignment_native_coordinate_ramps():
    hc, wc = 20, 28
    c = torch.tensor([[5.0, 3.0, 28.0, 20.0]])
    d = torch.tensor([[12.0, 8.0, 14.0, 10.0]])
    xs = 5.0 + (torch.arange(wc).float() + 0.5)
    ys = 3.0 + (torch.arange(hc).float() + 0.5)
    feature = torch.stack((xs[None, :].expand(hc, wc), ys[:, None].expand(hc, wc)))[
        None
    ].requires_grad_()
    out = align_context_features(feature, d, c, (7, 9))
    expected_x = (12.0 + (torch.arange(9).float() + 0.5) * 14.0 / 9).expand(7, 9)
    expected_y = (8.0 + (torch.arange(7).float() + 0.5) * 10.0 / 7)[:, None].expand(
        7, 9
    )
    assert torch.allclose(out[0, 0], expected_x, atol=1e-5, rtol=0)
    assert torch.allclose(out[0, 1], expected_y, atol=1e-5, rtol=0)
    out.square().mean().backward()
    assert feature.grad is not None and torch.isfinite(feature.grad).all()


def test_native_restore_gradient_and_inverse_transform():
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 21, 29, requires_grad=True)
    p = logits.softmax(dim=1)
    expected = restore_native_probs(p, (11, 15))
    actual = restore_native_probs(d4_forward(p, 1, True), (11, 15), k=1, flip=True)
    assert actual.shape == (1, 4, 11, 15)
    assert torch.allclose(expected, actual, atol=1e-7, rtol=0)
    assert torch.allclose(actual.sum(1), torch.ones_like(actual[:, 0]), atol=1e-6)
    (-actual[:, 1].clamp_min(1e-6).log().mean()).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_relative_features_constant_and_finite():
    x = torch.full((1, 1, 19, 23), 0.5)
    y = relative_features(x)
    assert y.shape == (1, 3, 19, 23)
    assert torch.isfinite(y).all()
    assert y[:, 1:].abs().max() < 2e-5
    assert torch.equal(y[:, :1], x)


def test_relative_features_range_and_no_labels():
    torch.manual_seed(1)
    x = torch.rand(1, 1, 37, 39, requires_grad=True)
    y = relative_features(x, sigmas=(1.0, 2.0))
    assert y[:, 1:].min() >= -1 and y[:, 1:].max() <= 1
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    with pytest.raises(ValueError):
        relative_features(torch.full((1, 1, 5, 5), 2.0))


def test_probability_average_k1_and_identical_members():
    rng = np.random.default_rng(0)
    p = rng.random((4, 7, 11))
    p = p / p.sum(axis=0, keepdims=True)
    assert np.allclose(np.stack([p]).mean(axis=0), p, atol=1e-15)
    assert np.allclose(np.stack([p] * 100).mean(axis=0), p, atol=1e-15)
