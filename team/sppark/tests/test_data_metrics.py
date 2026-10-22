import numpy as np
import pytest
import torch
from PIL import Image

from alphase.data.dataset import TrainDataset, channels
from alphase.data.geometry import crop_array, d4, tile_starts, undo_d4
from alphase.data.palette import PALETTE, decode_mask, save_mask
from alphase.evaluation.metrics import image_miou
from alphase.losses.segmentation import boundary_target, segmentation_loss


def test_palette_roundtrip_and_unknown(tmp_path):
    labels = np.arange(4).reshape(2, 2)
    path = tmp_path/"mask.png"
    save_mask(labels, path)
    np.testing.assert_array_equal(decode_mask(path), labels)
    Image.fromarray(np.zeros((2, 2, 3), np.uint8)).save(path)
    with pytest.raises(ValueError, match="Unknown"):
        decode_mask(path)
    with pytest.raises(ValueError):
        save_mask(np.full((2, 2), 255), path)


@pytest.mark.parametrize("policy,expected", [("exclude", 1), ("one", 1), ("zero", .25)])
def test_empty_class(policy, expected):
    target = np.full((3, 4), 3)
    assert image_miou(target, target, policy)[0] == expected


def test_false_positive_absent_class_and_image_mean():
    target = np.full((2, 2), 3)
    pred = target.copy()
    pred[0, 0] = 0
    assert image_miou(target, pred)[0] == .375
    perfect = np.arange(4).reshape(2, 2)
    wrong = np.zeros((20, 20), dtype=int)
    scores = [image_miou(perfect, perfect)[0], image_miou(wrong, wrong+1)[0]]
    assert np.mean(scores) == .5


def test_ignore_loss_and_boundary():
    torch.manual_seed(5)
    z = torch.randn(1, 4, 5, 7, requires_grad=True)
    t = torch.randint(4, (1, 5, 7))
    t[:, 0] = 255
    weights = dict(ce=1., dice=.5, lovasz=.5, auxiliary=0., boundary=0.)
    loss, _ = segmentation_loss(dict(logits=z), t, weights)
    changed = z.detach().clone()
    changed[:, :, 0] = 1000
    loss2, _ = segmentation_loss(dict(logits=changed), t, weights)
    torch.testing.assert_close(loss, loss2)
    loss.backward()
    assert torch.isfinite(z.grad).all()
    assert (z.grad[:, :, 0] == 0).all()
    t = torch.tensor([[[255, 3, 3], [255, 3, 3]]])
    assert boundary_target(t)[0].sum() == 0


@pytest.mark.parametrize("index", range(8))
def test_d4_rectangular_roundtrip(index):
    x = torch.arange(35).reshape(1, 5, 7)
    assert torch.equal(undo_d4(d4(x, index), index), x)


def test_tiles_cover_every_pixel():
    coverage = np.zeros((909, 1278), int)
    for y in tile_starts(909, 768, 384):
        for x in tile_starts(1278, 768, 384):
            coverage[y:y+768, x:x+768] += 1
    assert coverage.min() > 0


def test_full_image_channels_and_outside_crop():
    gray = np.arange(48, dtype=np.float32).reshape(6, 8)/255
    tensor = channels(gray)
    np.testing.assert_allclose(tensor[2, -1, -1], 1)
    np.testing.assert_array_equal(crop_array(tensor, (1, 2, 5, 6)), tensor[:, 2:6, 1:5])
    target = np.zeros((4, 6), dtype=np.uint8)
    cropped = crop_array(target, (-2, -2, 4, 4), 255)
    assert (cropped[:2] == 255).all()
    assert (cropped[2:, 2:] == 0).all()


def test_local_sampler_deterministic_and_halo(tmp_path):
    from alphase.utils import load_config
    cfg = load_config("configs/cnn_unet.yaml")
    gray = np.repeat(np.arange(64, dtype=np.uint8)[None], 96, 0)
    path, mask = tmp_path/"image.png", tmp_path/"mask.png"
    Image.fromarray(gray).save(path)
    save_mask(np.full(gray.shape, 2), mask)
    row = dict(image_id="tiny", image_path=str(path), mask_path=str(mask))
    dataset = TrainDataset([row], cfg, cfg["train"]["stages"][0])
    a, b = dataset[5], dataset[5]
    assert torch.equal(a["detail"], b["detail"])
    assert a["detail"].shape == (3, 640, 640)
    assert (a["target"][:64] == 255).all()
    assert (a["target"] != 255).any()
