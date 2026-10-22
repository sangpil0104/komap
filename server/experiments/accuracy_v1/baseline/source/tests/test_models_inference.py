from copy import deepcopy
import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from alphase.inference.predictor import predict_probabilities
from alphase.models.cnn import PhaseCNN
from alphase.models.hybrid import PhaseHybrid, sample_context
from alphase.utils import load_config

torch.set_num_threads(2)


class FakeBackbone(nn.Module):
    embed_dim, patch_size = 8, 16
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Conv2d(3, 8, 1)] + [nn.Conv2d(8, 8, 1) for _ in range(3)])
        self.norm = nn.Identity()
    def get_intermediate_layers(self, x, **kwargs):
        x = F.avg_pool2d(x, 16)
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features


def test_roi_full_resize_and_ramp():
    feature = torch.randn(1, 4, 6, 8)
    actual = sample_context(feature, torch.tensor([[0, 0, 128, 96]]), (96, 128), (12, 16))
    torch.testing.assert_close(actual, F.interpolate(feature, (12, 16), mode="bilinear", align_corners=False))
    x = (torch.arange(8)+.5)*16
    ramp = x[None, None, None].expand(1, 1, 6, 8).float()
    local = sample_context(ramp, torch.tensor([[16, 16, 80, 80]]), (96, 128), (4, 4))
    torch.testing.assert_close(local[0, 0, 0], torch.tensor([24., 40., 56., 72.]))


def test_hybrid_gradients_and_local_shape():
    model = PhaseHybrid(FakeBackbone(), [0, 1, 2, 3], context_channels=8)
    detail, context = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 64, 96)
    box = torch.tensor([[16., 16., 48., 48.]])
    output = model(detail, context, box)["logits"]
    assert output.shape == (1, 4, 32, 32)
    output.mean().backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert model.projections[0][0].weight.grad is not None
    model.zero_grad(set_to_none=True)
    model.set_backbone_trainable(1)
    model.train()
    assert not model.backbone.training
    model(detail, context, box)["logits"].square().mean().backward()
    assert model.backbone.blocks[-1].weight.grad is not None
    assert torch.isfinite(model.backbone.blocks[-1].weight.grad).all()


def test_cnn_checkpoint_roundtrip(tmp_path):
    model = PhaseCNN().eval()
    x = torch.rand(1, 3, 64, 96)
    with torch.no_grad():
        expected = model(x)["logits"]
    torch.save(model.state_dict(), tmp_path/"state.pt")
    restored = PhaseCNN().eval()
    restored.load_state_dict(torch.load(tmp_path/"state.pt", weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(restored(x)["logits"], expected, rtol=0, atol=0)


class ConstantModel(nn.Module):
    def forward(self, detail, *args):
        z = torch.tensor([0., 1., 2., 3.], device=detail.device)[None, :, None, None]
        return {"logits": z.expand(1, 4, *detail.shape[-2:])}


@pytest.mark.parametrize("mode", ["full_frame", "tiled"])
def test_constant_tta_and_tiles(mode):
    cfg = load_config("configs/cnn_unet.yaml")
    cfg["train"]["precision"] = "fp32"
    cfg["inference"].update(mode=mode, local_core_size=64, halo=16, d4=True, scales=[1., 1.25])
    gray = np.random.default_rng(3).random((75, 101)).astype(np.float32)
    result = predict_probabilities(ConstantModel(), gray, cfg, torch.device("cpu"))
    expected = torch.arange(4).float().softmax(0).numpy()[:, None, None]
    np.testing.assert_allclose(result, np.broadcast_to(expected, result.shape), atol=1e-6)


def test_missing_hybrid_weight_fails():
    from alphase.models.factory import build_model
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        build_model(load_config("configs/hybrid_hplus.yaml"))
