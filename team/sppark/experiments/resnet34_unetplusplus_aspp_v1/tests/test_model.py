from __future__ import annotations

import pytest
import torch
from model_lab.models import ResNet34UNetPlusPlusASPP, ResidualASPP, build_model


def test_residual_aspp_shape_and_gradient():
    aspp = ResidualASPP(in_channels=512, out_channels=512, width=128, rates=(1, 2, 4, 8))
    x = torch.randn(2, 512, 16, 16, requires_grad=True)
    out = aspp(x)
    assert out.shape == (2, 512, 16, 16)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_resnet34_unetplusplus_aspp_forward_and_backward():
    model = ResNet34UNetPlusPlusASPP(in_channels=3, num_classes=4, pretrained=False, deep_supervision=True)
    x = torch.randn(2, 3, 128, 128, requires_grad=True)
    out = model(x)

    assert "logits" in out
    assert out["logits"].shape == (2, 4, 128, 128)
    assert "aux_logits" in out
    assert len(out["aux_logits"]) == 4

    loss = out["logits"].sum()
    for aux in out["aux_logits"]:
        loss = loss + aux.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_resnet34_parameter_count_and_groups():
    model = ResNet34UNetPlusPlusASPP(in_channels=3, num_classes=4, pretrained=False)
    total_params = sum(p.numel() for p in model.parameters())
    # Should be around 27.48M
    assert 25_000_000 < total_params < 30_000_000

    param_groups = model.get_parameter_groups(encoder_lr=1e-4, decoder_lr=3e-4)
    assert len(param_groups) == 2
    assert param_groups[0]["lr"] == 1e-4
    assert param_groups[1]["lr"] == 3e-4

    all_grouped_params = sum(len(pg["params"]) for pg in param_groups)
    assert all_grouped_params == len(list(model.parameters()))


def test_full_resolution_padded_forward():
    # 909x1278 padded to 928x1280
    model = ResNet34UNetPlusPlusASPP(in_channels=3, num_classes=4, pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 928, 1280)
    with torch.no_grad():
        out = model(x)
        logits = out["logits"]
        cropped = logits[..., :909, :1278]
        assert cropped.shape == (1, 4, 909, 1278)

