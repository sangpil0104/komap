"""Actual training-loop integration on two synthetic PNGs, never project train data."""

from copy import deepcopy
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from alphase.io import sha256, jsonl_write
from alphase.contracts import ids_to_rgb
from alphase.config import load_config
import alphase.training as training


class TinyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 8, 3, padding=1)
        self.decoder = nn.Conv2d(8, 8, 3, padding=1)
        self.segmentation_head = nn.Conv2d(8, 4, 1)

    def forward(self, x):
        return self.segmentation_head(self.decoder(self.encoder(x).relu()).relu())


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = TinyBase()
        self.residual = None

    def set_base_frozen(self, value):
        for p in self.base.parameters():
            p.requires_grad_(not value)

    def forward(self, detail_rgb, **kwargs):
        return {"logits_detail": self.base(detail_rgb)}


def test_resume_loop_exact(tmp_path, monkeypatch):
    rows = []
    for i in range(2):
        rng = np.random.default_rng(i)
        gray = rng.integers(0, 256, (40, 48), dtype=np.uint8)
        target = rng.integers(0, 4, (40, 48), dtype=np.uint8)
        image = tmp_path / f"{i}.png"
        mask = tmp_path / f"{i}_mask.png"
        Image.fromarray(gray).save(image)
        Image.fromarray(ids_to_rgb(target)).save(mask)
        rows.append(
            {
                "image_id": str(i),
                "split": "train",
                "image_path": str(image),
                "mask_path": str(mask),
                "image_sha256": sha256(image),
                "mask_sha256": sha256(mask),
                "native_hw": [40, 48],
            }
        )
    manifest = tmp_path / "data.jsonl"
    jsonl_write(manifest, rows)
    cfg = load_config("configs/experiments/E02_scratch.yaml")
    cfg["paths"]["dataset_manifest"] = str(manifest)
    cfg["run"].update(name="whole", seed=7, output_root=str(tmp_path / "runs"))
    cfg["geometry"].update(detail_input_size=32, context_input_size=16, target_width=64)
    cfg["training"].update(
        total_optimizer_updates=4,
        checkpoint_every_updates=2,
        validation_enabled=False,
        save_results=False,
        ema_enabled=True,
    )
    monkeypatch.setattr(training, "load_model", lambda *a, **k: TinyModel())
    whole = training.train(cfg, device="cpu")
    expected = torch.load(whole / "last.pt", weights_only=True)
    broken = deepcopy(cfg)
    broken["run"]["name"] = "interrupted"
    save = training.torch_save

    def interrupt(path, state):
        save(path, state)
        if str(path).endswith("last.pt") and state["step"] == 2:
            raise RuntimeError("test interruption")

    monkeypatch.setattr(training, "torch_save", interrupt)
    with pytest.raises(RuntimeError, match="test interruption"):
        training.train(broken, device="cpu")
    monkeypatch.setattr(training, "torch_save", save)
    run = Path(broken["run"]["output_root"]) / "interrupted_seed7"
    training.train(broken, resume=run / "last.pt", device="cpu")
    actual = torch.load(run / "last.pt", weights_only=True)
    assert expected["step"] == actual["step"] == 4
    assert expected["scheduler"] == actual["scheduler"]
    assert expected["sampler"] == actual["sampler"]
    assert expected["samples_seen"] == actual["samples_seen"] == 8
    assert expected["ema_updates"] == actual["ema_updates"] == 4
    for field in ["model", "ema"]:
        for key in expected[field]:
            torch.testing.assert_close(
                expected[field][key], actual[field][key], atol=0, rtol=0
            )
