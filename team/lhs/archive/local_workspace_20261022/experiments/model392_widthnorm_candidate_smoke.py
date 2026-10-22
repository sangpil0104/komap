"""Synthetic-only regression smoke for the isolated Model392 candidate."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BASE_ROOT = ROOT / "model336_hrnet_rng_candidate"
CANDIDATE_ROOT = ROOT / "model336_hrnet_widthnorm_candidate"


def import_dataset(root: Path):
    for name in list(sys.modules):
        if name == "komap_seg" or name.startswith("komap_seg."):
            del sys.modules[name]
    sys.path.insert(0, str(root))
    return importlib.import_module("komap_seg.dataset")


def write_synthetic_pairs(root: Path) -> tuple[Path, Path]:
    images = root / "images"
    masks = root / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    base_module = import_dataset(BASE_ROOT)
    palette = importlib.import_module("komap_seg.palette").PALETTE
    for index, width in enumerate((40, 80, 80), start=1):
        yy, xx = np.mgrid[:48, :width]
        image = np.stack(((xx * 5) % 256, (yy * 7) % 256, (xx + yy * 3) % 256), axis=-1)
        labels = ((xx // 8 + yy // 8) % 4).astype(np.uint8)
        name = f"A{index}_step1_sample1"
        Image.fromarray(image.astype(np.uint8)).save(images / f"{name}_image.png")
        Image.fromarray(palette[labels]).save(masks / f"{name}_mask.png")
    return images, masks


def legacy_compatibility(images: Path, masks: Path) -> None:
    base = import_dataset(BASE_ROOT)
    candidate = import_dataset(CANDIDATE_ROOT)

    def make(module, cls):
        transform = module.build_train_transform()
        transform.set_random_seed(731)
        return cls(
            images,
            masks,
            patch_size=32,
            samples_per_image=2,
            positive_sample_prob=0.0,
            input_scale=2.0,
            transform=transform,
            seed=382,
            rng_mode="legacy",
        )

    original = make(base, base.KomapPatchDataset)
    modified = make(candidate, candidate.KomapPatchDataset)
    assert original.crop_size == modified.crop_size == 16
    assert modified.crop_sizes == [16, 16, 16]
    for index in range(len(original)):
        old_image, old_mask = original[index]
        new_image, new_mask = modified[index]
        assert torch.equal(old_image, new_image)
        assert torch.equal(old_mask, new_mask)


def width_normalization_geometry(images: Path, masks: Path) -> None:
    module = import_dataset(CANDIDATE_ROOT)
    target_width = module.modal_image_width(images)
    assert target_width == 80
    dataset = module.KomapPatchDataset(
        images,
        masks,
        patch_size=32,
        samples_per_image=1,
        positive_sample_prob=0.0,
        input_scale=2.0,
        target_width=target_width,
        transform=module.build_train_transform(),
        seed=391,
        rng_mode="legacy",
    )
    assert dataset.crop_sizes == [8, 16, 16]
    normalized_coverage = [crop / width for crop, width in zip(dataset.crop_sizes, (40, 80, 80))]
    assert np.allclose(normalized_coverage, [0.2, 0.2, 0.2])
    for index in range(len(dataset)):
        image, mask = dataset[index]
        assert image.shape == (3, 32, 32)
        assert mask.shape == (32, 32)
        assert int(mask.min()) >= 0 and int(mask.max()) < 4

    scales = [module.effective_input_scale(2.0, target_width, width) for width in (40, 80, 80)]
    overlaps = [module.scaled_overlap(32, 2.0, 10, scale) for scale in scales]
    tiles = [round(32 / scale) for scale in scales]
    assert tiles == [8, 16, 16]
    assert overlaps == [5, 10, 10]
    assert all(np.isclose(overlap / tile, 10 / 16, atol=0.07)
               for overlap, tile in zip(overlaps, tiles))

    inference = importlib.import_module("komap_seg.inference")

    class UniformModel(torch.nn.Module):
        def forward(self, batch):
            return torch.zeros((batch.shape[0], 4, batch.shape[2], batch.shape[3]))

    for width, scale, overlap in zip((40, 80), scales[:2], overlaps[:2]):
        probabilities = inference.predict_probs_tta(
            UniformModel(),
            torch.zeros((3, 48, width)),
            patch_size=32,
            overlap=overlap,
            batch_size=2,
            input_scale=scale,
            device=torch.device("cpu"),
        )
        assert probabilities.shape == (4, 48, width)
        assert torch.allclose(
            probabilities.sum(dim=0), torch.ones((48, width)), atol=1e-6
        )


def preprocessing_metadata_and_validation_guard(root: Path) -> None:
    module = import_dataset(CANDIDATE_ROOT)
    checkpoint_module = importlib.import_module("komap_seg.checkpoint")
    saved = {
        "patch_size": 512,
        "input_scale": 2.0,
        "target_width": 1278,
        "normalize_width": True,
    }
    assert checkpoint_module.resolve_preprocessing(saved) == saved
    assert checkpoint_module.resolve_preprocessing(saved, target_width=1278) == saved
    try:
        checkpoint_module.resolve_preprocessing(saved, target_width=0)
    except ValueError as error:
        assert "differs from checkpoint preprocessing" in str(error)
    else:
        raise AssertionError("mismatched target width was silently accepted")

    checkpoint_path = root / "synthetic_checkpoint.pt"
    checkpoint_module.save_checkpoint(
        checkpoint_path, torch.nn.Linear(2, 2), {"architecture": "synthetic"}, saved
    )
    stored = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert stored["preprocessing_config"] == saved

    command = [
        sys.executable,
        str(CANDIDATE_ROOT / "train.py"),
        "--data-root", str(root / "does_not_exist"),
        "--output-dir", str(root / "must_not_be_created"),
        "--device", "cpu",
        "--normalize-width",
        "--val-every", "1",
    ]
    environment = os.environ.copy()
    environment.update({"PYTHONPATH": str(CANDIDATE_ROOT), "CUDA_VISIBLE_DEVICES": ""})
    process = subprocess.run(command, env=environment, capture_output=True, text=True, check=False)
    assert process.returncode != 0
    assert "width-normalization experiments require --val-every 0" in process.stderr
    assert not (root / "must_not_be_created").exists()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="model392-widthnorm-smoke-") as temp:
        images, masks = write_synthetic_pairs(Path(temp))
        legacy_compatibility(images, masks)
        width_normalization_geometry(images, masks)
        preprocessing_metadata_and_validation_guard(Path(temp))
    print("PASS: fixed-scale legacy candidate is bitwise identical to its P100-base dataset on synthetic crops.")
    print("PASS: training crop fractions, inference scales, overlap ratios, and native probability-map shapes agree.")
    print("PASS: checkpoint preprocessing metadata, mismatch rejection, and width-mode valid-access guard.")
    print("Scope: temporary synthetic images/masks only; no KoMaP data, scores, or GPU used.")


if __name__ == "__main__":
    main()
