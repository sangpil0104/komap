"""Synthetic-only worker RNG test using the real KoMaP patch dataset pipeline.

Run from the repository root with the project's Python environment. This test
creates temporary PNGs and never opens KoMaP data, checkpoints, predictions,
or score files. It characterizes the current worker initializer against a
candidate that independently seeds crop and Albumentations RNG streams.
"""

from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, get_worker_info


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model336_hrnet_best"))
from komap_seg.dataset import (  # noqa: E402
    KomapPatchDataset,
    build_train_transform,
    seed_worker,
)
from komap_seg.palette import PALETTE  # noqa: E402


def derive_seed(base_seed: int, stream_id: int) -> int:
    """Derive independent deterministic seeds for separate worker RNG streams."""
    return int(
        np.random.SeedSequence([base_seed, stream_id]).generate_state(
            1, dtype=np.uint32
        )[0]
    )


class TracedKomapPatchDataset(KomapPatchDataset):
    """Real dataset sampling/augmentation with crop coordinates exposed."""

    def _sample_top_left(self, img_idx: int, h: int, w: int) -> tuple[int, int]:
        top_left = super()._sample_top_left(img_idx, h, w)
        self.last_top_left = top_left
        return top_left

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image, mask = super().__getitem__(index)
        worker = get_worker_info()
        if worker is None:
            raise RuntimeError("this smoke test requires DataLoader workers")
        top, left = self.last_top_left
        return {
            "index": torch.tensor(index),
            "worker": torch.tensor(worker.id),
            "top": torch.tensor(top),
            "left": torch.tensor(left),
            "image": image,
            "mask": mask,
        }


def corrected_seed_worker(worker_id: int) -> None:
    """Candidate initializer with separate private crop and transform streams."""
    seed_worker(worker_id)
    info = get_worker_info()
    if info is None:
        raise RuntimeError("worker_init_fn must execute inside a DataLoader worker")
    worker_seed = info.seed % (2**32)
    dataset = info.dataset
    if hasattr(dataset, "rng"):
        dataset.rng.seed(derive_seed(worker_seed, stream_id=0))
    transform = getattr(dataset, "transform", None)
    if hasattr(transform, "set_random_seed"):
        transform.set_random_seed(derive_seed(worker_seed, stream_id=1))


def write_synthetic_pair(root: Path) -> None:
    """Create one textured RGB image and a four-class palette mask."""
    image_dir = root / "images"
    mask_dir = root / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    height, width = 96, 112
    yy, xx = np.mgrid[:height, :width]
    image = np.stack(
        [
            (xx * 7 + yy * 3) % 256,
            (xx * 2 + yy * 11) % 256,
            (xx * 13 + yy * 5) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)
    class_mask = ((xx // 12 + yy // 12) % len(PALETTE)).astype(np.uint8)
    mask_rgb = PALETTE[class_mask]

    Image.fromarray(image).save(image_dir / "synthetic_image.png")
    Image.fromarray(mask_rgb).save(mask_dir / "synthetic_mask.png")


def make_loader(root: Path, worker_init_fn, *, shuffle: bool = False) -> DataLoader:
    dataset = TracedKomapPatchDataset(
        images_dir=root / "images",
        masks_dir=root / "masks",
        patch_size=32,
        samples_per_image=16,
        positive_sample_prob=0.5,
        input_scale=1.0,
        transform=build_train_transform(),
        seed=382,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=4,
        worker_init_fn=worker_init_fn,
        drop_last=False,
        persistent_workers=False,
    )


def first_item_per_worker(loader: DataLoader) -> list[dict[str, torch.Tensor]]:
    rows: dict[int, dict[str, torch.Tensor]] = {}
    for batch in loader:
        worker_id = int(batch["worker"].item())
        rows.setdefault(worker_id, {key: value.clone() for key, value in batch.items()})
        if len(rows) == 4:
            break
    if set(rows) != {0, 1, 2, 3}:
        raise AssertionError(f"did not observe all four workers: {sorted(rows)}")
    return [rows[index] for index in range(4)]


def items_equal(left, right) -> bool:
    return len(left) == len(right) and all(
        left_row.keys() == right_row.keys()
        and all(torch.equal(left_row[key], right_row[key]) for key in left_row)
        for left_row, right_row in zip(left, right)
    )


def loader_outputs_equal(left, right) -> bool:
    return len(left) == len(right) and all(
        batch.keys() == other.keys()
        and all(torch.equal(batch[key], other[key]) for key in batch)
        for batch, other in zip(left, right)
    )


def unseeded_compose_draws_after_global_seed_reset():
    random.seed(382)
    np.random.seed(382)
    torch.manual_seed(382)
    transform = A.Compose([A.HorizontalFlip(p=0.5)])
    draws = (
        transform.py_random.random(),
        float(transform.random_generator.random()),
        transform.transforms[0].py_random.random(),
    )
    return transform.seed, draws


def main() -> None:
    compose1 = unseeded_compose_draws_after_global_seed_reset()
    compose2 = unseeded_compose_draws_after_global_seed_reset()
    assert compose1[0] is None and compose2[0] is None
    assert compose1[1] != compose2[1]

    with tempfile.TemporaryDirectory(prefix="komap-rng-smoke-") as temp_dir:
        root = Path(temp_dir)
        write_synthetic_pair(root)

        # Current initializer leaves the dataset's private crop RNG and
        # Albumentations Compose RNG untouched after fork.
        baseline_loader = make_loader(root, seed_worker)
        baseline_epoch1 = first_item_per_worker(baseline_loader)
        baseline_epoch2 = first_item_per_worker(baseline_loader)
        assert len({(int(row["top"]), int(row["left"])) for row in baseline_epoch1}) == 1
        assert items_equal(baseline_epoch1, baseline_epoch2)
        assert all(
            torch.equal(baseline_epoch1[0][key], baseline_epoch1[i][key])
            for i in range(1, 4)
            for key in ("top", "left", "image", "mask")
        )

        # Candidate initializer gives each worker/epoch distinct RNG streams.
        corrected_loader = make_loader(root, corrected_seed_worker)
        corrected_epoch1 = first_item_per_worker(corrected_loader)
        corrected_epoch2 = first_item_per_worker(corrected_loader)
        assert len(
            {(int(row["top"]), int(row["left"])) for row in corrected_epoch1}
        ) > 1
        assert not items_equal(corrected_epoch1, corrected_epoch2)

        # Synthetic epoch-boundary replay using main CPU RNG state only.
        torch.manual_seed(382)
        resume_loader = make_loader(root, corrected_seed_worker, shuffle=True)
        _ = list(resume_loader)
        checkpoint_rng = torch.get_rng_state().clone()
        continued = list(resume_loader)
        torch.set_rng_state(checkpoint_rng)
        resumed = list(make_loader(root, corrected_seed_worker, shuffle=True))
        assert loader_outputs_equal(continued, resumed)

    print(f"albumentations={A.__version__} torch={torch.__version__}")
    print("PASS: global Python/NumPy/Torch seed resets do not seed Compose(seed=None).")
    print("PASS: actual KomapPatchDataset repeats initial crop/augmentation across forked workers and epochs with current init.")
    print("PASS: distinct derived crop/transform seeds separate worker streams and change them across epochs.")
    print("PASS: restoring main CPU torch RNG replays the next synthetic shuffled epoch.")
    print("Scope: synthetic PNGs only; no KoMaP data, checkpoints, predictions, or scores accessed.")
    print("Caveat: this is not an end-to-end train.py/CUDA/resume or performance test.")


if __name__ == "__main__":
    main()
