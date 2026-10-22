"""Synthetic regression suite for the isolated per-item RNG candidate tree.

No KoMaP data, checkpoints, predictions, or scores are read. With the project
environment available, run from the repository root:

    python experiments/rng_per_item_candidate_smoke.py

The final tiny CPU train/resume check may be skipped with --skip-train-smoke.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, get_worker_info


ROOT = Path(os.environ.get("KOMAP_REPO_ROOT", Path(__file__).resolve().parents[1]))
LEGACY_ROOT = ROOT / "model336_hrnet_best"
CANDIDATE_ROOT = ROOT / "model336_hrnet_rng_candidate"
SAMPLES_PER_IMAGE = 16


def import_dataset_module(project_root: Path):
    for name in list(sys.modules):
        if name == "komap_seg" or name.startswith("komap_seg."):
            del sys.modules[name]
    sys.path.insert(0, str(project_root))
    return importlib.import_module("komap_seg.dataset")


legacy_module = import_dataset_module(LEGACY_ROOT)
candidate_module = import_dataset_module(CANDIDATE_ROOT)
legacy_palette = importlib.import_module("komap_seg.palette").PALETTE
candidate_palette = legacy_palette


class LegacyTraceDataset(legacy_module.KomapPatchDataset):
    def _sample_top_left(self, img_idx: int, h: int, w: int) -> tuple[int, int]:
        top_left = super()._sample_top_left(img_idx, h, w)
        self.last_top_left = top_left
        return top_left

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image, mask = super().__getitem__(index)
        top, left = self.last_top_left
        return _record(index, top, left, image, mask)


class CandidateTraceDataset(candidate_module.KomapPatchDataset):
    def _sample_top_left(
        self, img_idx: int, h: int, w: int, rng=None
    ) -> tuple[int, int]:
        top_left = super()._sample_top_left(img_idx, h, w, rng=rng)
        self.last_top_left = top_left
        return top_left

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image, mask = super().__getitem__(index)
        top, left = self.last_top_left
        worker = get_worker_info()
        record = _record(index, top, left, image, mask)
        record["worker"] = torch.tensor(-1 if worker is None else worker.id)
        return record


def _record(index, top, left, image, mask):
    return {
        "index": torch.tensor(index),
        "top": torch.tensor(top),
        "left": torch.tensor(left),
        "image": image,
        "mask": mask,
    }


def write_synthetic_pairs(root: Path) -> None:
    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    height, width = 96, 112
    yy, xx = np.mgrid[:height, :width]
    image = np.stack(
        [(xx * 7 + yy * 3) % 256, (xx * 2 + yy * 11) % 256, (xx * 13 + yy * 5) % 256],
        axis=-1,
    ).astype(np.uint8)
    masks = [
        ((xx // 12 + yy // 12) % len(candidate_palette)).astype(np.uint8),
        (((xx // 12 + yy // 12) % 2) * 2).astype(np.uint8),
    ]
    for idx, labels in enumerate(masks):
        Image.fromarray(image).save(images_dir / f"sample{idx}_image.png")
        Image.fromarray(candidate_palette[labels]).save(masks_dir / f"sample{idx}_mask.png")


def make_legacy_dataset(module, dataset_cls, root: Path):
    transform = module.build_train_transform()
    transform.set_random_seed(731)
    return dataset_cls(
        root / "images",
        root / "masks",
        patch_size=32,
        samples_per_image=SAMPLES_PER_IMAGE,
        positive_sample_prob=0.5,
        input_scale=1.0,
        transform=transform,
        seed=382,
    )


def assert_record_equal(left, right, fields=("index", "top", "left", "image", "mask")):
    for field in fields:
        if not torch.equal(left[field], right[field]):
            raise AssertionError(f"mismatch in field {field}")


def legacy_bitwise_regression(root: Path) -> None:
    old_ds = make_legacy_dataset(legacy_module, LegacyTraceDataset, root)
    candidate_ds = make_legacy_dataset(candidate_module, CandidateTraceDataset, root)
    assert len(old_ds) == len(candidate_ds)
    for index in range(len(old_ds)):
        assert_record_equal(old_ds[index], candidate_ds[index])


def make_per_item_dataset(root: Path, epoch: int):
    dataset = CandidateTraceDataset(
        root / "images",
        root / "masks",
        patch_size=32,
        samples_per_image=SAMPLES_PER_IMAGE,
        positive_sample_prob=0.5,
        input_scale=1.0,
        transform=candidate_module.build_train_transform(),
        seed=382,
        rng_mode="per_item",
    )
    dataset.set_epoch(epoch)
    return dataset


def collect_by_index(root: Path, epoch: int, workers: int, batch_size: int, order=None):
    dataset = make_per_item_dataset(root, epoch)
    if order is None:
        order = list(range(len(dataset)))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=order,
        num_workers=workers,
        worker_init_fn=candidate_module.seed_worker if workers else None,
        drop_last=False,
        persistent_workers=False,
    )
    outputs = {}
    worker_ids = set()
    for batch in loader:
        batch_count = int(batch["index"].shape[0])
        for row in range(batch_count):
            index = int(batch["index"][row])
            worker_ids.add(int(batch["worker"][row]) if "worker" in batch else -1)
            outputs[index] = {key: value[row].clone() for key, value in batch.items() if key != "worker"}
    return outputs, worker_ids


def output_maps_equal(left, right) -> bool:
    return left.keys() == right.keys() and all(
        all(torch.equal(left[index][field], right[index][field]) for field in left[index])
        for index in left
    )


def assert_independent_transform_gates() -> dict[tuple[bool, bool], int]:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    counts: dict[tuple[bool, bool], int] = {}
    for sample_seed in range(512):
        transform = A.ReplayCompose(
            [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]
        )
        candidate_module.seed_transform_tree(transform, sample_seed)
        result = transform(image=image)
        applied = tuple(item["applied"] for item in result["replay"]["transforms"])
        counts[applied] = counts.get(applied, 0) + 1
    expected = {(False, False), (False, True), (True, False), (True, True)}
    if set(counts) != expected or min(counts.values()) < 60:
        raise AssertionError(f"flip gates are not behaving as separate streams: {counts}")
    return counts


def tiny_cpu_train_resume(root: Path) -> None:
    data_root = root / "tiny_train_data"
    train_images = data_root / "train" / "images"
    train_masks = data_root / "train" / "masks"
    train_images.mkdir(parents=True)
    train_masks.mkdir(parents=True)
    source_images = sorted((root / "images").glob("*_image.png"))
    source_masks = sorted((root / "masks").glob("*_mask.png"))
    for image_path, mask_path in zip(source_images, source_masks):
        Image.open(image_path).save(train_images / image_path.name)
        Image.open(mask_path).save(train_masks / mask_path.name)

    output_dir = root / "tiny_train_output"
    common = [
        sys.executable,
        str(CANDIDATE_ROOT / "train.py"),
        "--data-root",
        str(data_root),
        "--output-dir",
        str(output_dir),
        "--device",
        "cpu",
        "--architecture",
        "Unet",
        "--encoder-name",
        "resnet18",
        "--encoder-weights",
        "none",
        "--patch-size",
        "32",
        "--input-scale",
        "1",
        "--samples-per-image",
        "2",
        "--batch-size",
        "2",
        "--accumulation-steps",
        "1",
        "--num-workers",
        "2",
        "--lr",
        "0.001",
        "--seed",
        "382",
        "--rng-mode",
        "per_item",
        "--no-boundary-head",
        "--val-every",
        "0",
    ]
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(CANDIDATE_ROOT),
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    first = subprocess.run(
        [*common, "--epochs", "1"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    resumed = subprocess.run(
        [*common, "--epochs", "1", "--resume"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    if "done. final model:" not in first.stdout:
        raise AssertionError("tiny CPU training did not finish its first epoch")
    if "resumed from epoch 1; next epoch=2" not in resumed.stdout:
        raise AssertionError("tiny CPU trainer did not restore the next epoch index")

    rng_flag = common.index("--rng-mode")
    wrong_mode_command = [
        *common[:rng_flag],
        *common[rng_flag + 2 :],
        "--epochs",
        "1",
        "--resume",
    ]
    wrong_mode = subprocess.run(
        wrong_mode_command,
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    if wrong_mode.returncode == 0 or "resume arguments differ" not in wrong_mode.stderr:
        raise AssertionError("resume with a silently changed RNG mode was not rejected")
    recorded_args = json.loads((output_dir / "args.json").read_text())
    if recorded_args.get("rng_mode") != "per_item":
        raise AssertionError("failed resume overwrote args.json before rejecting the mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train-smoke", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="komap-rng-candidate-") as tmp:
        root = Path(tmp)
        write_synthetic_pairs(root)

        legacy_bitwise_regression(root)
        print("PASS: candidate default/explicit legacy mode is bitwise identical to the original dataset pipeline.")

        reference, _ = collect_by_index(root, epoch=3, workers=0, batch_size=1)
        sample_count = len(reference)
        configs = [
            (2, 2, list(range(sample_count))),
            (4, 4, list(range(sample_count))),
            (2, 3, list(reversed(range(sample_count)))),
        ]
        for workers, batch_size, order in configs:
            actual, worker_ids = collect_by_index(root, 3, workers, batch_size, order)
            if not output_maps_equal(reference, actual):
                raise AssertionError(
                    f"per-item stream changed with workers={workers}, batch={batch_size}, order={order}"
                )
            if workers > 1 and len(worker_ids) < 2:
                raise AssertionError(f"expected multiple workers, observed {worker_ids}")
        print("PASS: per-item crops and full augmentation outputs are invariant to worker count, batch size, and sample order.")

        next_epoch, _ = collect_by_index(root, epoch=4, workers=2, batch_size=2)
        if output_maps_equal(reference, next_epoch):
            raise AssertionError("per-item streams did not change between epochs")
        replay, _ = collect_by_index(root, epoch=3, workers=4, batch_size=2)
        if not output_maps_equal(reference, replay):
            raise AssertionError("same seed/epoch failed to replay per-item outputs")
        print("PASS: epoch changes produce new streams; restoring the same seed/epoch reproduces them.")

        gate_counts = assert_independent_transform_gates()
        print(f"PASS: independent H/V flip gates cover all four outcomes: {gate_counts}")

        full_transform = candidate_module.build_train_transform()
        candidate_module.seed_transform_tree(full_transform, 731)
        nodes = [full_transform]
        node_seeds = []
        node_names = []
        while nodes:
            node = nodes.pop(0)
            if hasattr(node, "seed"):
                node_seeds.append(node.seed)
                node_names.append(type(node).__name__)
            nodes.extend(getattr(node, "transforms", ()))
        if len(node_seeds) != len(set(node_seeds)):
            raise AssertionError("full augmentation tree contains duplicate node seeds")
        if "ElasticTransform" not in node_names or "GridDistortion" not in node_names:
            raise AssertionError("OneOf child transforms were absent from the seeded tree")
        print(f"PASS: every full augmentation-tree node, including OneOf children, has a distinct seed ({len(node_seeds)} nodes).")

        if not args.skip_train_smoke:
            tiny_cpu_train_resume(root)
            print("PASS: real train.py completes a synthetic CPU epoch, restores the next epoch index, and rejects RNG-mode mismatch on resume.")

    print(f"albumentations={A.__version__} torch={torch.__version__}")
    print("Scope: temporary synthetic PNGs/checkpoints only; no KoMaP data or score artifacts accessed.")
    print("This verifies data-stream correctness, not mIoU benefit.")


if __name__ == "__main__":
    main()
