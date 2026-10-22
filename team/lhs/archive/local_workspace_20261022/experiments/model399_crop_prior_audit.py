#!/usr/bin/env python3
"""CPU-only, train-mask-only audit of Model392's effective crop class prior.

This estimates source-mask class priors, the 70-epoch native-crop exposure
under the exact Model392 rare-centered sampler, and post-resize/post-augmentation
exposure from one probe epoch. It reads only ``train_foldN/{images,masks}``.
It does not train a model or read OOF/valid/test predictions or labels.

The sampler Monte Carlo uses independent deterministic per-image RNG streams;
it reproduces the locked crop rule and coordinate-cap seed, not the incidental
DataLoader worker scheduling/RNG stream. The post-augmentation arm executes the
actual training Albumentations transform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model336_hrnet_widthnorm_candidate"))

from komap_seg.dataset import (  # noqa: E402
    KomapPatchDataset,
    find_pairs,
    load_class_mask,
    modal_image_width,
    seed_transform_tree,
)
from komap_seg.palette import CLASS_NAMES, NUM_CLASSES  # noqa: E402


GROUP_RE = re.compile(r"^(A\d+)_")
PATCH_SIZE = 512
INPUT_SCALE = 2.0
SAMPLES_PER_IMAGE = 12
POSITIVE_SAMPLE_PROB = 0.5
SEEDS = (1, 2)
ARMS = (("model392_rare_centered", POSITIVE_SAMPLE_PROB), ("uniform_crop", 0.0))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def class_hist(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("expected a nonempty 2-D class mask")
    if int(values.min()) < 0 or int(values.max()) >= NUM_CLASSES:
        raise ValueError("mask contains a class ID outside the fixed 0..3 palette")
    return np.bincount(values.reshape(-1), minlength=NUM_CLASSES).astype(np.int64)


def shares(counts: np.ndarray) -> list[float]:
    values = np.asarray(counts, dtype=np.float64)
    total = float(values.sum())
    if total <= 0 or np.any(values < 0):
        raise ValueError("class counts must be nonnegative with a positive sum")
    return (values / total).tolist()


def rect_sum(integral: np.ndarray, top: int, left: int, size: int) -> int:
    bottom, right = top + size, left + size
    return int(
        integral[bottom, right]
        - integral[top, right]
        - integral[bottom, left]
        + integral[top, left]
    )


def _crop_rng(fold: int, seed: int, image_index: int, arm_index: int, stream: int) -> random.Random:
    value = 399_000_000 + fold * 1_000_000 + seed * 10_000 + image_index * 10 + arm_index + stream * 100_000
    return random.Random(value)


def _native_crop_shares(dataset: KomapPatchDataset, fold: int, seed: int, arm_index: int,
                        epochs: int) -> tuple[list[float], int]:
    totals = np.zeros(NUM_CLASSES, dtype=np.float64)
    draws = epochs * SAMPLES_PER_IMAGE
    for image_index, mask in enumerate(dataset.class_masks):
        height, width = mask.shape
        crop_size = dataset.crop_sizes[image_index]
        integrals = [
            cv2.integral((mask == class_id).astype(np.uint8), sdepth=cv2.CV_32S)
            for class_id in range(NUM_CLASSES)
        ]
        rng = _crop_rng(fold, seed, image_index, arm_index, stream=0)
        for _ in range(draws):
            top, left = dataset._sample_top_left(
                image_index, height, width, rng=rng, crop_size=crop_size
            )
            counts = np.fromiter(
                (rect_sum(integral, top, left, crop_size) for integral in integrals),
                dtype=np.float64,
                count=NUM_CLASSES,
            )
            # A training loss sees a fixed 512x512 patch after nearest resize;
            # average per-crop shares so variable native crop areas do not
            # accidentally reweight images.
            totals += counts / counts.sum()
    return shares(totals), len(dataset.class_masks) * draws


def _resized_augmented_shares(dataset: KomapPatchDataset, fold: int, seed: int,
                              arm_index: int, epochs: int) -> dict[str, Any]:
    resized_totals = np.zeros(NUM_CLASSES, dtype=np.int64)
    augmented_totals = np.zeros(NUM_CLASSES, dtype=np.int64)
    draws = epochs * SAMPLES_PER_IMAGE
    transform_seed = 399_100_000 + fold * 10_000 + seed * 100 + 1
    seed_transform_tree(dataset.transform, transform_seed)

    for image_index, (image, mask) in enumerate(zip(dataset.images, dataset.class_masks)):
        height, width = mask.shape
        crop_size = dataset.crop_sizes[image_index]
        rng = _crop_rng(fold, seed, image_index, arm_index, stream=1)
        for _ in range(draws):
            top, left = dataset._sample_top_left(
                image_index, height, width, rng=rng, crop_size=crop_size
            )
            image_crop = image[top : top + crop_size, left : left + crop_size]
            mask_crop = mask[top : top + crop_size, left : left + crop_size]
            if crop_size != PATCH_SIZE:
                image_crop = cv2.resize(image_crop, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_CUBIC)
                mask_crop = cv2.resize(mask_crop, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_NEAREST)
            resized_totals += class_hist(mask_crop)
            transformed = dataset.transform(image=image_crop, mask=mask_crop)["mask"]
            if torch.is_tensor(transformed):
                transformed = transformed.detach().cpu().numpy()
            augmented_totals += class_hist(np.asarray(transformed))

    return {
        "draws": len(dataset.class_masks) * draws,
        "after_resize_before_augmentation": shares(resized_totals),
        "after_full_training_augmentation": shares(augmented_totals),
        "augmentation_class_pixel_ratio_vs_resize": (
            np.asarray(shares(augmented_totals)) / np.asarray(shares(resized_totals))
        ).tolist(),
    }


def audit_fold(cv_root: Path, fold: int, seeds: tuple[int, ...],
               native_epochs: int, aug_probe_epochs: int) -> dict[str, Any]:
    train_root = cv_root / f"train_fold{fold}"
    images_dir, masks_dir = train_root / "images", train_root / "masks"
    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError(f"expected train-only images/masks under {train_root}")

    target_width = modal_image_width(images_dir)
    pairs = find_pairs(images_dir, masks_dir)
    if not pairs:
        raise FileNotFoundError(f"no training masks found under {masks_dir}")
    mask_paths = [Path(pair["mask"]) for pair in pairs]
    input_hashes = {str(path.relative_to(cv_root)): sha256(path) for path in mask_paths}

    source_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    image_priors: list[list[float]] = []
    stems: list[str] = []
    for pair, path in zip(pairs, mask_paths):
        mask = load_class_mask(path)
        counts = class_hist(mask)
        source_counts += counts
        image_priors.append(shares(counts))
        stems.append(pair["stem"])
    group_ids = []
    for stem in stems:
        match = GROUP_RE.match(stem)
        if not match:
            raise ValueError(f"cannot parse specimen group from training mask stem {stem!r}")
        group_ids.append(match.group(1))
    if np.any(source_counts == 0):
        missing = [CLASS_NAMES[i] for i, count in enumerate(source_counts) if count == 0]
        raise ValueError(f"required classes absent from outer-training fold {fold}: {missing}")

    seed_results: dict[str, Any] = {}
    for seed in seeds:
        arms: dict[str, Any] = {}
        dataset = KomapPatchDataset(
            images_dir,
            masks_dir,
            patch_size=PATCH_SIZE,
            input_scale=INPUT_SCALE,
            target_width=target_width,
            samples_per_image=SAMPLES_PER_IMAGE,
            positive_sample_prob=POSITIVE_SAMPLE_PROB,
            seed=seed,
        )
        for arm_index, (arm_name, positive_probability) in enumerate(ARMS):
            dataset.positive_sample_prob = positive_probability
            native, native_draws = _native_crop_shares(
                dataset, fold, seed, arm_index, native_epochs
            )
            augmented = _resized_augmented_shares(
                dataset, fold, seed, arm_index, aug_probe_epochs
            )
            arms[arm_name] = {
                "positive_sample_probability": positive_probability,
                "native_crop_expected_share": native,
                "native_crop_draws": native_draws,
                **augmented,
            }
        del dataset
        seed_results[str(seed)] = {"arms": arms}

    source_pooled = shares(source_counts)
    source_image_mean = np.mean(np.asarray(image_priors, dtype=np.float64), axis=0).tolist()
    for seed_record in seed_results.values():
        for arm_record in seed_record["arms"].values():
            arm_record["native_crop_over_source_pooled_prior"] = (
                np.asarray(arm_record["native_crop_expected_share"]) / np.asarray(source_pooled)
            ).tolist()
            arm_record["after_resize_over_source_pooled_prior"] = (
                np.asarray(arm_record["after_resize_before_augmentation"]) / np.asarray(source_pooled)
            ).tolist()
            arm_record["after_augmentation_over_source_pooled_prior"] = (
                np.asarray(arm_record["after_full_training_augmentation"]) / np.asarray(source_pooled)
            ).tolist()
            arm_record["after_augmentation_over_source_image_mean_prior"] = (
                np.asarray(arm_record["after_full_training_augmentation"]) / np.asarray(source_image_mean)
            ).tolist()

    return {
        "fold": fold,
        "images": len(mask_paths),
        "specimen_groups": len(set(group_ids)),
        "target_width_from_outer_training_images": target_width,
        "source_mask_prior_pooled_pixels": source_pooled,
        "source_mask_prior_mean_per_image": source_image_mean,
        "source_mask_pixel_counts": source_counts.tolist(),
        "native_crop_epochs_simulated": native_epochs,
        "augmentation_probe_epochs": aug_probe_epochs,
        "seeds": seed_results,
        "training_mask_sha256": input_hashes,
    }


def audit(cv_root: Path, folds: tuple[int, ...], seeds: tuple[int, ...],
          native_epochs: int = 70, aug_probe_epochs: int = 1) -> dict[str, Any]:
    if not folds or any(fold not in (1, 2, 3, 4) for fold in folds) or len(set(folds)) != len(folds):
        raise ValueError("folds must be a non-empty unique subset of 1..4")
    if not seeds or any(seed not in SEEDS for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be a non-empty unique subset of {SEEDS}")
    if native_epochs < 1 or aug_probe_epochs < 1:
        raise ValueError("simulation epoch counts must be positive")
    cv_root = cv_root.resolve()
    results = {
        str(fold): audit_fold(cv_root, fold, seeds, native_epochs, aug_probe_epochs)
        for fold in folds
    }
    return {
        "model": 399,
        "analysis": "Model392 train-only effective class-prior audit",
        "class_order": CLASS_NAMES,
        "training_data_only": True,
        "official_valid_or_test_read": False,
        "oof_data_or_predictions_read": False,
        "gpu_used": False,
        "training_started": False,
        "configuration": {
            "patch_size": PATCH_SIZE,
            "input_scale": INPUT_SCALE,
            "samples_per_image": SAMPLES_PER_IMAGE,
            "positive_sample_probability": POSITIVE_SAMPLE_PROB,
            "minority_classes": [CLASS_NAMES[1], CLASS_NAMES[3]],
            "native_crop_epochs": native_epochs,
            "post_resize_augmentation_probe_epochs": aug_probe_epochs,
            "augmentation": "the exact build_train_transform from model336_hrnet_widthnorm_candidate",
            "crop_rng": "independent deterministic per-image Monte Carlo; locked crop rule and seed-specific coordinate cap",
            "source_prior_views": ["pooled source pixels", "mean per-image source prior"],
        },
        "folds": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--native-epochs", type=int, default=70)
    parser.add_argument("--aug-probe-epochs", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    result = audit(args.cv_root, tuple(args.folds), tuple(args.seeds),
                   args.native_epochs, args.aug_probe_epochs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "model": result["model"],
        "folds": list(result["folds"]),
        "training_data_only": result["training_data_only"],
        "official_valid_or_test_read": result["official_valid_or_test_read"],
        "gpu_used": result["gpu_used"],
        "training_started": result["training_started"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
