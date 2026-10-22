#!/usr/bin/env python3
"""CPU-only realized-dose simulation for the locked Model395 sampler candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model336_hrnet_component_candidate"))

from komap_seg.dataset import KomapPatchDataset, find_pairs, seed_worker  # noqa: E402


CLASSES = ((1, "Al3Ni"), (3, "Eutectic_Si"))
AREA_BINS = (
    ("1_to_lt64", 1, 64),
    ("64_to_lt256", 64, 256),
    ("256_to_lt1024", 256, 1024),
    ("1024_to_lt4096", 1024, 4096),
    ("4096_or_more", 4096, None),
)
GROUP_RE = re.compile(r"^(A\d+)_")
WORKERS = 4
BATCH_SIZE = 4
EPOCHS = 70
SEED = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _area_bin_index(area: int) -> int:
    for index, (_, lower, upper) in enumerate(AREA_BINS):
        if area >= lower and (upper is None or area < upper):
            return index
    return -1


def _rect_sum(integral: np.ndarray, top: int, left: int, size: int) -> int:
    bottom = top + size
    right = left + size
    return int(
        integral[bottom, right]
        - integral[top, right]
        - integral[bottom, left]
        + integral[top, left]
    )


class SamplerProbeDataset(KomapPatchDataset):
    """Uses the real sampler and DataLoader worker behavior, but no image transforms."""

    def __init__(self, images_dir: Path, masks_dir: Path, sampler: str, seed: int):
        super().__init__(
            images_dir,
            masks_dir,
            patch_size=512,
            input_scale=2.0,
            samples_per_image=12,
            seed=seed,
            minority_sampler=sampler,
            transform=lambda **kwargs: kwargs,
        )
        self.audit_labels: list[dict[int, np.ndarray]] = []
        self.audit_areas: list[dict[int, np.ndarray]] = []
        self.audit_integrals: list[dict[int, list[np.ndarray]]] = []
        for mask in self.class_masks:
            label_by_class: dict[int, np.ndarray] = {}
            area_by_class: dict[int, np.ndarray] = {}
            integrals_by_class: dict[int, list[np.ndarray]] = {}
            for class_id, _ in CLASSES:
                component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
                    (mask == class_id).astype(np.uint8),
                    connectivity=8,
                    ltype=cv2.CV_32S,
                )
                areas = stats[:, cv2.CC_STAT_AREA].astype(np.int32)
                label_by_class[class_id] = labels
                area_by_class[class_id] = areas
                bins: list[np.ndarray] = []
                for _, lower, upper in AREA_BINS:
                    keep = np.zeros(component_count, dtype=bool)
                    keep[1:] = (areas[1:] >= lower) & (
                        True if upper is None else areas[1:] < upper
                    )
                    binary = keep[labels].astype(np.uint8, copy=False)
                    bins.append(cv2.integral(binary, sdepth=cv2.CV_32S))
                integrals_by_class[class_id] = bins
            self.audit_labels.append(label_by_class)
            self.audit_areas.append(area_by_class)
            self.audit_integrals.append(integrals_by_class)

    def __getitem__(self, index: int) -> torch.Tensor:
        image_index = index % len(self.images)
        mask = self.class_masks[image_index]
        height, width = mask.shape
        top, left, details = self._sample_top_left(
            image_index, height, width, return_details=True
        )
        anchor_class = int(details["class_id"]) if details["positive"] else -1
        anchor_bin = -1
        if anchor_class >= 0:
            y, x = int(details["anchor_y"]), int(details["anchor_x"])
            component_id = int(self.audit_labels[image_index][anchor_class][y, x])
            area = int(self.audit_areas[image_index][anchor_class][component_id])
            anchor_bin = _area_bin_index(area)
            if anchor_bin < 0:
                raise ValueError(f"sampled minority anchor has unexpected component area: {area}")

        crop_pixels: list[int] = []
        for class_id, _ in CLASSES:
            for integral in self.audit_integrals[image_index][class_id]:
                crop_pixels.append(_rect_sum(integral, top, left, self.crop_size))
        return torch.tensor(
            [image_index, anchor_class, anchor_bin, *crop_pixels], dtype=torch.int64
        )


def _summarize_arm(
    dataset: SamplerProbeDataset,
    data_loader: DataLoader,
    epochs: int,
) -> dict[str, Any]:
    anchor_counts = np.zeros((len(CLASSES), len(AREA_BINS)), dtype=np.int64)
    crop_pixels = np.zeros((len(CLASSES), len(AREA_BINS)), dtype=np.int64)
    group_anchor: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros((len(CLASSES), len(AREA_BINS)), dtype=np.int64)
    )
    group_crop_pixels: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros((len(CLASSES), len(AREA_BINS)), dtype=np.int64)
    )
    image_group: dict[int, str] = {}
    for image_index, stem in enumerate(dataset.stems):
        match = GROUP_RE.match(stem)
        if not match:
            raise ValueError(f"cannot parse specimen group from {stem}")
        image_group[image_index] = match.group(1)
    base_dataset_rng_state = dataset.rng.getstate()

    # Each iterator respawns workers (persistent_workers=False). Their private
    # dataset.rng object is fork-copied and seed_worker does not reseed it.
    for _epoch in range(epochs):
        for batch in data_loader:
            for row in batch.tolist():
                image_index, anchor_class, anchor_bin = map(int, row[:3])
                group = image_group[image_index]
                if anchor_class >= 0:
                    class_index = next(i for i, (value, _) in enumerate(CLASSES) if value == anchor_class)
                    anchor_counts[class_index, anchor_bin] += 1
                    group_anchor[group][class_index, anchor_bin] += 1
                crop = np.asarray(row[3:], dtype=np.int64).reshape(len(CLASSES), len(AREA_BINS))
                crop_pixels += crop
                group_crop_pixels[group] += crop

    if dataset.rng.getstate() != base_dataset_rng_state:
        raise RuntimeError("parent dataset RNG unexpectedly changed during worker simulation")

    group_summary: dict[str, Any] = {}
    for class_index, (_, class_name) in enumerate(CLASSES):
        group_summary[class_name] = {}
        for bin_index, (bin_name, _, _) in enumerate(AREA_BINS):
            anchor_values = {group: int(values[class_index, bin_index]) for group, values in group_anchor.items()}
            crop_values = {group: int(values[class_index, bin_index]) for group, values in group_crop_pixels.items()}
            total_anchors = sum(anchor_values.values())
            total_crop_pixels = sum(crop_values.values())
            group_summary[class_name][bin_name] = {
                "groups_with_positive_anchors": sum(value > 0 for value in anchor_values.values()),
                "max_group_share_of_positive_anchors": (
                    max(anchor_values.values()) / total_anchors if total_anchors else None
                ),
                "groups_with_crop_pixel_exposure": sum(value > 0 for value in crop_values.values()),
                "max_group_share_of_crop_pixel_exposure": (
                    max(crop_values.values()) / total_crop_pixels if total_crop_pixels else None
                ),
            }
    return {
        "images": len(dataset.images),
        "groups": len(set(image_group.values())),
        "epochs": epochs,
        "samples_per_image": dataset.samples_per_image,
        "workers": WORKERS,
        "batch_size": BATCH_SIZE,
        "persistent_workers": False,
        "drop_last": True,
        "worker_private_rng_reseeded_by_train_worker_init": False,
        "anchor_counts": {
            class_name: {AREA_BINS[b][0]: int(anchor_counts[c, b]) for b in range(len(AREA_BINS))}
            for c, (_, class_name) in enumerate(CLASSES)
        },
        "crop_gt_pixel_exposure": {
            class_name: {AREA_BINS[b][0]: int(crop_pixels[c, b]) for b in range(len(AREA_BINS))}
            for c, (_, class_name) in enumerate(CLASSES)
        },
        "group_support_and_concentration": group_summary,
    }


def simulate_fold(cv_root: Path, fold: int, epochs: int = EPOCHS) -> dict[str, Any]:
    train_root = cv_root / f"train_fold{fold}"
    images_dir, masks_dir = train_root / "images", train_root / "masks"
    pairs = find_pairs(images_dir, masks_dir)
    input_hashes = {
        str(path): sha256(path)
        for pair in pairs
        for path in (pair["image"], pair["mask"])
    }
    arms: dict[str, Any] = {}
    base_rng_state = None
    for sampler in ("legacy", "model388"):
        dataset = SamplerProbeDataset(images_dir, masks_dir, sampler=sampler, seed=SEED)
        if base_rng_state is None:
            base_rng_state = dataset.rng.getstate()
        elif dataset.rng.getstate() != base_rng_state:
            raise ValueError("candidate sampler changed the legacy coordinate-cap RNG initialization")
        shuffle_generator = torch.Generator(device="cpu")
        shuffle_generator.manual_seed(395000 + fold)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=WORKERS,
            worker_init_fn=seed_worker,
            drop_last=True,
            pin_memory=False,
            persistent_workers=False,
            generator=shuffle_generator,
        )
        arms[sampler] = _summarize_arm(dataset, loader, epochs)
        del loader, dataset

    ratios: dict[str, Any] = {}
    for class_id, class_name in CLASSES:
        ratios[class_name] = {"realized_anchor_ratio": {}, "crop_pixel_exposure_ratio": {}}
        for bin_name, _, _ in AREA_BINS:
            legacy_anchor = arms["legacy"]["anchor_counts"][class_name][bin_name]
            candidate_anchor = arms["model388"]["anchor_counts"][class_name][bin_name]
            legacy_pixels = arms["legacy"]["crop_gt_pixel_exposure"][class_name][bin_name]
            candidate_pixels = arms["model388"]["crop_gt_pixel_exposure"][class_name][bin_name]
            ratios[class_name]["realized_anchor_ratio"][bin_name] = (
                candidate_anchor / legacy_anchor if legacy_anchor else None
            )
            ratios[class_name]["crop_pixel_exposure_ratio"][bin_name] = (
                candidate_pixels / legacy_pixels if legacy_pixels else None
            )
    return {
        "fold": fold,
        "scope": "primary" if fold in (3, 4) else "exploratory",
        "train_images": len(pairs),
        "arms": arms,
        "candidate_to_legacy_ratios": ratios,
        "train_file_sha256": input_hashes,
    }


def analyze(cv_root: Path, folds: tuple[int, ...], epochs: int = EPOCHS) -> dict[str, Any]:
    if not folds or any(fold not in (1, 2, 3, 4) for fold in folds) or len(set(folds)) != len(folds):
        raise ValueError("folds must be a non-empty unique subset of 1..4")
    if epochs != EPOCHS:
        raise ValueError(f"the frozen simulation requires exactly {EPOCHS} epochs")
    cv_root = cv_root.resolve()
    cv_inventory = sorted(str(path.relative_to(cv_root)) for path in cv_root.glob("train_fold*"))
    folds_result = {str(fold): simulate_fold(cv_root, fold, epochs) for fold in folds}
    primary_gate: dict[str, Any] = {"evaluated": False, "passed": False}
    if 3 in folds and 4 in folds:
        realized_anchor_ok = all(
            folds_result[str(fold)]["candidate_to_legacy_ratios"][class_name]
            ["realized_anchor_ratio"]["64_to_lt256"] is not None
            and folds_result[str(fold)]["candidate_to_legacy_ratios"][class_name]
            ["realized_anchor_ratio"]["64_to_lt256"] >= 2.0
            for fold in (3, 4)
            for _, class_name in CLASSES
        )
        crop_pixel_ok = all(
            any(
                folds_result[str(fold)]["candidate_to_legacy_ratios"][class_name]
                ["crop_pixel_exposure_ratio"]["64_to_lt256"] is not None
                and folds_result[str(fold)]["candidate_to_legacy_ratios"][class_name]
                ["crop_pixel_exposure_ratio"]["64_to_lt256"] >= 1.25
                for _, class_name in CLASSES
            )
            for fold in (3, 4)
        )
        target_group_support_ok = all(
            folds_result[str(fold)]["arms"]["model388"]["group_support_and_concentration"]
            [class_name]["64_to_lt256"]["groups_with_positive_anchors"] >= 5
            and folds_result[str(fold)]["arms"]["model388"]["group_support_and_concentration"]
            [class_name]["64_to_lt256"]["max_group_share_of_positive_anchors"] < 0.5
            for fold in (3, 4)
            for _, class_name in CLASSES
        )
        primary_gate = {
            "evaluated": True,
            "realized_anchor_ratio_at_least_2x_both_target_classes_both_folds": realized_anchor_ok,
            "at_least_one_target_class_crop_pixel_exposure_ratio_at_least_1_25x_each_fold": crop_pixel_ok,
            "each_target_class_anchor_coverage_at_least_5_groups_and_max_group_share_below_50pct": target_group_support_ok,
            "passed": realized_anchor_ok and crop_pixel_ok and target_group_support_ok,
        }
    return {
        "model": 395,
        "analysis": "train-mask-only realized sampler dose and native crop pixel exposure simulation",
        "metric": "not a segmentation score; no predictions or labels outside training folds read",
        "folds": list(folds),
        "configuration": {
            "epochs": epochs,
            "seed": SEED,
            "samples_per_image": 12,
            "positive_probability": 0.5,
            "minority_classes": [name for _, name in CLASSES],
            "candidate_min_component_area_px": 64,
            "connectivity": 8,
            "native_crop_size": 256,
            "workers": WORKERS,
            "batch_size": BATCH_SIZE,
            "drop_last": True,
            "persistent_workers": False,
            "shuffle_generator_seed_by_fold": "395000 + fold; identical order for paired arms",
            "private_dataset_rng_worker_reseeding": "none; reproduces current worker_init_fn behavior",
        },
        "training_data_only": True,
        "official_valid_or_test_read": False,
        "oof_predictions_read": False,
        "training_started": False,
        "cpu_preflight_gate": primary_gate,
        "folds": folds_result,
        "cv_train_fold_inventory": cv_inventory,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    result = analyze(args.cv_root, tuple(args.folds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "model": result["model"],
        "folds": result["folds"],
        "cpu_preflight_gate": result["cpu_preflight_gate"],
        "training_started": False,
        "official_valid_or_test_read": False,
        "oof_predictions_read": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
