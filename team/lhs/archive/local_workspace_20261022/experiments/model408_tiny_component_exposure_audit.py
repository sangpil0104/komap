#!/usr/bin/env python3
"""Read-only train-crop exposure audit for Model408's tiny GT components.

Uses the same deterministic 32-crop/fold/epoch probe stream as the existing
class-presence screen, but counts the exact border-excluded components and
size bins consumed by Model408's auxiliary loss. No model, score, valid, or
test data is read.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "model336_hrnet_widthnorm_candidate"
AREA_BINS = ((4, 16), (16, 64), (64, 256))
TARGETS = ((1, "Al3Ni"), (3, "Eutectic Si"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_paths(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-root", type=Path,
        default=ROOT / "outputs/model403_local_supcon/splits/model392_cv",
    )
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "experiments/results/model408_tiny_component_exposure_audit.json",
    )
    parser.add_argument("--probes-per-fold-epoch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    if args.probes_per_fold_epoch != 32 or args.epochs != 8:
        raise ValueError("probe stream is fixed at 32 crops/fold/epoch for 8 epochs")
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite tiny-component exposure audit: {output}")

    import sys

    import torch

    sys.path.insert(0, str(SOURCE_ROOT))
    sys.path.insert(0, str(ROOT / "experiments"))
    from komap_seg.dataset import KomapPatchDataset, modal_image_width  # noqa: E402
    from model408_tiny_confusion_loss import build_component_targets  # noqa: E402

    fold_records: dict[str, dict] = {}
    for fold in range(1, 5):
        data_root = args.split_root / f"seed1_fold{fold}"
        images_dir = data_root / "train/images"
        masks_dir = data_root / "train/masks"
        image_paths = sorted(images_dir.glob("*_image.png"))
        mask_paths = sorted(masks_dir.glob("*_mask.png"))
        image_stems = [path.name[:-len("_image.png")] for path in image_paths]
        mask_stems = [path.name[:-len("_mask.png")] for path in mask_paths]
        if not mask_paths or image_stems != mask_stems:
            raise ValueError(f"fold {fold}: mismatched/empty train image-mask lists")
        target_width = modal_image_width(images_dir)
        dataset = KomapPatchDataset(
            images_dir,
            masks_dir,
            patch_size=512,
            input_scale=2.0,
            target_width=target_width,
            samples_per_image=12,
            seed=1,
            rng_mode="per_item",
        )
        indices = np.linspace(
            0, len(dataset) - 1, args.probes_per_fold_epoch, dtype=np.int64
        ).tolist()
        if len(set(indices)) != args.probes_per_fold_epoch:
            raise ValueError(f"fold {fold} probe indices are not unique")

        component_totals = {
            name: np.zeros(len(AREA_BINS), dtype=np.int64) for _, name in TARGETS
        }
        crop_hits = {
            name: np.zeros(len(AREA_BINS), dtype=np.int64) for _, name in TARGETS
        }
        for epoch in range(1, args.epochs + 1):
            dataset.set_epoch(epoch)
            for index in indices:
                _, mask = dataset[index]
                components = build_component_targets(mask.unsqueeze(0))
                class_ids = components.class_ids.numpy()
                bin_ids = components.bin_indices.numpy()
                for class_id, name in TARGETS:
                    selected_bins = bin_ids[class_ids == class_id]
                    component_totals[name] += np.bincount(
                        selected_bins, minlength=len(AREA_BINS)
                    )[:len(AREA_BINS)]
                    crop_hits[name] += np.bincount(
                        np.unique(selected_bins), minlength=len(AREA_BINS)
                    )[:len(AREA_BINS)]

        per_class = {}
        for _, name in TARGETS:
            per_class[name] = {
                f"{lower}-{upper - 1}_augmented_px": {
                    "components": int(component_totals[name][slot]),
                    "crops_with_at_least_one": int(crop_hits[name][slot]),
                    "crop_exposure_fraction": float(
                        crop_hits[name][slot]
                        / (args.probes_per_fold_epoch * args.epochs)
                    ),
                }
                for slot, (lower, upper) in enumerate(AREA_BINS)
            }
        fold_records[str(fold)] = {
            "train_images": len(dataset.images),
            "target_width": target_width,
            "train_images_sha256": hash_paths(image_paths),
            "train_masks_file_count": len(mask_paths),
            "train_masks_sha256": hash_paths(mask_paths),
            "virtual_crops_per_epoch": len(dataset),
            "probe_indices_per_epoch": indices,
            "probes": args.probes_per_fold_epoch * args.epochs,
            "size_bin_exposure": per_class,
        }
        print(
            f"fold={fold} probes={args.probes_per_fold_epoch * args.epochs} "
            f"Al3Ni={component_totals['Al3Ni'].tolist()} "
            f"EutecticSi={component_totals['Eutectic Si'].tolist()}",
            flush=True,
        )

    payload = {
        "experiment": "Model408 deterministic tiny-GT-component crop exposure audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "fixed outer-train splits only; sampled deterministic post-augmentation crops",
        "sampling": {
            "seed": 1,
            "rng_mode": "per_item",
            "probes_per_fold_epoch": args.probes_per_fold_epoch,
            "epochs": args.epochs,
            "total_probes": 4 * args.probes_per_fold_epoch * args.epochs,
            "patch_size": 512,
            "input_scale": 2.0,
            "samples_per_image": 12,
            "crop_indices": "32 unique evenly spaced virtual-dataset indices per fold and epoch",
            "component_definition": "same Model408 CPU 8-connected labels, area bins, and crop-border exclusion",
            "area_bins_augmented_pixels": [list(bounds) for bounds in AREA_BINS],
        },
        "folds": fold_records,
        "optimizer_steps": 0,
        "miou_computed": False,
        "official_valid_read": False,
        "test_read": False,
        "dataset_code_sha256": sha256(SOURCE_ROOT / "komap_seg/dataset.py"),
        "loss_code_sha256": sha256(ROOT / "experiments/model408_tiny_confusion_loss.py"),
        "script_sha256": sha256(Path(__file__).resolve()),
        "interpretation": (
            "Descriptive exposure diagnostic only; it does not estimate model performance "
            "or authorize loss/sampler tuning on the same OOF split."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
