#!/usr/bin/env python3
"""Train-only screen of target-class absence in Model392-style augmented crops.

This screen samples a fixed, evenly spaced subset of the deterministic
per-item crop stream used by Models 404/407. It measures only whether the
post-augmentation training mask contains Al3Ni or Eutectic Si; it does not
train a model, compute a score, or read official validation/test data.
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    import sys

    import torch

    sys.path.insert(0, str(SOURCE_ROOT))
    from komap_seg.dataset import KomapPatchDataset, modal_image_width  # noqa: E402

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-root", type=Path,
        default=ROOT / "outputs/model403_local_supcon/splits/model392_cv",
    )
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "experiments/results/model408_crop_presence_screen.json",
    )
    parser.add_argument("--probes-per-fold-epoch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    if args.probes_per_fold_epoch != 32 or args.epochs != 8:
        raise ValueError("the screen is fixed at 32 probes/fold/epoch for 8 epochs")
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite crop-presence screen: {output}")

    records: dict[str, dict] = {}
    for fold in range(1, 5):
        data_root = args.split_root / f"seed1_fold{fold}"
        images_dir = data_root / "train/images"
        masks_dir = data_root / "train/masks"
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
        if len(dataset) < args.probes_per_fold_epoch:
            raise ValueError(f"fold {fold} has too few train crops: {len(dataset)}")
        indices = np.linspace(
            0, len(dataset) - 1, args.probes_per_fold_epoch, dtype=np.int64
        ).tolist()
        if len(set(indices)) != args.probes_per_fold_epoch:
            raise ValueError(f"fold {fold} probe indices are not unique")

        fold_record = {
            "train_images": len(dataset.images),
            "virtual_crops_per_epoch": len(dataset),
            "target_width": target_width,
            "probe_indices_per_epoch": indices,
            "epochs": {},
        }
        totals = {"Al3Ni": {"present": 0, "absent": 0},
                  "Eutectic Si": {"present": 0, "absent": 0}}
        for epoch in range(1, args.epochs + 1):
            dataset.set_epoch(epoch)
            counts = {"Al3Ni": {"present": 0, "absent": 0},
                      "Eutectic Si": {"present": 0, "absent": 0}}
            for index in indices:
                _, mask = dataset[index]
                for name, class_id in (("Al3Ni", 1), ("Eutectic Si", 3)):
                    present = bool(torch.any(mask == class_id).item())
                    counts[name]["present" if present else "absent"] += 1
            for name in totals:
                for state in ("present", "absent"):
                    totals[name][state] += counts[name][state]
            fold_record["epochs"][str(epoch)] = counts
        fold_record["all_probe_epochs"] = {
            name: {
                **counts,
                "total": sum(counts.values()),
                "absent_fraction": counts["absent"] / sum(counts.values()),
            }
            for name, counts in totals.items()
        }
        records[str(fold)] = fold_record
        print(
            f"fold={fold} images={len(dataset.images)} crops/epoch={len(dataset)} "
            f"Al3Ni_absent={totals['Al3Ni']['absent']}/256 "
            f"Eutectic_absent={totals['Eutectic Si']['absent']}/256",
            flush=True,
        )

    payload = {
        "experiment": "Model408 deterministic training-crop target-presence screen",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "four fixed outer-train splits only; sampled pre-training crop stream",
        "sampling": {
            "run_seed": 1,
            "rng_mode": "per_item",
            "probes_per_fold_epoch": args.probes_per_fold_epoch,
            "epochs": args.epochs,
            "total_probes": 4 * args.probes_per_fold_epoch * args.epochs,
            "probe_selection": "32 unique evenly spaced virtual-dataset indices per fold and epoch",
            "patch_size": 512,
            "input_scale": 2.0,
            "samples_per_image": 12,
            "post_augmentation_mask": True,
        },
        "classes": {"Al3Ni": 1, "Eutectic Si": 3},
        "folds": records,
        "optimizer_steps": 0,
        "miou_computed": False,
        "official_valid_read": False,
        "test_read": False,
        "dataset_code_sha256": sha256(SOURCE_ROOT / "komap_seg/dataset.py"),
        "screen_script_sha256": sha256(Path(__file__).resolve()),
        "interpretation": (
            "Exposure screen only; crop-level absence is not evidence of model improvement. "
            "Any follow-up loss needs a separately preregistered grouped-OOF evaluation."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
