#!/usr/bin/env python3
"""Immutable source and exactly-once train-OOF artifact locks for Model410."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE_LOCK = ROOT / "experiments/results/model410_component_sampler_source_lock.json"
OOF_LOCK = ROOT / "experiments/results/model410_component_sampler_oof_lock.json"
SPLITS = ROOT / "outputs/model403_local_supcon/splits/model392_cv"
PARENTS = ROOT / "outputs/model392_widthnorm/seed1"
OUT = ROOT / "outputs/model410_component_sampler"
GROUP_RE = re.compile(r"^(A\d+)_")
STEP_RE = re.compile(r"_step(\d+)_")

SOURCE_FILES = (
    "experiments/model410_small_component_sampler_preregistration.md",
    "experiments/model410_component_uniform_sampler.py",
    "experiments/model410_component_sampler_train.py",
    "experiments/model410_component_sampler_smoke.py",
    "experiments/model410_scale_aware_bin_audit.py",
    "experiments/model410_predict_oof_probabilities.py",
    "experiments/model410_component_sampler_lock.py",
    "experiments/score_model410_component_sampler_oof.py",
    "experiments/run_model410_component_sampler_oof.sh",
    "experiments/queue_model410_component_sampler_after409.sh",
    "model397_reproduction_bundle/model392/train.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/dataset.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/model.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/losses.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/inference.py",
    "experiments/model396_fixed_architecture_oof.py",
    "experiments/model405_small_component_fixed_blend_oof.py",
    "experiments/model397_fixed_b4_model392_blend.py",
    "experiments/model406_component_hardnegative_lock.py",
    "experiments/score_model404_component_recall_oof_v6.py",
    "experiments/score_model404_component_recall_oof_v5.py",
    "experiments/score_model404_component_recall_oof_v4.py",
    "experiments/score_model404_component_recall_oof.py",
    "experiments/score_model406_component_hardnegative_oof.py",
    "model397_reproduction_bundle/scripts/score_model397_oof.py",
    "model397_reproduction_bundle/scripts/write_oof_manifest.py",
    "model397_reproduction_bundle/config/model397_group_folds.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(paths: list[Path]) -> dict[str, dict[str, int | str]]:
    result = {}
    for path in sorted({item.resolve() for item in paths}):
        if not path.is_file():
            raise FileNotFoundError(path)
        result[path.relative_to(ROOT.resolve()).as_posix()] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    return result


def split_files_and_ids() -> tuple[list[Path], dict[int, set[str]]]:
    paths: list[Path] = []
    heldout_by_fold: dict[int, set[str]] = {}
    heldout_group_fold: dict[str, int] = {}
    for fold in range(1, 5):
        fold_root = SPLITS / f"seed1_fold{fold}"
        ids_by_split = {}
        for split in ("train", "valid"):
            images_dir = fold_root / split / "images"
            masks_dir = fold_root / split / "masks"
            images = {p.name.removesuffix("_image.png"): p
                      for p in sorted(images_dir.glob("*_image.png"))}
            masks = {p.name.removesuffix("_mask.png"): p
                     for p in sorted(masks_dir.glob("*_mask.png"))}
            if not images or images.keys() != masks.keys():
                raise ValueError(f"image/mask IDs differ or are empty: fold{fold}/{split}")
            for image_id in images:
                match = GROUP_RE.match(image_id)
                if match is None or STEP_RE.search(image_id) is None:
                    raise ValueError(f"cannot parse specimen/step from {image_id}")
                if split == "valid":
                    group = match.group(1)
                    previous = heldout_group_fold.setdefault(group, fold)
                    if previous != fold:
                        raise ValueError(f"specimen group crosses held-out folds: {group}")
            ids_by_split[split] = set(images)
            paths.extend(images.values())
            paths.extend(masks.values())
        train_groups = {GROUP_RE.match(name).group(1) for name in ids_by_split["train"]}
        valid_groups = {GROUP_RE.match(name).group(1) for name in ids_by_split["valid"]}
        if train_groups & valid_groups:
            raise ValueError(f"specimen leakage in fold {fold}")
        heldout_by_fold[fold] = ids_by_split["valid"]

    all_ids = set().union(*heldout_by_fold.values())
    if (len(all_ids) != 70 or sum(map(len, heldout_by_fold.values())) != 70
            or len(heldout_group_fold) != 59):
        raise ValueError(
            f"expected 70 exactly-once held-out images/59 groups; "
            f"got {len(all_ids)}/{len(heldout_group_fold)}"
        )
    with (ROOT / "model397_reproduction_bundle/config/model397_group_folds.json").open(
        encoding="utf-8"
    ) as stream:
        fold_manifest = json.load(stream)
    manifest_groups = {
        int(item["fold"]): set(item["heldout_groups"])
        for item in fold_manifest["folds"]
    }
    actual_groups = {
        fold: {GROUP_RE.match(image_id).group(1) for image_id in ids}
        for fold, ids in heldout_by_fold.items()
    }
    if actual_groups != manifest_groups:
        raise ValueError("materialized Model410 folds differ from the fixed Model397 specimen-group manifest")

    official_train_ids = {
        path.name.removesuffix("_image.png")
        for path in (ROOT / "data/train/images").glob("*_image.png")
    }
    official_train_masks = {
        path.name.removesuffix("_mask.png")
        for path in (ROOT / "data/train/masks").glob("*_mask.png")
    }
    if official_train_ids != all_ids or official_train_masks != all_ids:
        raise ValueError("fixed folds do not exactly cover data/train images and masks")

    component_audit_ids = set()
    for fold in range(1, 5):
        component_audit_ids.update(
            path.name.removesuffix("_mask.png")
            for path in (ROOT / f"data_model322_group_cv/oof_fold{fold}/masks").glob("*_mask.png")
        )
    if component_audit_ids != all_ids:
        raise ValueError("Model406 component-audit masks differ from the fixed 70-image train set")
    return paths, heldout_by_fold


def expected_source_inventory() -> tuple[dict, dict, dict, dict, dict, dict[int, set[str]]]:
    source_paths = [ROOT / name for name in SOURCE_FILES]
    data_paths, heldout = split_files_and_ids()
    parent_paths = [PARENTS / f"fold{fold}/last.pt" for fold in range(1, 5)]
    parent_paths.extend(
        PARENTS / f"fold{fold}_oof/valid_probabilities.npz" for fold in range(1, 5)
    )
    reference_paths = []
    reference_paths.extend((ROOT / "data/train/images").glob("*_image.png"))
    reference_paths.extend((ROOT / "data/train/masks").glob("*_mask.png"))
    for fold in range(1, 5):
        reference_paths.extend(
            (ROOT / f"data_model322_group_cv/oof_fold{fold}/masks").glob("*_mask.png")
        )
    if not reference_paths:
        raise FileNotFoundError("missing fixed train labels/images used by the one-shot scorer")
    import torch

    parent_metadata = {}
    for fold in range(1, 5):
        checkpoint_path = PARENTS / f"fold{fold}/last.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_config = checkpoint.get("model_config")
        preprocessing = checkpoint.get("preprocessing_config")
        if model_config != {
            "architecture": "UnetPlusPlus",
            "encoder_name": "tu-hrnet_w48",
            "boundary_head": True,
        }:
            raise ValueError(f"unexpected Model392 parent architecture in fold {fold}")
        train_images = SPLITS / f"seed1_fold{fold}/train/images"
        widths = []
        for path in sorted(train_images.glob("*_image.png")):
            with Image.open(path) as image:
                widths.append(image.width)
        if not widths or not isinstance(preprocessing, dict):
            raise ValueError(f"missing training images or preprocessing metadata in fold {fold}")
        counts = {width: widths.count(width) for width in set(widths)}
        target_width = min(width for width, count in counts.items() if count == max(counts.values()))
        if (preprocessing.get("patch_size") != 512
                or preprocessing.get("input_scale") != 2.0
                or preprocessing.get("normalize_width") is not True
                or preprocessing.get("target_width") != target_width):
            raise ValueError(f"Model392 preprocessing mismatch in fold {fold}")
        parent_metadata[str(fold)] = {
            "model_config": model_config,
            "preprocessing_config": preprocessing,
        }
    return (
        inventory(source_paths), inventory(data_paths), inventory(parent_paths),
        inventory(reference_paths), parent_metadata, heldout,
    )


def create_source_lock() -> dict:
    if SOURCE_LOCK.exists():
        raise FileExistsError(f"refusing to overwrite {SOURCE_LOCK}")
    sources, data, parents, references, parent_metadata, heldout = expected_source_inventory()
    payload = {
        "experiment": "Model410 small-component-balanced crop sampling",
        "locked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "fixed train specimen-group OOF only; official valid/test excluded",
        "locked_sources": sources,
        "fixed_fold_train_and_oof_images_masks": data,
        "model392_seed1_parent_checkpoints_and_oof": parents,
        "train_labels_and_baseline_oof_used_by_scorer": references,
        "model392_parent_model_and_preprocessing_metadata": parent_metadata,
        "heldout_image_ids_by_fold": {str(k): sorted(v) for k, v in heldout.items()},
        "expected_train_images": 70,
        "expected_specimen_groups": 59,
    }
    SOURCE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with SOURCE_LOCK.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return payload


def verify_source_lock() -> dict:
    record = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    sources, data, parents, references, parent_metadata, heldout = expected_source_inventory()
    expected_ids = {str(k): sorted(v) for k, v in heldout.items()}
    if (record.get("experiment") != "Model410 small-component-balanced crop sampling"
            or record.get("scope") != "fixed train specimen-group OOF only; official valid/test excluded"
            or record.get("locked_sources") != sources
            or record.get("fixed_fold_train_and_oof_images_masks") != data
            or record.get("model392_seed1_parent_checkpoints_and_oof") != parents
            or record.get("train_labels_and_baseline_oof_used_by_scorer") != references
            or record.get("model392_parent_model_and_preprocessing_metadata") != parent_metadata
            or record.get("heldout_image_ids_by_fold") != expected_ids):
        raise ValueError("Model410 source/data/parent lock mismatch")
    return record


def expected_oof_artifacts() -> dict[str, dict[str, int | str]]:
    source = verify_source_lock()
    paths: list[Path] = []
    heldout = split_files_and_ids()[1]
    for arm in ("control", "candidate"):
        for fold in range(1, 5):
            run = OUT / arm / f"fold{fold}"
            oof = OUT / arm / f"fold{fold}_oof"
            arm_receipt = run / "model410_sampler_arm.json"
            args = run / "args.json"
            checkpoint = run / "last.pt"
            probabilities = oof / "valid_probabilities.npz"
            required = (arm_receipt, args, checkpoint, probabilities)
            if any(not path.is_file() for path in required):
                raise FileNotFoundError(next(path for path in required if not path.is_file()))
            receipt = json.loads(arm_receipt.read_text(encoding="utf-8"))
            args_record = json.loads(args.read_text(encoding="utf-8"))
            expected_args = {
                "seed": 1,
                "epochs": 8,
                "architecture": "UnetPlusPlus",
                "encoder_name": "tu-hrnet_w48",
                "encoder_weights": "none",
                "patch_size": 512,
                "input_scale": 2.0,
                "batch_size": 4,
                "accumulation_steps": 2,
                "samples_per_image": 12,
                "normalize_width": True,
                "val_every": 0,
                "lr": 1e-4,
                "encoder_lr_mult": 0.3,
                "weight_decay": 1e-4,
                "rng_mode": "per_item",
                "num_workers": 4,
                "no_boundary_head": False,
                "boundary_head_weight": 1.0,
            }
            expected_data = SPLITS / f"seed1_fold{fold}"
            expected_parent = PARENTS / f"fold{fold}/last.pt"
            expected_target_width = source["model392_parent_model_and_preprocessing_metadata"][
                str(fold)
            ]["preprocessing_config"]["target_width"]
            expected_receipt = {
                "experiment": "Model410 small-component-balanced crop sampling",
                "arm": arm,
                "area_bins_modal_width_equivalent_pixels": [[1, 4], [4, 16], [16, 64]],
                "normalized_area_formula": "raw_component_area*(target_width/image_width)^2",
                "positive_sample_probability": 0.5,
                "classes": ["Al3Ni", "Eutectic Si"],
                "valid_read": False,
                "test_read": False,
            }
            if (any(receipt.get(key) != value for key, value in expected_receipt.items())
                    or Path(args_record.get("data_root", "")).resolve() != expected_data.resolve()
                    or Path(args_record.get("init_checkpoint", "")).resolve() != expected_parent.resolve()
                    or args_record.get("target_width") != expected_target_width
                    or any(args_record.get(key) != value for key, value in expected_args.items())):
                raise ValueError(f"Model410 training recipe mismatch: {arm}/fold{fold}")
            with np.load(probabilities, allow_pickle=False) as archive:
                if set(archive.files) != heldout[fold]:
                    raise ValueError(f"OOF image IDs mismatch: {arm}/fold{fold}")
                for image_id in archive.files:
                    value = archive[image_id]
                    if value.ndim != 3 or value.shape[0] != 4 or not np.isfinite(value).all():
                        raise ValueError(f"invalid probability map: {arm}/fold{fold}/{image_id}")
            manifest = oof / "inference_manifest.json"
            if not manifest.is_file():
                raise FileNotFoundError(manifest)
            manifest_record = json.loads(manifest.read_text(encoding="utf-8"))
            inference = manifest_record.get("inference", {})
            if (manifest_record.get("masks_opened") is not False
                    or manifest_record.get("metrics_computed") is not False
                    or manifest_record.get("official_valid_or_test_used") is not False
                    or manifest_record.get("images") != len(heldout[fold])
                    or manifest_record.get("checkpoint_sha256") != sha256(checkpoint)
                    or inference.get("patch_size") != 512
                    or inference.get("input_scale") != 2.0
                    or inference.get("target_width") != expected_target_width
                    or inference.get("overlap_base") != 170
                    or inference.get("tta") != "flip4"
                    or inference.get("batch_size") != 2
                    or set(manifest_record.get("image_ids", [])) != heldout[fold]):
                raise ValueError(f"Model410 prediction-only manifest mismatch: {arm}/fold{fold}")
            paths.extend(required)
            paths.append(manifest)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("incomplete Model410 evaluator metadata")
    return inventory(paths)


def create_oof_lock() -> dict:
    if OOF_LOCK.exists():
        raise FileExistsError(f"refusing to overwrite {OOF_LOCK}")
    source = verify_source_lock()
    artifacts = expected_oof_artifacts()
    payload = {
        "experiment": "Model410 paired small-component sampler train-OOF",
        "locked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_lock_sha256": sha256(SOURCE_LOCK),
        "locked_oof_artifacts": artifacts,
        "training_images": source["expected_train_images"],
        "specimen_groups": source["expected_specimen_groups"],
        "official_valid_and_test_used": False,
        "one_shot_scorer": "score_model410_component_sampler_oof.py",
    }
    with OOF_LOCK.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return payload


def verify_oof_lock() -> dict:
    record = json.loads(OOF_LOCK.read_text(encoding="utf-8"))
    artifacts = expected_oof_artifacts()
    if (record.get("experiment") != "Model410 paired small-component sampler train-OOF"
            or record.get("source_lock_sha256") != sha256(SOURCE_LOCK)
            or record.get("locked_oof_artifacts") != artifacts
            or record.get("training_images") != 70 or record.get("specimen_groups") != 59
            or record.get("official_valid_and_test_used") is not False):
        raise ValueError("Model410 one-shot OOF lock mismatch")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--create-source-lock", action="store_true")
    actions.add_argument("--verify-source-lock", action="store_true")
    actions.add_argument("--create-oof-lock", action="store_true")
    actions.add_argument("--verify-oof-lock", action="store_true")
    args = parser.parse_args()
    if args.root.resolve() != ROOT.resolve():
        raise ValueError("Model410 lock is tied to this repository root")
    if args.create_source_lock:
        result = create_source_lock()
    elif args.verify_source_lock:
        result = verify_source_lock()
    elif args.create_oof_lock:
        result = create_oof_lock()
    else:
        result = verify_oof_lock()
    print(json.dumps({"ok": True, "experiment": result["experiment"]}, indent=2))


if __name__ == "__main__":
    main()
