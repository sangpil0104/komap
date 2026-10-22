#!/usr/bin/env python3
"""Validate and aggregate the four leakage-safe Model374 HRNet OOF archives."""
from __future__ import annotations

import argparse
import io
import hashlib
import json
import os
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

PALETTE = np.asarray(
    [[153, 127, 76], [25, 76, 153], [76, 178, 76], [204, 204, 204]],
    dtype=np.uint8,
)
CLASS_NAMES = ("Al", "Al3Ni", "Primary_Si", "Eutectic_Si")


def decode_mask(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    target = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        target[np.all(rgb == color, axis=-1)] = class_id
    if np.any(target == 255):
        raise ValueError(f"unknown mask color in {path}")
    return target


def image_iou(probability: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = probability.argmax(axis=0)
    values = np.zeros(4, dtype=np.float64)
    for class_id in range(4):
        truth = target == class_id
        guess = pred == class_id
        union = np.logical_or(truth, guess).sum()
        values[class_id] = np.logical_and(truth, guess).sum() / union if union else 0.0
    return values


def group_for(name: str) -> str:
    match = re.match(r"(A\d+)_", name)
    if not match:
        raise ValueError(f"cannot parse specimen group from {name}")
    return match.group(1)


def image_key(path: Path) -> str:
    suffix = "_image.png"
    if not path.name.endswith(suffix):
        raise ValueError(f"unexpected image name: {path}")
    return path.name[: -len(suffix)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-data", type=Path, default=Path("/workspace/komap_model1/data_model322_group_cv"))
    parser.add_argument("--run-root", type=Path, default=Path("/workspace/komap_model1/outputs/model374_hrnet_group_oof"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--expected-images", type=int, default=70)
    parser.add_argument("--expected-groups", type=int, default=59)
    args = parser.parse_args()
    output = args.output or args.run_root / "train_oof_probs.npz"
    summary_path = output.with_name("train_oof_summary.json")
    partial_path = output.with_name(output.name + ".incomplete")
    if output.exists() or summary_path.exists() or partial_path.exists():
        raise FileExistsError(f"refusing to overwrite Model374 aggregate: {output}")

    fold_records = []
    expected_all = set()
    groups = set()
    all_train_groups = set()
    heldout_group_owner: dict[str, int] = {}
    rows = []
    class_scores = []
    output.parent.mkdir(parents=True, exist_ok=True)
    for fold in range(1, 5):
        image_dir = args.cv_data / f"oof_fold{fold}" / "images"
        mask_dir = args.cv_data / f"oof_fold{fold}" / "masks"
        train_image_dir = args.cv_data / f"train_fold{fold}" / "images"
        archive_path = args.run_root / f"fold{fold}_oof" / "valid_probabilities.npz"
        if not archive_path.is_file():
            raise FileNotFoundError(f"missing fold-{fold} HRNet OOF archive: {archive_path}")
        expected = {image_key(path) for path in image_dir.glob("*_image.png")}
        train_names = {image_key(path) for path in train_image_dir.glob("*_image.png")}
        if not expected:
            raise FileNotFoundError(f"no held-out images in {image_dir}")
        if not train_names:
            raise FileNotFoundError(f"no training images in {train_image_dir}")
        heldout_groups = {group_for(name) for name in expected}
        train_groups = {group_for(name) for name in train_names}
        if heldout_groups & train_groups:
            raise ValueError(f"fold {fold} specimen leakage: {sorted(heldout_groups & train_groups)}")
        for group in heldout_groups:
            if group in heldout_group_owner:
                raise ValueError(
                    f"specimen {group} held out in folds {heldout_group_owner[group]} and {fold}")
            heldout_group_owner[group] = fold
        all_train_groups |= train_groups | heldout_groups
        if expected & expected_all:
            raise ValueError(f"an image occurs in multiple held-out folds: {sorted(expected & expected_all)}")
        expected_all |= expected

        run_dir = args.run_root / f"fold{fold}"
        args_path = run_dir / "args.json"
        checkpoint = run_dir / "last.pt"
        train_log = run_dir / "train.log"
        oof_dir = args.run_root / f"fold{fold}_oof"
        manifest_path = oof_dir / "inference_manifest.json"
        summary_path = oof_dir / "valid_summary.json"
        if (not args_path.is_file() or not checkpoint.is_file() or not train_log.is_file()
                or not manifest_path.is_file() or not summary_path.is_file()):
            raise FileNotFoundError(f"missing fold {fold} training/inference provenance")
        if "done. final model:" not in train_log.read_text(encoding="utf-8", errors="replace"):
            raise ValueError(f"fold {fold} training log does not record a completed 70-epoch run")
        train_args = json.loads(args_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        inference_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        data_root = args.cv_data.parent / f"data_model374_fold{fold}"
        expected_train_args = {
            "architecture": "UnetPlusPlus", "encoder_name": "tu-hrnet_w48",
            "encoder_weights": "imagenet", "patch_size": 512, "input_scale": 2.0,
            "samples_per_image": 12, "batch_size": 4, "accumulation_steps": 2,
            "epochs": 70, "lr": 3e-4, "encoder_lr_mult": 1.0,
            "weight_decay": 1e-4, "boundary_head_weight": 1.0,
            "seed": 0, "val_every": 0,
        }
        for key, expected_value in expected_train_args.items():
            if train_args.get(key) != expected_value:
                raise ValueError(f"fold {fold} training arg {key}={train_args.get(key)!r}; "
                                 f"expected {expected_value!r}")
        if manifest.get("fold") != fold or manifest.get("inference_split") != "valid":
            raise ValueError(f"fold {fold} inference manifest has wrong fold/split")
        if inference_summary.get("split") != "valid" or inference_summary.get("images") != len(expected):
            raise ValueError(f"fold {fold} OOF inference summary has wrong split/image count")
        if Path(train_args.get("data_root", "")).resolve() != data_root.resolve():
            raise ValueError(f"fold {fold} training args point to the wrong data root")
        if train_args.get("val_every") != 0 or int(train_args.get("epochs", -1)) != 70:
            raise ValueError(f"fold {fold} training used an unexpected validation/epoch setting")
        if (data_root / "train").resolve() != (args.cv_data / f"train_fold{fold}").resolve():
            raise ValueError(f"fold {fold} train symlink does not match its split")
        if (data_root / "valid").resolve() != (args.cv_data / f"oof_fold{fold}").resolve():
            raise ValueError(f"fold {fold} held-out symlink does not match its OOF split")
        if (Path(manifest.get("training_data_root", "")).resolve() != data_root.resolve()
                or Path(manifest.get("inference_data_root", "")).resolve() != data_root.resolve()
                or Path(manifest.get("heldout_data_root", "")).resolve()
                != (args.cv_data / f"oof_fold{fold}").resolve()):
            raise ValueError(f"fold {fold} inference manifest points at the wrong split")
        expected_inference = {"patch_size": 512, "input_scale": 2.0,
                              "overlap": 170, "tta": "4-way horizontal/vertical flips",
                              "downsample_antialias": False}
        if inference_summary.get("inference") != expected_inference:
            raise ValueError(f"fold {fold} inference settings differ from the fixed protocol")
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if (Path(manifest.get("checkpoint", "")).resolve() != checkpoint.resolve()
                or manifest.get("checkpoint_sha256") != checkpoint_sha
                or Path(manifest.get("training_args", "")).resolve() != args_path.resolve()):
            raise ValueError(f"fold {fold} inference manifest does not match its checkpoint/args")

        with np.load(archive_path) as archive, zipfile.ZipFile(
            partial_path, mode="a", compression=zipfile.ZIP_DEFLATED, allowZip64=True
        ) as aggregate:
            actual = set(archive.files)
            if actual != expected:
                raise ValueError(
                    f"fold {fold} archive keys mismatch; missing={sorted(expected-actual)[:5]}, "
                    f"extra={sorted(actual-expected)[:5]}"
                )
            for name in sorted(expected):
                prob = np.asarray(archive[name], dtype=np.float32)
                if prob.ndim != 3 or prob.shape[0] != 4 or not np.isfinite(prob).all():
                    raise ValueError(f"invalid probabilities for {name}: shape={prob.shape}")
                if np.any(prob < -1e-5) or not np.allclose(prob.sum(0), 1.0, atol=3e-2):
                    raise ValueError(f"non-normalized probability map for {name}")
                target = decode_mask(mask_dir / f"{name}_mask.png")
                if target.shape != prob.shape[1:]:
                    raise ValueError(f"mask/probability shape mismatch for {name}")
                ious = image_iou(prob, target)
                class_scores.append(ious)
                group = group_for(name)
                groups.add(group)
                rows.append({"image": name, "specimen_group": group,
                             "image_miou_empty0": float(ious.mean()),
                             "class_iou_empty0": dict(zip(CLASS_NAMES, ious.tolist()))})
                buf = io.BytesIO()
                np.lib.format.write_array(buf, prob.astype(np.float16), allow_pickle=False)
                aggregate.writestr(f"{name}.npy", buf.getvalue())
        fold_records.append({"fold": fold, "heldout_images": len(expected),
                             "heldout_groups": sorted(heldout_groups),
                             "checkpoint_sha256": checkpoint_sha,
                             "archive": str(archive_path)})

    if (len(expected_all) != args.expected_images or len(groups) != args.expected_groups
            or len(all_train_groups) != args.expected_groups
            or len(heldout_group_owner) != args.expected_groups):
        raise ValueError(
            f"expected {args.expected_images} unique images/{args.expected_groups} groups; "
            f"got {len(expected_all)}/{len(groups)}"
        )
    os.replace(partial_path, output)
    result = {
        "model": 374,
        "metric": "mean over 70 held-out train images of four class IoUs; empty union=0",
        "image_count": len(expected_all),
        "specimen_group_count": len(groups),
        "image_mean_miou_empty0": float(np.mean([row["image_miou_empty0"] for row in rows])),
        "class_image_mean_iou_empty0": dict(zip(
            CLASS_NAMES, np.asarray(class_scores).mean(axis=0).tolist())),
        "folds": fold_records,
        "per_image": rows,
        "probabilities": str(output),
    }
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "image_count", "specimen_group_count", "image_mean_miou_empty0",
        "class_image_mean_iou_empty0")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
