#!/usr/bin/env python3
"""One-shot, train-only OOF localization diagnostic for Model392's boundary head."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


PROTOCOL_REL = "experiments/model392_boundary_head_localization_preregistration.md"
SCRIPT_REL = "experiments/model392_boundary_head_localization.py"
LOCK_REL = "experiments/results/model392_boundary_head_localization_lock.json"
OUTPUT_REL = "experiments/results/model392_boundary_head_localization_oof.json"
GROUP_RE = re.compile(r"^(A\d+)")
EXPECTED_IMAGES = 70
EXPECTED_GROUPS = 59
SOURCE_RELS = (
    PROTOCOL_REL,
    SCRIPT_REL,
    "model336_hrnet_widthnorm_candidate/komap_seg/checkpoint.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/dataset.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/inference.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/losses.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/model.py",
    "model336_hrnet_widthnorm_candidate/train.py",
    "experiments/results/model392_widthnorm_runtime_lock.json",
    "experiments/results/model392_sequence_closure_lock.json",
)
CONFIG = {
    "model": 392,
    "seeds": [1, 2],
    "folds": [1, 2, 3, 4],
    "patch_size": 512,
    "base_input_scale": 2.0,
    "tta": "flip4",
    "overlap_ratio": 0.25,
    "stitching": "Gaussian center-weighted mean of sigmoid boundary probabilities",
    "boundary_target": "gt_boundary_band(width=1, four classes)",
    "primary_metric": "per-image average precision for class-agnostic phase boundary",
    "group_unit": "A-prefixed specimen group",
    "bootstrap_draws": 20000,
    "bootstrap_seed": 392081,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def group_of(stem: str) -> str:
    match = GROUP_RE.match(stem)
    if not match:
        raise ValueError(f"cannot parse specimen group from image ID: {stem}")
    return match.group(1)


def paired_files(root: Path, fold: int) -> tuple[dict[str, Path], dict[str, Path]]:
    base = root / "data_model322_group_cv" / f"oof_fold{fold}"
    images_dir, masks_dir = base / "images", base / "masks"
    if images_dir.is_symlink() or masks_dir.is_symlink():
        raise ValueError(f"refusing symlinked OOF input directories: {base}")
    images = {p.name.removesuffix("_image.png"): p for p in images_dir.glob("*_image.png")}
    masks = {p.name.removesuffix("_mask.png"): p for p in masks_dir.glob("*_mask.png")}
    if not images or set(images) != set(masks):
        raise ValueError(f"fold {fold}: OOF images/masks are empty or do not pair exactly")
    return images, masks


def verify_split(root: Path, fold: int, oof_images: dict[str, Path]) -> dict[str, Any]:
    cv_root = root / "data_model322_group_cv"
    train_dir = cv_root / f"train_fold{fold}" / "images"
    train_ids = {
        p.name.removesuffix("_image.png") for p in train_dir.glob("*_image.png")
    }
    heldout_ids = set(oof_images)
    train_groups = {group_of(stem) for stem in train_ids}
    heldout_groups = {group_of(stem) for stem in heldout_ids}
    overlap = train_groups & heldout_groups
    if overlap:
        raise ValueError(f"fold {fold}: training/OOF specimen groups overlap: {sorted(overlap)}")
    return {
        "fold": fold,
        "train_images": len(train_ids),
        "heldout_images": len(heldout_ids),
        "train_groups": len(train_groups),
        "heldout_groups": len(heldout_groups),
        "group_overlap": sorted(overlap),
        "heldout_ids": sorted(heldout_ids),
    }


def snapshot_inputs(root: Path) -> dict[str, Any]:
    source_hashes: dict[str, str] = {}
    for rel in SOURCE_RELS:
        path = root / rel
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"missing or symlinked locked source: {rel}")
        source_hashes[rel] = sha256(path)

    runtime_path = root / "experiments/results/model392_widthnorm_runtime_lock.json"
    sequence_path = root / "experiments/results/model392_sequence_closure_lock.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    sequence = json.loads(sequence_path.read_text(encoding="utf-8"))
    if runtime.get("revision") != 3 or sequence.get("revision") != 1:
        raise ValueError("unexpected Model392 runtime/sequence lock revision")
    if sha256(runtime_path) != "68e7f2d524e338f24d9e88472d1463aadc69d5cb4daf7a56e0d5997edc3ebc98":
        raise ValueError("Model392 revision-3 runtime lock differs from the reviewed lock")

    output_root = root / "outputs/model392_widthnorm"
    if output_root.is_symlink():
        raise ValueError("refusing a symlinked Model392 output root")
    fixed_inputs: dict[str, str] = {}
    fold_inventory: dict[str, Any] = {}
    all_ids: set[str] = set()
    all_groups: set[str] = set()
    group_fold: dict[str, int] = {}

    for fold in (1, 2, 3, 4):
        images, masks = paired_files(root, fold)
        split = verify_split(root, fold, images)
        for stem in images:
            if stem in all_ids:
                raise ValueError(f"duplicate OOF image across folds: {stem}")
            group = group_of(stem)
            if group in group_fold and group_fold[group] != fold:
                raise ValueError(f"specimen group {group} appears in multiple OOF folds")
            group_fold[group] = fold
            all_ids.add(stem)
            all_groups.add(group)
            fixed_inputs[str(images[stem].relative_to(root))] = sha256(images[stem])
            fixed_inputs[str(masks[stem].relative_to(root))] = sha256(masks[stem])

        train_widths = []
        train_images_dir = root / "data_model322_group_cv" / f"train_fold{fold}" / "images"
        for image_path in sorted(train_images_dir.glob("*_image.png")):
            with Image.open(image_path) as im:
                train_widths.append(im.width)
        if not train_widths:
            raise FileNotFoundError(f"fold {fold}: no training images for target-width audit")
        counts = {width: train_widths.count(width) for width in set(train_widths)}
        modal_width = min(width for width, count in counts.items()
                          if count == max(counts.values()))

        for seed in (1, 2):
            run_dir = output_root / f"seed{seed}" / f"fold{fold}"
            oof_dir = output_root / f"seed{seed}" / f"fold{fold}_oof"
            checkpoint = run_dir / "last.pt"
            args_path = run_dir / "args.json"
            receipt_path = run_dir / "model392_run_receipt.json"
            manifest_path = oof_dir / "model392_manifest.json"
            for path in (checkpoint, args_path, receipt_path, manifest_path):
                if not path.is_file() or path.is_symlink():
                    raise FileNotFoundError(f"missing or symlinked Model392 artifact: {path}")
            args = json.loads(args_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_data = root / f"data_model382_seed{seed}_fold{fold}"
            if Path(args["data_root"]).resolve() != expected_data.resolve():
                raise ValueError(f"seed{seed}/fold{fold}: unexpected training data root")
            if (expected_data / "train").resolve() != (
                    root / "data_model322_group_cv" / f"train_fold{fold}").resolve():
                raise ValueError(f"seed{seed}/fold{fold}: training split link mismatch")
            if (expected_data / "valid").resolve() != (
                    root / "data_model322_group_cv" / f"oof_fold{fold}").resolve():
                raise ValueError(f"seed{seed}/fold{fold}: held-out OOF link mismatch")
            expected_args = {
                "architecture": "UnetPlusPlus",
                "encoder_name": "tu-hrnet_w48",
                "patch_size": 512,
                "input_scale": 2.0,
                "target_width": modal_width,
                "normalize_width": True,
                "epochs": 70,
                "seed": seed,
                "no_boundary_head": False,
                "boundary_head_weight": 1.0,
                "val_every": 0,
            }
            for key, expected in expected_args.items():
                if args.get(key) != expected:
                    raise ValueError(
                        f"seed{seed}/fold{fold}: args[{key!r}]={args.get(key)!r}, "
                        f"expected {expected!r}"
                    )
            checkpoint_hash = sha256(checkpoint)
            if (manifest.get("model") != 392 or manifest.get("seed") != seed
                    or manifest.get("fold") != fold or manifest.get("epochs") != 70
                    or manifest.get("checkpoint_sha256") != checkpoint_hash
                    or manifest.get("target_width") != modal_width):
                raise ValueError(f"seed{seed}/fold{fold}: manifest/checkpoint mismatch")
            if manifest.get("model392_lock_sha256") != sha256(runtime_path):
                raise ValueError(f"seed{seed}/fold{fold}: runtime lock reference mismatch")
            fold_inventory.setdefault(str(fold), {})[str(seed)] = {
                "image_ids": sorted(images),
                "group_ids": sorted({group_of(stem) for stem in images}),
                "target_width": modal_width,
                "checkpoint_sha256": checkpoint_hash,
                "args_sha256": sha256(args_path),
                "run_receipt_sha256": sha256(receipt_path),
                "oof_manifest_sha256": sha256(manifest_path),
            }

        fold_inventory[str(fold)]["split"] = split

    if len(all_ids) != EXPECTED_IMAGES or len(all_groups) != EXPECTED_GROUPS:
        raise ValueError(
            f"expected {EXPECTED_IMAGES} OOF images/{EXPECTED_GROUPS} groups; "
            f"found {len(all_ids)}/{len(all_groups)}"
        )
    return {
        "project_root": str(root.resolve()),
        "config": CONFIG,
        "source_sha256": source_hashes,
        "input_sha256": fixed_inputs,
        "fold_inventory": fold_inventory,
        "oof_images": len(all_ids),
        "specimen_groups": len(all_groups),
        "scope": "train-only grouped OOF images/masks and Model392 checkpoints; no valid/test",
    }


def make_lock(root: Path, lock_path: Path) -> dict[str, Any]:
    if lock_path.exists():
        raise FileExistsError(f"refusing to overwrite existing lock: {lock_path}")
    output_path = root / OUTPUT_REL
    if output_path.exists() or Path(str(output_path) + ".claim").exists():
        raise FileExistsError("a prior localization result/claim exists; refusing a new lock")
    body = {
        "experiment": "Model392 boundary-head localization probe",
        "revision": 1,
        "locked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "snapshot": snapshot_inputs(root),
    }
    body["lock_sha256"] = canonical_sha256(body)
    write_exclusive(lock_path, body)
    return body


def verify_lock(root: Path, lock_path: Path) -> dict[str, Any]:
    locked = json.loads(lock_path.read_text(encoding="utf-8"))
    digest = locked.get("lock_sha256")
    body = {key: value for key, value in locked.items() if key != "lock_sha256"}
    if digest != canonical_sha256(body):
        raise ValueError("localization lock content hash is invalid")
    current = snapshot_inputs(root)
    if current != locked.get("snapshot"):
        raise ValueError("localization inputs/source/checkpoints changed since lock creation")
    return locked


def gaussian_window(size: int, device: torch.device) -> torch.Tensor:
    sigma = size * 0.125
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    one = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    window = one[:, None] * one[None, :]
    return (window / window.max()).clamp(min=1e-3)


def tile_starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def boundary_tta_batch(
    model: torch.nn.Module,
    batch: torch.Tensor,
    forward_with_boundary,
) -> torch.Tensor:
    """Return flip4-averaged sigmoid boundary probabilities in input orientation."""
    total = torch.zeros(
        (batch.shape[0], 1, batch.shape[-2], batch.shape[-1]),
        dtype=torch.float32, device=batch.device,
    )
    for dims in (None, (-1,), (-2,), (-2, -1)):
        transformed = batch if dims is None else batch.flip(dims)
        _, logits = forward_with_boundary(model, transformed)
        probabilities = torch.sigmoid(logits.float())
        if dims is not None:
            probabilities = probabilities.flip(dims)
        total += probabilities
    return total / 4.0


@torch.inference_mode()
def predict_boundary_map(
    model: torch.nn.Module,
    image: torch.Tensor,
    patch_size: int,
    input_scale: float,
    device: torch.device,
    batch_size: int,
    overlap_ratio: float = 0.25,
    forward_with_boundary=None,
) -> tuple[np.ndarray, float, int]:
    if forward_with_boundary is None:
        raise ValueError("forward_with_boundary callback is required")
    if not 0 < overlap_ratio < 0.9:
        raise ValueError("overlap_ratio must be in (0, 0.9)")
    _, height, width = image.shape
    tile = round(patch_size / input_scale)
    overlap = min(tile - 1, round(tile * overlap_ratio))
    stride = tile - overlap
    pad_h, pad_w = max(tile - height, 0), max(tile - width, 0)
    mode = "reflect" if pad_h < height and pad_w < width else "replicate"
    padded = F.pad(image.unsqueeze(0), (0, pad_w, 0, pad_h), mode=mode).squeeze(0)
    full_h, full_w = padded.shape[-2:]
    starts_y = tile_starts(full_h, tile, stride)
    starts_x = tile_starts(full_w, tile, stride)
    tiles = [(y, x) for y in starts_y for x in starts_x]
    score_sum = torch.zeros((full_h, full_w), dtype=torch.float32, device=device)
    weight_sum = torch.zeros_like(score_sum)
    window = gaussian_window(tile, device)

    model.eval()
    for offset in range(0, len(tiles), batch_size):
        batch_positions = tiles[offset:offset + batch_size]
        native = torch.stack([
            padded[:, y:y + tile, x:x + tile] for y, x in batch_positions
        ]).to(device, non_blocking=False)
        if tile != patch_size:
            network_input = F.interpolate(
                native, size=(patch_size, patch_size), mode="bicubic",
                align_corners=False,
            )
        else:
            network_input = native
        probabilities = boundary_tta_batch(model, network_input, forward_with_boundary)
        if tile != patch_size:
            probabilities = F.interpolate(
                probabilities, size=(tile, tile), mode="bilinear",
                align_corners=False, antialias=False,
            )
        for (y, x), probability in zip(batch_positions, probabilities[:, 0]):
            score_sum[y:y + tile, x:x + tile] += probability * window
            weight_sum[y:y + tile, x:x + tile] += window

    if torch.any(weight_sum[:height, :width] <= 0):
        raise RuntimeError("fixed overlap tiling left uncovered image pixels")
    score = score_sum / weight_sum.clamp(min=1e-8)
    coverage = float((weight_sum[:height, :width] > 0).float().mean().item())
    return score[:height, :width].float().cpu().numpy(), coverage, len(tiles)


def average_precision(target: np.ndarray, score: np.ndarray) -> float | None:
    target = np.asarray(target, dtype=bool).ravel()
    score = np.asarray(score, dtype=np.float32).ravel()
    positive = int(target.sum())
    negative = int(target.size - positive)
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    ranked = target[order]
    true_positive = np.cumsum(ranked, dtype=np.int64)
    precision = true_positive / np.arange(1, len(ranked) + 1, dtype=np.float64)
    return float(precision[ranked].sum() / positive)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in records if row["average_precision"] is not None]
    if not valid:
        raise ValueError("no image has both positive and negative boundary pixels")

    def group_values(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
        by_group: dict[str, list[float]] = {}
        for row in rows:
            if row[metric] is not None:
                by_group.setdefault(row["group"], []).append(float(row[metric]))
        return {group: float(np.mean(values)) for group, values in by_group.items()}

    ap_by_group = group_values(valid, "average_precision")
    prevalence_by_group = group_values(valid, "prevalence")
    names = sorted(ap_by_group.keys() & prevalence_by_group.keys())
    ap_array = np.asarray([ap_by_group[name] for name in names])
    prevalence_array = np.asarray([prevalence_by_group[name] for name in names])
    delta_array = ap_array - prevalence_array
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    draws = rng.integers(0, len(names), size=(CONFIG["bootstrap_draws"], len(names)))
    ci_ap = np.percentile(ap_array[draws].mean(axis=1), [2.5, 97.5]).tolist()
    ci_delta = np.percentile(delta_array[draws].mean(axis=1), [2.5, 97.5]).tolist()

    seed_summary = {}
    for seed in (1, 2):
        seed_rows = [row for row in valid if row["seed"] == seed]
        seed_ap = group_values(seed_rows, "average_precision")
        seed_prev = group_values(seed_rows, "prevalence")
        seed_groups = sorted(seed_ap.keys() & seed_prev.keys())
        seed_summary[str(seed)] = {
            "images": len(seed_rows),
            "groups": len(seed_groups),
            "group_macro_ap_pct": float(np.mean([seed_ap[g] for g in seed_groups]) * 100),
            "group_macro_prevalence_pct": float(np.mean([seed_prev[g] for g in seed_groups]) * 100),
            "group_macro_ap_minus_prevalence_pp": float(
                np.mean([seed_ap[g] - seed_prev[g] for g in seed_groups]) * 100
            ),
        }

    fold_summary = {}
    for fold in (1, 2, 3, 4):
        fold_rows = [row for row in valid if row["fold"] == fold]
        fold_ap = group_values(fold_rows, "average_precision")
        fold_prev = group_values(fold_rows, "prevalence")
        fold_groups = sorted(fold_ap.keys() & fold_prev.keys())
        fold_summary[str(fold)] = {
            "images_seed_rows": len(fold_rows),
            "groups": len(fold_groups),
            "group_macro_ap_pct": float(np.mean([fold_ap[g] for g in fold_groups]) * 100),
            "group_macro_prevalence_pct": float(np.mean([fold_prev[g] for g in fold_groups]) * 100),
            "group_macro_ap_minus_prevalence_pp": float(
                np.mean([fold_ap[g] - fold_prev[g] for g in fold_groups]) * 100
            ),
        }

    signal = (
        ci_delta[0] > 0
        and all(value["group_macro_ap_minus_prevalence_pp"] > 0
                for value in seed_summary.values())
        and sum(value["group_macro_ap_minus_prevalence_pp"] > 0
                for value in fold_summary.values()) >= 3
    )
    return {
        "images_seed_rows": len(valid),
        "images": len({row["image_id"] for row in valid}),
        "specimen_groups": len(names),
        "group_macro_average_precision_pct": float(ap_array.mean() * 100),
        "group_macro_average_precision_group_bootstrap_95ci_pct":
            [float(value * 100) for value in ci_ap],
        "group_macro_prevalence_pct": float(prevalence_array.mean() * 100),
        "group_macro_ap_minus_prevalence_pp": float(delta_array.mean() * 100),
        "ap_minus_prevalence_group_bootstrap_95ci_pp":
            [float(value * 100) for value in ci_delta],
        "seed_summary": seed_summary,
        "fold_summary": fold_summary,
        "decision": "LOCALIZATION_SIGNAL" if signal else "NO_LOCALIZATION_SIGNAL",
        "decision_guard": (
            "This decision only determines whether to request separate Claude review; "
            "it does not authorize model training or official-valid access."
        ),
    }


def load_api(project_root: Path):
    source = project_root / "model336_hrnet_widthnorm_candidate"
    sys.path.insert(0, str(source))
    from komap_seg.checkpoint import load_model_and_preprocessing, resolve_preprocessing
    from komap_seg.dataset import KomapFullDataset, effective_input_scale
    from komap_seg.losses import gt_boundary_band
    from komap_seg.model import forward_with_boundary
    return (load_model_and_preprocessing, resolve_preprocessing, KomapFullDataset,
            effective_input_scale, gt_boundary_band, forward_with_boundary)


def run_probe(
    root: Path,
    lock: dict[str, Any],
    output_path: Path,
    device_name: str,
    batch_size: int,
) -> dict[str, Any]:
    if output_path.exists() or Path(str(output_path) + ".claim").exists():
        raise FileExistsError(f"refusing to overwrite result/claim at {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_root = root / "outputs/model392_widthnorm"
    output_root.mkdir(parents=True, exist_ok=True)
    lock_handle = (output_root / ".model392_oof.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_handle.close()
        raise RuntimeError("Model392 runner holds its OOF lock; refusing concurrent inference") from error

    claim_path = Path(str(output_path) + ".claim")
    claim_record = {
        "experiment": "Model392 boundary-head localization probe",
        "lock_sha256": lock["lock_sha256"],
        "claimed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "policy": "permanent one-shot claim; an interrupted attempt is not rerun",
    }
    write_exclusive(claim_path, claim_record)
    (load_model, resolve_preprocessing, FullDataset, effective_scale,
     make_boundary_target, forward_with_boundary) = load_api(root)
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this locked probe is intended for one P100 CUDA device")
    records: list[dict[str, Any]] = []
    total_tiles = 0
    total_rows = EXPECTED_IMAGES * 2
    completed = 0

    for seed in (1, 2):
        for fold in (1, 2, 3, 4):
            data_root = root / "data_model322_group_cv" / f"oof_fold{fold}"
            dataset = FullDataset(data_root / "images", data_root / "masks")
            checkpoint_path = output_root / f"seed{seed}" / f"fold{fold}" / "last.pt"
            model, saved_preprocessing = load_model(checkpoint_path, device)
            preprocessing = resolve_preprocessing(saved_preprocessing)
            args = json.loads(
                (output_root / f"seed{seed}" / f"fold{fold}" / "args.json")
                .read_text(encoding="utf-8")
            )
            if (preprocessing["patch_size"] != 512
                    or preprocessing["input_scale"] != 2.0
                    or preprocessing["target_width"] != args["target_width"]
                    or not preprocessing["normalize_width"]):
                raise ValueError(f"seed{seed}/fold{fold}: unsupported preprocessing metadata")
            if getattr(model, "boundary_head", None) is None:
                raise ValueError(f"seed{seed}/fold{fold}: checkpoint lacks boundary_head")

            for sample in dataset:
                image_id = sample["stem"]
                group = group_of(image_id)
                image = sample["image"]
                target_ids = sample["mask"]
                if target_ids is None:
                    raise ValueError(f"missing train OOF mask for {image_id}")
                scale = effective_scale(
                    preprocessing["input_scale"],
                    preprocessing["target_width"],
                    int(sample["width"]),
                )
                score, coverage, tile_count = predict_boundary_map(
                    model=model,
                    image=image,
                    patch_size=preprocessing["patch_size"],
                    input_scale=scale,
                    device=device,
                    batch_size=batch_size,
                    overlap_ratio=CONFIG["overlap_ratio"],
                    forward_with_boundary=forward_with_boundary,
                )
                target = make_boundary_target(
                    target_ids.unsqueeze(0).long(), num_classes=4, width=1
                )[0].cpu().numpy().astype(bool)
                if score.shape != target.shape:
                    raise ValueError(f"{image_id}: prediction/target shape mismatch")
                if not np.isfinite(score).all():
                    raise ValueError(f"{image_id}: boundary score contains non-finite values")
                ap = average_precision(target, score)
                prevalence = float(target.mean())
                records.append({
                    "image_id": image_id,
                    "group": group,
                    "fold": fold,
                    "seed": seed,
                    "height": int(sample["height"]),
                    "width": int(sample["width"]),
                    "tiles": tile_count,
                    "coverage_fraction": coverage,
                    "boundary_positive_pixels": int(target.sum()),
                    "boundary_negative_pixels": int(target.size - target.sum()),
                    "prevalence": prevalence,
                    "average_precision": ap,
                })
                total_tiles += tile_count
                completed += 1
                print(json.dumps({
                    "stage": "boundary_oof",
                    "seed": seed,
                    "fold": fold,
                    "image_id": image_id,
                    "progress": f"{completed}/{total_rows}",
                    "tiles_for_image": tile_count,
                    "coverage": round(coverage, 6),
                }), flush=True)
            del model
            torch.cuda.empty_cache()

    ids_by_seed = {
        seed: {row["image_id"] for row in records if row["seed"] == seed}
        for seed in (1, 2)
    }
    if len(records) != total_rows or ids_by_seed[1] != ids_by_seed[2]:
        raise ValueError("probe did not produce exactly-once predictions for all 70 images/seeds")
    if any(row["coverage_fraction"] != 1.0 for row in records):
        raise ValueError("one or more images had uncovered pixels")
    result = {
        "experiment": "Model392 auxiliary boundary-head localization probe",
        "revision": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "70 train images/59 specimen groups, grouped OOF; no official valid/test",
        "metric_scope": "boundary localization only; no segmentation mIoU computed",
        "lock_sha256": lock["lock_sha256"],
        "checkpoint_count": 8,
        "image_seed_rows": len(records),
        "total_tiles": total_tiles,
        "config": CONFIG,
        "summary": summarize_records(records),
        "per_image": records,
    }
    temporary = Path(str(output_path) + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing existing result temporary: {temporary}")
    write_exclusive(temporary, result)
    os.replace(temporary, output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--create-lock", action="store_true")
    parser.add_argument("--verify-lock", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--lock", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if sum((args.create_lock, args.verify_lock, args.run)) != 1:
        parser.error("choose exactly one of --create-lock, --verify-lock, or --run")
    root = args.project_root.resolve()
    lock_path = args.lock or (root / LOCK_REL)
    output_path = args.output or (root / OUTPUT_REL)
    if args.create_lock:
        lock = make_lock(root, lock_path)
        print(json.dumps({
            "status": "LOCKED",
            "lock": str(lock_path),
            "lock_sha256": lock["lock_sha256"],
            "oof_images": lock["snapshot"]["oof_images"],
            "specimen_groups": lock["snapshot"]["specimen_groups"],
            "checkpoint_count": 8,
        }, indent=2))
    elif args.verify_lock:
        lock = verify_lock(root, lock_path)
        print(json.dumps({
            "status": "LOCK_VERIFIED",
            "lock": str(lock_path),
            "lock_sha256": lock["lock_sha256"],
        }, indent=2))
    else:
        lock = verify_lock(root, lock_path)
        result = run_probe(root, lock, output_path, args.device, args.batch_size)
        print(json.dumps({
            "status": "COMPLETE",
            "output": str(output_path),
            "summary": result["summary"],
            "total_tiles": result["total_tiles"],
        }, indent=2), flush=True)


if __name__ == "__main__":
    main()
