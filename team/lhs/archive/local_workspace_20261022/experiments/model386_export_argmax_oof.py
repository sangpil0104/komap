#!/usr/bin/env python3
"""Lossless class-id export of train-OOF argmax maps for small CPU diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


GROUP_RE = re.compile(r"^(A\d+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(cv_root: Path, oof_root: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_root}")
    output_root.mkdir(parents=True)
    group_fold: dict[str, int] = {}
    manifest: dict[str, Any] = {
        "model": 386,
        "source_model": 374,
        "operation": "argmax over source OOF probabilities; no refit, resize, calibration, or postprocess",
        "folds": {},
    }

    for fold in (1, 2):
        mask_dir = cv_root / f"oof_fold{fold}" / "masks"
        source = oof_root / f"fold{fold}_oof" / "valid_probabilities.npz"
        if not mask_dir.is_dir() or not source.is_file():
            raise FileNotFoundError(f"missing fold{fold} train masks or Model374 OOF probabilities")
        masks = {p.name.removesuffix("_mask.png"): p for p in mask_dir.glob("*_mask.png")}
        if len(masks) != 18:
            raise ValueError(f"fold{fold}: expected 18 OOF masks, found {len(masks)}")
        prediction_path = output_root / f"fold{fold}_oof" / "valid_predictions.npz"
        prediction_path.parent.mkdir(parents=True)
        predictions: dict[str, np.ndarray] = {}
        with np.load(source, allow_pickle=False) as archive:
            if set(archive.files) != set(masks):
                raise ValueError(f"fold{fold}: source OOF keys do not exactly match train masks")
            for image_id in sorted(archive.files):
                match = GROUP_RE.match(image_id)
                if not match:
                    raise ValueError(f"cannot parse specimen group from {image_id}")
                group = match.group(1)
                if group in group_fold and group_fold[group] != fold:
                    raise ValueError(f"specimen group {group} crosses folds")
                group_fold[group] = fold
                probabilities = np.asarray(archive[image_id], dtype=np.float32)
                if probabilities.ndim != 3 or probabilities.shape[0] != 4:
                    raise ValueError(f"bad probability shape for {image_id}: {probabilities.shape}")
                if not np.isfinite(probabilities).all():
                    raise ValueError(f"non-finite source probabilities for {image_id}")
                if np.max(np.abs(probabilities.sum(axis=0) - 1.0)) > 0.03:
                    raise ValueError(f"unnormalized source probabilities for {image_id}")
                with Image.open(masks[image_id]) as mask_image:
                    mask_shape = (mask_image.height, mask_image.width)
                if probabilities.shape[1:] != mask_shape:
                    raise ValueError(f"probability/mask size mismatch for {image_id}")
                predictions[image_id] = probabilities.argmax(axis=0).astype(np.uint8)
        np.savez_compressed(prediction_path, **predictions)
        manifest["folds"][str(fold)] = {
            "images": len(predictions),
            "specimen_groups": sorted({GROUP_RE.match(name).group(1) for name in predictions}),
            "source_probabilities_sha256": sha256(source),
            "argmax_predictions_sha256": sha256(prediction_path),
            "prediction_archive": prediction_path.relative_to(output_root).as_posix(),
        }

    manifest["total_images"] = sum(record["images"] for record in manifest["folds"].values())
    manifest["total_specimen_groups"] = len(group_fold)
    if manifest["total_images"] != 36 or manifest["total_specimen_groups"] != 30:
        raise ValueError("expected the fixed 36-image/30-group folds1-2 train-OOF payload")
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.cv_root, args.oof_root, args.output_root), indent=2))


if __name__ == "__main__":
    main()
