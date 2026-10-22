#!/usr/bin/env python3
"""Fixed-weight, train-OOF-only half-blend diagnostic for Model374 fold 1."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

MODEL329_REFERENCE_SHA256 = "539bc1983bffb3ff55303f6865fc9a2094edc5dea9910551a481005fc1a01837"


def archive_probability(archive, name: str) -> np.ndarray:
    for key in (name, f"{name}_image"):
        if key in archive:
            return np.asarray(archive[key], dtype=np.float32)
    raise KeyError(f"{name} (or {name}_image) missing from probability archive")


def image_class_iou(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    confusion = np.bincount(target.astype(np.int64).ravel() * 4 +
                            prediction.astype(np.int64).ravel(), minlength=16).reshape(4, 4)
    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(axis=0) + confusion.sum(axis=1) - intersection
    return np.divide(intersection, union, out=np.zeros(4, dtype=np.float64), where=union > 0)


def model329_to_model374_order(probability: np.ndarray) -> np.ndarray:
    """Validate Model329 B4 archive's canonical [Al, Al3Ni, Primary, Eutectic] order."""
    if probability.ndim != 3 or probability.shape[0] != 4:
        raise ValueError(f"expected Model329 4xHxW probabilities, got {probability.shape}")
    return probability


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-data", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True,
                        help="Model374 fold-1 held-out probability archive")
    parser.add_argument("--reference", type=Path, required=True,
                        help="Model329 exactly-once grouped train-OOF probabilities")
    parser.add_argument("--common-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite preregistered gate result: {args.output}")
    reference_sha = sha256(args.reference)
    if reference_sha != MODEL329_REFERENCE_SHA256:
        raise ValueError(f"Model329 OOF reference archive changed: {reference_sha}")
    sys.path.insert(0, str(args.common_dir))
    from common import list_split, load_mask  # pylint: disable=import-outside-toplevel

    pairs = list_split(str(args.cv_data), "oof_fold1")
    if len(pairs) != 18:
        raise ValueError(f"expected 18 fold-1 held-out images; found {len(pairs)}")
    rows = []
    with np.load(args.candidate) as candidate_archive, np.load(args.reference) as reference_archive:
        expected = {Path(image_path).name.replace("_image.png", "")
                    for image_path, _ in pairs}
        candidate_keys = {key.removesuffix("_image") for key in candidate_archive.files}
        reference_keys = {key.removesuffix("_image") for key in reference_archive.files}
        if candidate_keys != expected:
            raise ValueError("Model374 archive keys do not exactly match fold-1 held-out images")
        if not expected <= reference_keys:
            raise ValueError("Model329 archive is missing fold-1 held-out reference images")
        for image_path, mask_path in pairs:
            name = Path(image_path).name.replace("_image.png", "")
            candidate = archive_probability(candidate_archive, name)
            reference_raw = archive_probability(reference_archive, name)
            reference = model329_to_model374_order(reference_raw)
            target = load_mask(mask_path)
            if (candidate.shape != reference.shape or candidate.ndim != 3
                    or candidate.shape[0] != 4 or candidate.shape[1:] != target.shape):
                raise ValueError(f"probability/target shape mismatch for {name}")
            for label, probability in (("Model374", candidate), ("Model329", reference)):
                if (not np.isfinite(probability).all() or np.any(probability < -1e-5)
                        or not np.allclose(probability.sum(axis=0), 1.0, atol=3e-2)):
                    raise ValueError(f"invalid {label} probability map for {name}")
            candidate_iou = image_class_iou(candidate.argmax(axis=0), target)
            reference_iou = image_class_iou(reference.argmax(axis=0), target)
            blended_iou = image_class_iou((0.5 * candidate + 0.5 * reference).argmax(axis=0), target)
            rows.append({
                "image": name,
                "group": name.split("_", 1)[0],
                "model374_image_miou_empty0": float(candidate_iou.mean()),
                "model329_image_miou_empty0": float(reference_iou.mean()),
                "fixed_half_blend_image_miou_empty0": float(blended_iou.mean()),
                "fixed_half_blend_delta_vs_model329_pp": float(
                    (blended_iou.mean() - reference_iou.mean()) * 100.0),
            })

    deltas = np.asarray([row["fixed_half_blend_delta_vs_model329_pp"] for row in rows])
    mean_delta = float(deltas.mean())
    result = {
        "audit": "pre-registered fixed half-blend diagnostic; not a continuation gate",
        "metric": "mean over fold-1 held-out images of four class IoUs; empty union=0",
        "candidate_model": 374,
        "reference_model": 329,
        "reference_split": "exactly-once specimen-group train OOF, same fold-1 held-out images",
        "blend": "fixed 0.5 Model374 + 0.5 Model329 probabilities in identical canonical order, then argmax",
        "model374_class_order": ["Al", "Al3Ni", "Primary_Si", "Eutectic_Si"],
        "model329_archive_class_order": ["Al", "Al3Ni", "Primary_Si", "Eutectic_Si"],
        "model329_to_canonical_channel_indices": [0, 1, 2, 3],
        "pinned_model329_reference_sha256": MODEL329_REFERENCE_SHA256,
        "image_count": len(rows),
        "specimen_group_count": len({row["group"] for row in rows}),
        "mean_delta_pp_vs_model329": mean_delta,
        "positive_images": int((deltas > 0).sum()),
        "interpretation": "one fixed comparison only; 0.5 is not presumed optimal",
        "candidate_archive": str(args.candidate),
        "candidate_archive_sha256": sha256(args.candidate),
        "reference_archive": str(args.reference),
        "reference_archive_sha256": reference_sha,
        "per_image": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "image_count", "specimen_group_count", "mean_delta_pp_vs_model329",
        "positive_images", "candidate_archive_sha256", "reference_archive_sha256")},
        indent=2), flush=True)


if __name__ == "__main__":
    main()
