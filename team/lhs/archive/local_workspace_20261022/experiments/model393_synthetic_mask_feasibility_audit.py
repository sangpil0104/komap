#!/usr/bin/env python3
"""CPU-only morphology inventory of training masks for Model393 feasibility."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
from pathlib import Path
import re
import zipfile

import cv2
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = REPO_ROOT / "train.zip"
LOCK_PATH = REPO_ROOT / "experiments/results/model393_synthetic_mask_feasibility_lock.json"
DEFAULT_OUTPUT = REPO_ROOT / "experiments/results/model393_synthetic_mask_feasibility.json"
GROUP_RE = re.compile(r"^(A\d+)_")
CLASS_NAMES = ("Al", "Al3Ni", "Primary_Si", "Eutectic_Si")
PALETTE = np.asarray(
    ((153, 127, 76), (25, 76, 153), (76, 178, 76), (204, 204, 204)),
    dtype=np.uint8,
)
EXPECTED_IMAGES = 70
LAGS = (1, 2, 4, 8, 16, 32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_mask(payload: bytes, member: str) -> np.ndarray:
    rgb = np.asarray(Image.open(BytesIO(payload)).convert("RGB"), dtype=np.uint8)
    labels = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        labels[np.all(rgb == color, axis=2)] = class_id
    if np.any(labels == 255):
        unknown = int(np.count_nonzero(labels == 255))
        raise ValueError(f"{member} has {unknown} off-palette pixels")
    return labels


def component_summary(mask: np.ndarray) -> dict:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    if not len(areas):
        return {"components": 0, "area_px": [], "bbox_aspect_ratio": []}
    widths = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float64)
    heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float64)
    aspect = np.maximum(widths, heights) / np.maximum(np.minimum(widths, heights), 1.0)
    return {
        "components": int(count - 1),
        "area_px": areas.tolist(),
        "bbox_aspect_ratio": aspect.tolist(),
        "largest_component_fraction_of_class": float(areas.max() / areas.sum()),
    }


def distance_width_summary(mask: np.ndarray) -> dict:
    if not mask.any():
        return {"pixels": 0, "width_proxy_px_q": [None, None, None, None]}
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    # Twice the distance to the nearest label boundary is only a 2-D width
    # proxy; it is not a metallurgical length measurement.
    values = 2.0 * distance[mask]
    return {
        "pixels": int(mask.sum()),
        "width_proxy_px_q": [float(x) for x in np.quantile(values, (0.5, 0.9, 0.95, 0.99))],
    }


def adjacency_counts(labels: np.ndarray) -> np.ndarray:
    result = np.zeros((4, 4), dtype=np.int64)
    for left, right in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        changed = left != right
        if not changed.any():
            continue
        lo = np.minimum(left[changed], right[changed]).astype(np.int64)
        hi = np.maximum(left[changed], right[changed]).astype(np.int64)
        pairs = np.bincount(lo * 4 + hi, minlength=16).reshape(4, 4)
        upper = np.triu(pairs, 1)
        result += upper + upper.T
    return result


def two_point(mask: np.ndarray) -> dict[str, float | None]:
    indicator = mask.astype(np.float64)
    height, width = mask.shape
    output: dict[str, float | None] = {}
    for lag in LAGS:
        if width > lag:
            output[f"x{lag}"] = float(np.mean(indicator[:, :-lag] * indicator[:, lag:]))
        else:
            output[f"x{lag}"] = None
        if height > lag:
            output[f"y{lag}"] = float(np.mean(indicator[:-lag, :] * indicator[lag:, :]))
        else:
            output[f"y{lag}"] = None
    return output


def percentile(values: list[float]) -> list[float] | None:
    if not values:
        return None
    return [float(x) for x in np.quantile(np.asarray(values, dtype=np.float64), (0.1, 0.5, 0.9))]


def summarize_archive(archive_path: Path) -> dict:
    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist()
                   if name.startswith("train/masks/") and name.endswith("_mask.png")]
        if len(members) != EXPECTED_IMAGES:
            raise ValueError(f"expected {EXPECTED_IMAGES} training masks, found {len(members)}")
        rows = []
        seen_ids: set[str] = set()
        for member in sorted(members):
            image_id = Path(member).name.removesuffix("_mask.png")
            match = GROUP_RE.match(image_id)
            if not match:
                raise ValueError(f"cannot parse specimen group from {member}")
            if image_id in seen_ids:
                raise ValueError(f"duplicate image ID: {image_id}")
            seen_ids.add(image_id)
            labels = decode_mask(archive.read(member), member)
            pixels = labels.size
            fractions = np.bincount(labels.ravel(), minlength=4).astype(np.float64) / pixels
            components = {}
            widths = {}
            autocorrelations = {}
            for class_id, class_name in enumerate(CLASS_NAMES):
                binary = labels == class_id
                components[class_name] = component_summary(binary)
                widths[class_name] = distance_width_summary(binary)
                autocorrelations[class_name] = two_point(binary)
            adj = adjacency_counts(labels)
            adj_total = int(adj.sum())
            rows.append({
                "image_id": image_id,
                "specimen_group": match.group(1),
                "height": int(labels.shape[0]),
                "width": int(labels.shape[1]),
                "phase_fraction": fractions.tolist(),
                "present_classes": [CLASS_NAMES[i] for i, f in enumerate(fractions) if f > 0],
                "component_geometry": components,
                "distance_width_proxy": widths,
                "boundary_contact_share": (adj / adj_total).tolist() if adj_total else np.zeros((4, 4)).tolist(),
                "two_point_probability": autocorrelations,
            })

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["specimen_group"]].append(row)
    if len(groups) != 59:
        raise ValueError(f"expected 59 specimen groups, found {len(groups)}")

    group_rows = []
    for group, image_rows in sorted(groups.items()):
        class_fraction = np.mean([r["phase_fraction"] for r in image_rows], axis=0)
        present = [
            any(r["phase_fraction"][class_id] > 0 for r in image_rows)
            for class_id in range(4)
        ]
        group_rows.append({
            "specimen_group": group,
            "images": len(image_rows),
            "mean_phase_fraction": class_fraction.tolist(),
            "phase_present": present,
        })

    class_group_support = {
        CLASS_NAMES[class_id]: sum(item["phase_present"][class_id] for item in group_rows)
        for class_id in range(4)
    }
    if min(class_group_support.values()) < 5:
        support_status = "NO_GO: at least one phase occurs in fewer than five specimen groups"
    else:
        support_status = "support threshold passed; morphology/rendering feasibility remains undecided"

    fraction_summary = {}
    component_summary_all = {}
    width_summary_all = {}
    autocorrelation_summary = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        fraction_summary[class_name] = {
            "image_q10_q50_q90": percentile([r["phase_fraction"][class_id] for r in rows]),
            "equal_group_mean": float(np.mean([g["mean_phase_fraction"][class_id] for g in group_rows])),
            "group_q10_q50_q90": percentile([g["mean_phase_fraction"][class_id] for g in group_rows]),
            "groups_present": class_group_support[class_name],
        }
        component_summary_all[class_name] = {
            "image_median_component_count_q10_q50_q90": percentile([
                r["component_geometry"][class_name]["components"] for r in rows
            ]),
            "component_area_px_q10_q50_q90": percentile([
                float(x) for r in rows for x in r["component_geometry"][class_name]["area_px"]
            ]),
            "component_bbox_aspect_q10_q50_q90": percentile([
                float(x) for r in rows for x in r["component_geometry"][class_name]["bbox_aspect_ratio"]
            ]),
            "largest_component_fraction_q10_q50_q90": percentile([
                float(r["component_geometry"][class_name]["largest_component_fraction_of_class"])
                for r in rows
                if r["component_geometry"][class_name]["components"] > 0
            ]),
        }
        width_summary_all[class_name] = {
            "per_image_pixel_width_proxy_q50_px_q10_q50_q90": percentile([
                float(r["distance_width_proxy"][class_name]["width_proxy_px_q"][0])
                for r in rows if r["distance_width_proxy"][class_name]["pixels"] > 0
            ]),
            "per_image_pixel_width_proxy_q90_px_q10_q50_q90": percentile([
                float(r["distance_width_proxy"][class_name]["width_proxy_px_q"][1])
                for r in rows if r["distance_width_proxy"][class_name]["pixels"] > 0
            ]),
        }
        autocorrelation_summary[class_name] = {
            key: percentile([float(r["two_point_probability"][class_name][key]) for r in rows
                             if r["two_point_probability"][class_name][key] is not None])
            for key in [f"{axis}{lag}" for lag in LAGS for axis in ("x", "y")]
        }

    contact = np.mean([np.asarray(r["boundary_contact_share"]) for r in rows], axis=0)
    resolution_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        resolution_counts[f"{row['width']}x{row['height']}"] += 1

    return {
        "model": 393,
        "stage": "synthetic_mask_feasibility_screen",
        "scope": "train.zip masks only; no image pixels, valid/test labels, OOF predictions, model checkpoints, or model scores",
        "images": len(rows),
        "specimen_groups": len(groups),
        "resolution_counts": dict(sorted(resolution_counts.items())),
        "class_order": list(CLASS_NAMES),
        "class_group_support": class_group_support,
        "support_gate": support_status,
        "phase_fraction": fraction_summary,
        "connected_component_geometry": component_summary_all,
        "distance_transform_width_proxies": width_summary_all,
        "mean_per_image_boundary_contact_share": {
            CLASS_NAMES[i]: {CLASS_NAMES[j]: float(contact[i, j]) for j in range(4)}
            for i in range(4)
        },
        "two_point_probability_q10_q50_q90": autocorrelation_summary,
        "per_specimen_group": group_rows,
        "per_image": rows,
        "interpretation_guard": (
            "All values are mask-geometry descriptors, not segmentation scores, physical grain measurements, "
            "annotation-noise estimates, or proof that a synthetic generator can improve mIoU."
        ),
        "audited_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


def verify_lock(archive_path: Path) -> dict:
    if not LOCK_PATH.is_file():
        raise FileNotFoundError(f"Model393 pre-analysis lock missing: {LOCK_PATH}")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("train_zip_sha256") != sha256(archive_path):
        raise ValueError("train.zip differs from the pre-analysis locked archive")
    for relative, expected in lock["sources_sha256"].items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Model393 pre-analysis source hash mismatch: {relative}")
    return lock


def run(archive_path: Path, output_path: Path) -> dict:
    lock = verify_lock(archive_path)
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite Model393 audit output: {output_path}")
    if output_path != DEFAULT_OUTPUT.resolve():
        raise ValueError(f"Model393 audit output path is fixed at {DEFAULT_OUTPUT}")
    result = summarize_archive(archive_path)
    result["lock_sha256"] = sha256(LOCK_PATH)
    result["source_train_zip_sha256"] = lock["train_zip_sha256"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
    temporary.replace(output_path)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("per_image", "per_specimen_group")}, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-zip", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.train_zip.resolve(), args.output)


if __name__ == "__main__":
    main()
