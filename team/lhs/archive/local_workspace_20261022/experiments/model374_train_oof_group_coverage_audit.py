#!/usr/bin/env python3
"""Exploratory link between Model374 OOF group scores and train-group coverage."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import re
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

from valid_test_similarity_audit import FEATURES, image_features


PALETTE = np.asarray(
    ((153, 127, 76), (25, 76, 153), (76, 178, 76), (204, 204, 204)),
    dtype=np.uint8,
)
GROUP_RE = re.compile(r"^(A\d+)_")
STEP_RE = re.compile(r"step\d+")
INTENSITY_FEATURES = set(range(7))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mask_fractions(payload: bytes, image_id: str) -> np.ndarray:
    rgb = np.asarray(Image.open(BytesIO(payload)).convert("RGB"))
    ids = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        ids[np.all(rgb == color, axis=2)] = class_id
    if np.any(ids == 255):
        raise ValueError(f"unknown palette value in mask for {image_id}")
    return np.bincount(ids.ravel(), minlength=4).astype(np.float64) / ids.size


def load_rows(train_zip: Path, oof_json: Path) -> list[dict]:
    oof = json.loads(oof_json.read_text(encoding="utf-8"))
    score_by_id = {row["image_id"]: row for row in oof["per_image"]}
    if len(score_by_id) != len(oof["per_image"]):
        raise ValueError("duplicate OOF image IDs")

    with zipfile.ZipFile(train_zip) as archive:
        names = archive.namelist()
        image_members = {
            Path(name).name.removesuffix("_image.png"): name
            for name in names if "/images/" in name and name.endswith("_image.png")
        }
        mask_members = {
            Path(name).name.removesuffix("_mask.png"): name
            for name in names if "/masks/" in name and name.endswith("_mask.png")
        }
        if set(image_members) != set(mask_members) or set(image_members) != set(score_by_id):
            raise ValueError(
                "train image/mask IDs must exactly match Model374 OOF IDs: "
                f"images={len(image_members)}, masks={len(mask_members)}, oof={len(score_by_id)}"
            )

        rows = []
        with tempfile.TemporaryDirectory(prefix="model374-group-coverage-") as tempdir:
            tempdir_path = Path(tempdir)
            for image_id in sorted(image_members):
                match = GROUP_RE.match(image_id)
                step_match = STEP_RE.search(image_id)
                if not match or not step_match:
                    raise ValueError(f"unexpected image ID: {image_id}")
                image_payload = archive.read(image_members[image_id])
                image_path = tempdir_path / f"{image_id}_image.png"
                image_path.write_bytes(image_payload)
                feat = image_features(image_path)
                score = score_by_id[image_id]
                if score["specimen_group"] != match.group(1):
                    raise ValueError(f"group mismatch for {image_id}")
                rows.append({
                    "image_id": image_id,
                    "group": match.group(1),
                    "fold": int(score["fold"]),
                    "score": float(score["strict_image_miou_empty0"]),
                    "appearance": np.asarray(feat["features"], dtype=np.float64),
                    "step": step_match.group(0),
                    "resolution": f"{feat['width']}x{feat['height']}",
                    "class_fraction": mask_fractions(archive.read(mask_members[image_id]), image_id),
                })

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    group_folds = {group: {row["fold"] for row in group_rows} for group, group_rows in groups.items()}
    if any(len(folds) != 1 for folds in group_folds.values()):
        raise ValueError("a specimen group crosses OOF folds")
    if len(group_folds) != int(oof["specimen_groups"]):
        raise ValueError("OOF specimen-group count mismatch")
    return rows


def group_summary(rows: list[dict]) -> dict[str, dict]:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)
    result = {}
    for group, group_rows in sorted(by_group.items()):
        result[group] = {
            "fold": group_rows[0]["fold"],
            "images": len(group_rows),
            "score": float(np.mean([row["score"] for row in group_rows])),
            "appearance": np.mean([row["appearance"] for row in group_rows], axis=0),
            "class_fraction": np.mean([row["class_fraction"] for row in group_rows], axis=0),
            "step_hist": _hist([row["step"] for row in group_rows]),
            "resolution_hist": _hist([row["resolution"] for row in group_rows]),
        }
    return result


def _hist(values: list[str]) -> dict[str, float]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return {key: count / len(values) for key, count in sorted(counts.items())}


def _total_variation(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def _appearance_distances(train_groups: list[dict], heldout_groups: list[dict]) -> list[tuple[str, float]]:
    train_x = np.asarray([row["appearance"] for row in train_groups], dtype=np.float64)
    heldout_x = np.asarray([row["appearance"] for row in heldout_groups], dtype=np.float64)
    center = np.median(train_x, axis=0)
    q25, q75 = np.percentile(train_x, [25, 75], axis=0)
    scale = np.maximum((q75 - q25) / 1.349, np.std(train_x, axis=0) * 0.1)
    scale = np.maximum(scale, 1e-6)
    weights = np.asarray([
        1 / np.sqrt(7) if i in INTENSITY_FEATURES else 1 / np.sqrt(8)
        for i in range(len(FEATURES))
    ])
    distance = cdist((heldout_x - center) / scale * weights,
                     (train_x - center) / scale * weights)
    return [
        (train_groups[int(index)]["group"], float(distance[row_index, index]))
        for row_index, index in enumerate(distance.argmin(axis=1))
    ]


def _discrete_distances(train_groups: list[dict], heldout_groups: list[dict], key: str):
    result = []
    for heldout in heldout_groups:
        if key == "class_fraction":
            candidates = [
                (float(0.5 * np.abs(heldout[key] - train_group[key]).sum()), train_group["group"])
                for train_group in train_groups
            ]
        else:
            candidates = [
                (_total_variation(heldout[key], train_group[key]), train_group["group"])
                for train_group in train_groups
            ]
        distance, group = min(candidates)
        result.append((group, float(distance)))
    return result


def _rho(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    value = spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else float("nan")


def analyze(rows: list[dict], bootstrap_draws: int = 20_000, seed: int = 374) -> dict:
    groups = group_summary(rows)
    all_group_ids = sorted(groups)
    per_fold: dict[str, list[dict]] = {}
    distance_keys = ("appearance", "step_resolution", "class_mix")

    for fold in (1, 2, 3, 4):
        heldout = [groups[group] | {"group": group}
                   for group in all_group_ids if groups[group]["fold"] == fold]
        training = [groups[group] | {"group": group}
                    for group in all_group_ids if groups[group]["fold"] != fold]
        if not heldout or not training:
            raise ValueError(f"empty train/heldout group set for fold {fold}")
        appearance = _appearance_distances(training, heldout)
        step_distance = _discrete_distances(training, heldout, "step_hist")
        resolution_distance = _discrete_distances(training, heldout, "resolution_hist")
        class_distance = _discrete_distances(training, heldout, "class_fraction")
        fold_rows = []
        for i, row in enumerate(heldout):
            fold_rows.append({
                "group": row["group"],
                "images": row["images"],
                "image_mean_miou_empty0": row["score"],
                "nearest_train_group_appearance": appearance[i][0],
                "appearance_distance": appearance[i][1],
                "nearest_train_group_step": step_distance[i][0],
                "step_distribution_distance": step_distance[i][1],
                "nearest_train_group_resolution": resolution_distance[i][0],
                "resolution_distribution_distance": resolution_distance[i][1],
                "step_resolution_distance": 0.5 * (step_distance[i][1] + resolution_distance[i][1]),
                "nearest_train_group_class_mix": class_distance[i][0],
                "class_mix_distance": class_distance[i][1],
            })
        per_fold[str(fold)] = fold_rows

    distance_field = {
        "appearance": "appearance_distance",
        "step_resolution": None,
        "class_mix": "class_mix_distance",
    }
    relation_rows = {}
    rng = np.random.default_rng(seed)
    for name in distance_keys:
        fold_rhos = {}
        fold_samples = {}
        for fold in (1, 2, 3, 4):
            current = per_fold[str(fold)]
            if name == "appearance":
                x = np.asarray([row["appearance_distance"] for row in current])
            elif name == "step_resolution":
                x = np.asarray([row["step_resolution_distance"] for row in current])
            else:
                x = np.asarray([row[distance_field[name]] for row in current])
            y = np.asarray([row["image_mean_miou_empty0"] for row in current])
            rho_value = _rho(x, y)
            fold_rhos[str(fold)] = rho_value if np.isfinite(rho_value) else None
            fold_samples[fold] = (x, y)

        finite_fold_rhos = [value for value in fold_rhos.values() if value is not None]
        observed = float(np.mean(finite_fold_rhos)) if finite_fold_rhos else None
        boot = np.empty(bootstrap_draws, dtype=np.float64)
        for draw in range(bootstrap_draws):
            draw_rhos = []
            for fold in (1, 2, 3, 4):
                x, y = fold_samples[fold]
                selected = rng.integers(0, len(x), size=len(x))
                value = _rho(x[selected], y[selected])
                if np.isfinite(value):
                    draw_rhos.append(value)
            boot[draw] = np.mean(draw_rhos) if draw_rhos else np.nan
        finite = boot[np.isfinite(boot)]
        relation_rows[name] = {
            "direction": "negative means more distant heldout groups have lower mIoU",
            "mean_fold_spearman_rho": observed,
            "fold_spearman_rho": fold_rhos,
            "group_bootstrap_95ci_conditional_on_fixed_oof":
                np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else [None, None],
            "valid_bootstrap_draws": int(len(finite)),
        }

    return {
        "model": 374,
        "scope": "exploratory training-only join of fixed grouped OOF scores and train ZIP image/mask features",
        "images": len(rows),
        "specimen_groups": len(groups),
        "metric": "per-group mean of per-image four-class mIoU; empty union=0",
        "distance_definitions": {
            "appearance": "nearest train-group mean of the 15 fixed grayscale intensity/texture features from valid_test_similarity_audit.py; robust scaling fitted on train groups within each outer fold",
            "step_resolution": "mean of step-histogram and resolution-histogram total-variation distances; nearest train group",
            "class_mix": "nearest train-group total-variation distance between four GT class-pixel-fraction vectors; descriptive case-mix only",
        },
        "folds": {key: {"heldout_groups": len(value), "rows": value}
                  for key, value in per_fold.items()},
        "distance_score_associations": relation_rows,
        "interpretation_guard": (
            "Post-hoc exploratory association, not causal evidence that more training groups improve mIoU. "
            "The fixed OOF predictions and historical score summaries have been inspected before this audit. "
            "Bootstrap intervals condition on the fitted OOF models, omit training/subset variance, and folds have overlapping training sets. "
            "No result here is an automatic training GO/NO-GO."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-zip", type=Path, required=True)
    parser.add_argument("--oof-json", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=374)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.bootstrap_draws < 100:
        parser.error("--bootstrap-draws must be >=100")
    result = analyze(load_rows(args.train_zip, args.oof_json), args.bootstrap_draws, args.seed)
    result["source_sha256"] = {
        "train_zip": sha256(args.train_zip),
        "model374_oof_summary": sha256(args.oof_json),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"distance_score_associations": result["distance_score_associations"],
                      "images": result["images"], "specimen_groups": result["specimen_groups"],
                      "interpretation_guard": result["interpretation_guard"]},
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
