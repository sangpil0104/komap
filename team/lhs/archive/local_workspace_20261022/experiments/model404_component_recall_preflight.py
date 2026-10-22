#!/usr/bin/env python3
"""Recheck Model397 OOF small-component error using recall thresholds, not any-FN."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import label

import model396_fixed_architecture_oof as m396


ROOT = Path(__file__).resolve().parents[1]
CV_ROOT = "data_model322_group_cv"
B4_ROOT = "outputs/model329_fullstats_width4000_uniform_group_oof"
MODEL392_ROOT = "outputs/model392_widthnorm"
GROUP_RE = re.compile(r"^(A\d+)_")
AREA_BINS = ((1, 64, "1_to_lt64"), (64, 256, "64_to_lt256"),
             (256, 1024, "256_to_lt1024"), (1024, 4096, "1024_to_lt4096"),
             (4096, None, "4096_or_more"))
PALETTE = np.asarray(((153, 127, 76), (25, 76, 153), (76, 178, 76), (204, 204, 204)), dtype=np.uint8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def component_rows(truth: np.ndarray, prediction: np.ndarray, *, fold: int,
                   group: str, image_id: str) -> list[dict]:
    labels, count = label(truth == 3, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    rows = []
    for component_id in range(1, count + 1):
        area = int(areas[component_id])
        low, high, name = next(
            spec for spec in AREA_BINS if area >= spec[0] and (spec[1] is None or area < spec[1])
        )
        del low, high
        selected = labels == component_id
        recall = float(np.count_nonzero(prediction[selected] == 3) / area)
        rows.append({
            "fold": fold,
            "group": group,
            "image_id": image_id,
            "bin": name,
            "area": area,
            "recall": recall,
        })
    return rows


def summarize_bin(rows: list[dict], name: str, rng: np.random.Generator,
                  draws: int) -> dict:
    selected = [row for row in rows if row["bin"] == name]
    if not selected:
        return {"components": 0, "groups": 0, "images": 0}
    recall = np.asarray([row["recall"] for row in selected], dtype=np.float64)
    support = np.asarray([row["area"] for row in selected], dtype=np.float64)
    group_fold: dict[int, dict[str, list[dict]]] = {fold: {} for fold in range(1, 5)}
    for row in selected:
        group_fold[row["fold"]].setdefault(row["group"], []).append(row)
    hit50_draws = np.zeros(draws, dtype=np.float64)
    fullmiss_draws = np.zeros(draws, dtype=np.float64)
    draw_components = np.zeros(draws, dtype=np.float64)
    for fold, groups in group_fold.items():
        if not groups:
            continue
        names = sorted(groups)
        samples = rng.integers(0, len(names), size=(draws, len(names)))
        n = np.asarray([len(groups[group]) for group in names], dtype=np.float64)
        hit50 = np.asarray([sum(item["recall"] >= 0.50 for item in groups[group]) for group in names], dtype=np.float64)
        fullmiss = np.asarray([sum(item["recall"] == 0.0 for item in groups[group]) for group in names], dtype=np.float64)
        draw_components += (n[samples]).sum(axis=1)
        hit50_draws += (hit50[samples]).sum(axis=1)
        fullmiss_draws += (fullmiss[samples]).sum(axis=1)
    hit50_ci = np.quantile(hit50_draws / draw_components, [0.025, 0.975])
    fullmiss_ci = np.quantile(fullmiss_draws / draw_components, [0.025, 0.975])
    folds = {}
    for fold in range(1, 5):
        fold_rows = [row for row in selected if row["fold"] == fold]
        fold_recall = np.asarray([row["recall"] for row in fold_rows], dtype=np.float64)
        folds[str(fold)] = {
            "components": len(fold_rows),
            "groups": len({row["group"] for row in fold_rows}),
            "mean_recall_percent": float(100 * fold_recall.mean()) if len(fold_rows) else None,
            "hit_rate_recall_ge_50_percent": float(100 * np.mean(fold_recall >= 0.50)) if len(fold_rows) else None,
        }
    return {
        "components": int(len(selected)),
        "groups": int(len({row["group"] for row in selected})),
        "images": int(len({row["image_id"] for row in selected})),
        "mean_component_recall_percent": float(100 * recall.mean()),
        "median_component_recall_percent": float(100 * np.median(recall)),
        "p10_component_recall_percent": float(100 * np.quantile(recall, 0.10)),
        "component_any_fn_rate_percent": float(100 * np.mean(recall < 1.0)),
        "component_majority_miss_rate_recall_lt_50_percent": float(100 * np.mean(recall < 0.50)),
        "component_complete_miss_rate_recall_eq_0_percent": float(100 * np.mean(recall == 0.0)),
        "component_hit_rate_recall_ge_50_group_bootstrap_95ci_percent": [float(x * 100) for x in hit50_ci],
        "complete_miss_rate_group_bootstrap_95ci_percent": [float(x * 100) for x in fullmiss_ci],
        "pixel_support_weighted_recall_percent": float(100 * np.sum(recall * support) / support.sum()),
        "folds": folds,
    }


def run(root: Path, draws: int) -> dict:
    masks_by_fold = m396.mask_files(root)
    all_rows: list[dict] = []
    used = {}
    total_images = 0
    for fold, masks in masks_by_fold.items():
        ids = set(masks)
        paths = {
            "b4": root / B4_ROOT / f"fold{fold}_oof/probabilities.npz",
            "hrnet_seed1": root / MODEL392_ROOT / f"seed1/fold{fold}_oof/valid_probabilities.npz",
            "hrnet_seed2": root / MODEL392_ROOT / f"seed2/fold{fold}_oof/valid_probabilities.npz",
        }
        archives = {name: np.load(path, allow_pickle=False) for name, path in paths.items()}
        try:
            if any(set(archive.files) != ids for archive in archives.values()):
                raise ValueError(f"OOF IDs do not exactly match fixed heldout fold {fold}")
            for name, path in paths.items():
                used[str(path.relative_to(root))] = sha256(path)
            for image_id, mask_path in sorted(masks.items()):
                group_match = GROUP_RE.match(image_id)
                if group_match is None:
                    raise ValueError(f"cannot parse specimen group for {image_id}")
                truth = m396.mask_to_ids(mask_path)
                p_b4 = np.asarray(archives["b4"][image_id], dtype=np.float32)
                p_h1 = np.asarray(archives["hrnet_seed1"][image_id], dtype=np.float32)
                p_h2 = np.asarray(archives["hrnet_seed2"][image_id], dtype=np.float32)
                if any(p.shape != (4, *truth.shape) or not np.isfinite(p).all()
                       for p in (p_b4, p_h1, p_h2)):
                    raise ValueError(f"invalid OOF probability tensor for {image_id}")
                prediction = (0.50 * p_b4 + 0.25 * p_h1 + 0.25 * p_h2).argmax(axis=0)
                all_rows.extend(component_rows(
                    truth, prediction, fold=fold, group=group_match.group(1), image_id=image_id
                ))
                total_images += 1
        finally:
            for archive in archives.values():
                archive.close()
    rng = np.random.default_rng(404_502)
    bins = {name: summarize_bin(all_rows, name, rng, draws) for _, _, name in AREA_BINS}
    return {
        "experiment": "Model404 preflight: 50%-recall recheck of Model397 Eutectic-Si OOF components",
        "prediction": "fixed 0.50 Model329 B4 + 0.25 Model392 seed1 + 0.25 Model392 seed2 probabilities; argmax",
        "component_definition": "native-grid GT class 3; 8-connectivity",
        "hit_definition": "component recall >= 50%; separate any-FN and complete-miss definitions",
        "images": total_images,
        "groups": 59,
        "valid_read": False,
        "test_read": False,
        "mIoU_tuned_or_scored": False,
        "bootstrap_draws": draws,
        "bootstrap_seed": 404502,
        "input_sha256": used,
        "area_bins": bins,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.draws < 1000:
        raise ValueError("at least 1000 group-bootstrap draws are required")
    result = run(args.root.resolve(), args.draws)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "result": str(args.out),
        "valid_read": False,
        "test_read": False,
        "images": result["images"],
        "target_bins": {key: result["area_bins"][key] for key in ("1_to_lt64", "64_to_lt256")},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
