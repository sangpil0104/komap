#!/usr/bin/env python3
"""Compare tiny Eutectic-Si component recall in Model392 parent vs Model397 blend."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

import model396_fixed_architecture_oof as m396
from model404_component_recall_preflight import (
    B4_ROOT,
    MODEL392_ROOT,
    component_rows,
    sha256,
    summarize_bin,
)


GROUP_RE = re.compile(r"^(A\d+)_")
MODEL397_LOCK = "experiments/results/model397_fixed_b4_model392_blend_lock.json"
MODEL397_RESULT = "experiments/results/model397_fixed_b4_model392_blend_oof.json"
AREA_BINS = ("1_to_lt64", "64_to_lt256")


def run(root: Path, draws: int) -> dict:
    masks_by_fold = m396.mask_files(root)
    arms = {name: [] for name in ("model392_seed1", "model397_fixed_blend")}
    used: dict[str, str] = {}
    images = 0
    for fold, masks in masks_by_fold.items():
        ids = set(masks)
        paths = {
            "b4": root / B4_ROOT / f"fold{fold}_oof/probabilities.npz",
            "seed1": root / MODEL392_ROOT / f"seed1/fold{fold}_oof/valid_probabilities.npz",
            "seed2": root / MODEL392_ROOT / f"seed2/fold{fold}_oof/valid_probabilities.npz",
        }
        archives = {name: np.load(path, allow_pickle=False) for name, path in paths.items()}
        try:
            if any(set(archive.files) != ids for archive in archives.values()):
                raise ValueError(f"OOF inventory mismatch for fold {fold}")
            for path in paths.values():
                used[str(path.relative_to(root))] = sha256(path)
            for image_id, mask_path in sorted(masks.items()):
                match = GROUP_RE.match(image_id)
                if match is None:
                    raise ValueError(f"cannot parse group for {image_id}")
                truth = m396.mask_to_ids(mask_path)
                p1 = np.asarray(archives["seed1"][image_id], dtype=np.float32)
                p2 = np.asarray(archives["seed2"][image_id], dtype=np.float32)
                pb = np.asarray(archives["b4"][image_id], dtype=np.float32)
                if any(p.shape != (4, *truth.shape) or not np.isfinite(p).all()
                       for p in (p1, p2, pb)):
                    raise ValueError(f"invalid probability archive for {image_id}")
                predictions = {
                    "model392_seed1": p1.argmax(axis=0),
                    "model397_fixed_blend": (0.50 * pb + 0.25 * p1 + 0.25 * p2).argmax(axis=0),
                }
                for arm, pred in predictions.items():
                    arms[arm].extend(component_rows(
                        truth, pred, fold=fold, group=match.group(1), image_id=image_id
                    ))
                used[str(mask_path.relative_to(root))] = sha256(mask_path)
                images += 1
        finally:
            for archive in archives.values():
                archive.close()

    bootstrap_rng = np.random.default_rng(404_641)
    metrics = {}
    for arm, rows in arms.items():
        arm_rng = np.random.default_rng(int(bootstrap_rng.integers(0, 2**32 - 1)))
        metrics[arm] = {
            name: summarize_bin(rows, name, arm_rng, draws)
            for name in AREA_BINS
        }
    for path in (MODEL397_LOCK, MODEL397_RESULT):
        used[path] = sha256(root / path)
    return {
        "experiment": "Model404 parent-component alignment diagnostic",
        "metric_scope": "native-grid Eutectic-Si component recall only; no mIoU calculation",
        "arms": {
            "model392_seed1": "the exact parent model to be fine-tuned by Model404",
            "model397_fixed_blend": "0.50 Model329 B4 + 0.25 Model392 seed1 + 0.25 Model392 seed2",
        },
        "images": images,
        "groups": 59,
        "valid_read": False,
        "test_read": False,
        "bootstrap_draws": draws,
        "bootstrap_seed": 404641,
        "component_recall_by_arm": metrics,
        "input_sha256": used,
        "audit_script_sha256": sha256(Path(__file__).resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.draws < 1_000:
        raise ValueError("need at least 1000 group-bootstrap draws")
    if args.out.exists():
        raise FileExistsError(args.out)
    result = run(args.root.resolve(), args.draws)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "result": str(args.out),
        "images": result["images"],
        "valid_read": False,
        "test_read": False,
        "tiny_bin": {arm: result["component_recall_by_arm"][arm]["1_to_lt64"]
                     for arm in result["component_recall_by_arm"]},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
