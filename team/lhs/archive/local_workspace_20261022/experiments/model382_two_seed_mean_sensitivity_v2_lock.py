#!/usr/bin/env python3
"""Freeze fixed Model382 train-OOF masks before seed2 OOF outputs exist."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GROUP_RE = re.compile(r"^(A\d+)_")
STEP_RE = re.compile(r"_step(\d+)_")
LOCKED_FILES = (
    "experiments/model382_two_seed_mean_sensitivity_v2_addendum.md",
    "experiments/model382_two_seed_mean_sensitivity_v2.py",
    "experiments/model382_two_seed_mean_sensitivity_v2_smoke.py",
    "experiments/model382_two_seed_mean_sensitivity_v2_lock.py",
    "experiments/model382_two_seed_mean_sensitivity_preregistration.md",
    "experiments/model382_two_seed_mean_sensitivity.py",
    "experiments/model382_two_seed_mean_sensitivity_smoke.py",
    "experiments/model382_two_seed_mean_sensitivity_lock.py",
    "experiments/model382_oof_ensemble_compare.py",
    "experiments/model382_seed_count_control_preregistration.md",
    "experiments/model382_five_model_ensemble_addendum.md",
    "experiments/results/model382_e5_preregistration_lock.json",
    "experiments/results/model382_two_seed_mean_preregistration_lock.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory_masks(cv_root: Path) -> tuple[dict[str, str], dict[str, object]]:
    hashes: dict[str, str] = {}
    group_fold: dict[str, int] = {}
    all_ids: set[str] = set()
    step4_count = 0
    a71_count = 0
    fold_counts: dict[str, int] = {}

    for fold in range(1, 5):
        mask_dir = cv_root / f"oof_fold{fold}" / "masks"
        paths = sorted(mask_dir.glob("*_mask.png"))
        if not paths:
            raise FileNotFoundError(f"no fixed OOF masks in {mask_dir}")
        fold_counts[str(fold)] = len(paths)
        for path in paths:
            image_id = path.name.removesuffix("_mask.png")
            group_match = GROUP_RE.match(image_id)
            step_match = STEP_RE.search(image_id)
            if not group_match or not step_match:
                raise ValueError(f"cannot parse group/step from fixed mask ID: {image_id}")
            if image_id in all_ids:
                raise ValueError(f"duplicate fixed mask ID across folds: {image_id}")
            all_ids.add(image_id)
            group = group_match.group(1)
            prior_fold = group_fold.setdefault(group, fold)
            if prior_fold != fold:
                raise ValueError(f"specimen group crosses folds: {group}")
            step4_count += int(step_match.group(1) == "4")
            a71_count += int(group == "A71")
            relative = path.relative_to(cv_root).as_posix()
            hashes[relative] = sha256(path)

    if len(all_ids) != 70 or len(group_fold) != 59:
        raise ValueError(f"expected 70 images/59 groups, got {len(all_ids)}/{len(group_fold)}")
    if step4_count == 0:
        raise ValueError(f"required Step4 coverage absent: Step4={step4_count}")
    inventory = {
        "image_count": len(all_ids),
        "group_count": len(group_fold),
        "fold_image_counts": fold_counts,
        "step4_image_count": step4_count,
        "a71_image_count": a71_count,
        "a71_subset_available": a71_count > 0,
    }
    return hashes, inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--model382-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing v2 lock: {args.output}")

    # Lock creation must precede every seed2 OOF file. The training runner may
    # pre-create empty output directories, so directory presence alone is not
    # evidence that an OOF artifact exists. Never inspect logs or score values.
    for fold in range(1, 5):
        seed2_oof = args.model382_root / f"seed2/fold{fold}_oof"
        if seed2_oof.exists() and any(path.is_file() for path in seed2_oof.rglob("*")):
            raise SystemExit(f"seed2 OOF files already exist; cannot establish pre-score lock: {seed2_oof}")

    source_sha = {relative: sha256(ROOT / relative) for relative in LOCKED_FILES}
    mask_sha, inventory = inventory_masks(args.cv_root)
    now = time.time()
    receipt = {
        "model": 382,
        "analysis": "model382_two_seed_mean_sensitivity_v2",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "locked_at_unix": now,
        "seed1_oof_scores_read_before_v2_lock": False,
        "seed2_oof_scores_read_before_v2_lock": False,
        "official_valid_or_test_read": False,
        "v1_lock_sha256": sha256(ROOT / "experiments/results/model382_two_seed_mean_preregistration_lock.json"),
        "source_sha256": source_sha,
        "cv_root_inventory": inventory,
        "fixed_train_oof_mask_sha256": mask_sha,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "analysis": receipt["analysis"],
        "locked_at_utc": receipt["locked_at_utc"],
        "mask_count": len(mask_sha),
        "inventory": inventory,
        "official_valid_or_test_read": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
