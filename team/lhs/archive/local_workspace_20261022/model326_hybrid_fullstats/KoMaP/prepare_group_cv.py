#!/usr/bin/env python3
"""Create symlink-only specimen-group CV splits using a preregistered fold manifest."""
import argparse
import json
import re
from pathlib import Path


def group_id(image_path: Path) -> str:
    match = re.match(r"(A\d+)_", image_path.name)
    if not match:
        raise ValueError(f"cannot parse specimen group from {image_path.name}")
    return match.group(1)


def link_pair(root: Path, split: str, image: Path, mask: Path) -> None:
    image_dir, mask_dir = root / split / "images", root / split / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / image.name).symlink_to(image.resolve())
    (mask_dir / mask.name).symlink_to(mask.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path, help="original dataset root")
    parser.add_argument("--manifest", required=True, type=Path,
                        help="Model303 progress.json with fixed specimen-group folds")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing CV data tree: {args.out}")
    images = sorted((args.data / "train" / "images").glob("*_image.png"))
    if not images:
        raise FileNotFoundError(f"no train images under {args.data}")
    pairs = {}
    for image in images:
        mask = args.data / "train" / "masks" / image.name.replace("_image.png", "_mask.png")
        if not mask.is_file():
            raise FileNotFoundError(mask)
        pairs[image.stem.removesuffix("_image")] = (image, mask, group_id(image))

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest["completed_folds"]
    all_groups = {group for _, _, group in pairs.values()}
    heldout_seen: set[str] = set()
    for record in records:
        fold = int(record["fold"])
        heldout_groups = set(record["heldout_groups"])
        train_groups = set(record["train_groups"])
        if heldout_groups & train_groups:
            raise ValueError(f"fold {fold} has specimen leakage")
        if heldout_groups | train_groups != all_groups:
            raise ValueError(f"fold {fold} does not partition the current train groups")
        heldout_seen |= heldout_groups
        for key, (image, mask, group) in pairs.items():
            split = f"oof_fold{fold}" if group in heldout_groups else f"train_fold{fold}"
            link_pair(args.out, split, image, mask)
        expected_train = sum(group not in heldout_groups for _, _, group in pairs.values())
        expected_holdout = sum(group in heldout_groups for _, _, group in pairs.values())
        actual_train = len(list((args.out / f"train_fold{fold}" / "images").glob("*_image.png")))
        actual_holdout = len(list((args.out / f"oof_fold{fold}" / "images").glob("*_image.png")))
        if (actual_train, actual_holdout) != (expected_train, expected_holdout):
            raise RuntimeError(f"fold {fold} link counts mismatch")
        print(f"fold={fold} train={actual_train} heldout={actual_holdout} "
              f"heldout_groups={len(heldout_groups)}", flush=True)

    if heldout_seen != all_groups:
        raise RuntimeError("fold manifest does not cover every train specimen")
    print(f"verified {len(pairs)} images across {len(records)} grouped folds; "
          "input data were symlinked, not copied", flush=True)


if __name__ == "__main__":
    main()
