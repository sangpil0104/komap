#!/usr/bin/env python3
"""Create the fixed Model397 specimen-group OOF symlink trees."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FOLD_FILE = ROOT / "config/model397_group_folds.json"
GROUP_RE = re.compile(r"^(A[0-9]+)_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def group_of(image_id: str) -> str:
    match = GROUP_RE.match(image_id + "_")
    if match is None:
        raise ValueError(f"cannot parse specimen group from image ID {image_id!r}")
    return match.group(1)


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise FileExistsError(f"symlink points elsewhere: {destination}")
        return
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing file: {destination}")
    destination.symlink_to(source.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path,
                        help="Extracted external dataset root containing train/images and train/masks")
    parser.add_argument("--output-root", required=True, type=Path,
                        help="Directory in which model329_cv and model392_cv will be created")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    image_dir = data_root / "train/images"
    mask_dir = data_root / "train/masks"
    images = sorted(image_dir.glob("*_image.png"))
    if not images:
        raise FileNotFoundError(f"no training images found in {image_dir}")

    pairs: dict[str, tuple[Path, Path, str]] = {}
    for image in images:
        image_id = image.name.removesuffix("_image.png")
        group = group_of(image_id)
        mask = mask_dir / f"{image_id}_mask.png"
        if not mask.is_file():
            raise FileNotFoundError(f"mask not found for {image.name}: {mask}")
        if image_id in pairs:
            raise ValueError(f"duplicate training image ID: {image_id}")
        pairs[image_id] = (image.resolve(), mask.resolve(), group)

    fold_config = json.loads(FOLD_FILE.read_text(encoding="utf-8"))
    folds = fold_config["folds"]
    all_groups = {record[2] for record in pairs.values()}
    if len(pairs) != 70 or len(all_groups) != 59 or len(folds) != 4:
        raise ValueError(
            f"Model397 expects 70 train images/59 specimen groups/4 folds; "
            f"found {len(pairs)}/{len(all_groups)}/{len(folds)}"
        )

    heldout_by_fold: dict[int, set[str]] = {}
    seen: set[str] = set()
    for entry in folds:
        fold = int(entry["fold"])
        heldout = set(entry["heldout_groups"])
        if fold in heldout_by_fold or not heldout or heldout & seen:
            raise ValueError(f"invalid or overlapping held-out group list for fold {fold}")
        if not heldout <= all_groups:
            raise ValueError(f"fold {fold} names groups absent from this train set: {heldout-all_groups}")
        heldout_by_fold[fold] = heldout
        seen |= heldout
    if seen != all_groups:
        raise ValueError(f"fixed folds do not cover groups: missing={all_groups-seen}")

    b4_root = output_root / "model329_cv"
    for fold, heldout in sorted(heldout_by_fold.items()):
        for image_id, (image, mask, group) in sorted(pairs.items()):
            split = f"oof_fold{fold}" if group in heldout else f"train_fold{fold}"
            link(image, b4_root / split / "images" / image.name)
            link(mask, b4_root / split / "masks" / mask.name)

        expected = {image_id for image_id, (_, _, group) in pairs.items() if group in heldout}
        observed = {
            path.name.removesuffix("_image.png")
            for path in (b4_root / f"oof_fold{fold}/images").glob("*_image.png")
        }
        if observed != expected:
            raise RuntimeError(f"B4 fold {fold} held-out inventory mismatch")
        expected_counts = {1: 18, 2: 18, 3: 17, 4: 17}
        if len(expected) != expected_counts[fold]:
            raise ValueError(
                f"fold {fold} expected {expected_counts[fold]} held-out images, found {len(expected)}"
            )

    hr_root = output_root / "model392_cv"
    for seed in (1, 2):
        for fold, heldout in sorted(heldout_by_fold.items()):
            fold_root = hr_root / f"seed{seed}_fold{fold}"
            for image_id, (image, mask, group) in sorted(pairs.items()):
                split = "valid" if group in heldout else "train"
                link(image, fold_root / split / "images" / image.name)
                link(mask, fold_root / split / "masks" / mask.name)
            expected = {image_id for image_id, (_, _, group) in pairs.items() if group in heldout}
            observed = {
                path.name.removesuffix("_image.png")
                for path in (fold_root / "valid/images").glob("*_image.png")
            }
            if observed != expected:
                raise RuntimeError(f"HRNet seed{seed} fold{fold} held-out inventory mismatch")

    manifest = {
        "experiment": "Model397 fixed grouped train-OOF preparation",
        "data_root": str(data_root),
        "train_images": len(pairs),
        "specimen_groups": len(all_groups),
        "fold_config_sha256": sha256(FOLD_FILE),
        "image_ids_and_groups": [
            {"image_id": image_id, "group": group}
            for image_id, (_, _, group) in sorted(pairs.items())
        ],
        "folds": [
            {"fold": fold, "heldout_groups": sorted(groups),
             "heldout_images": sum(group in groups for _, _, group in pairs.values())}
            for fold, groups in sorted(heldout_by_fold.items())
        ],
        "valid_and_test_read": False,
    }
    manifest_path = output_root / "model397_split_manifest.json"
    payload = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if manifest_path.exists():
        if manifest_path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"existing split manifest differs: {manifest_path}")
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(payload, encoding="utf-8")
    print(f"prepared {len(pairs)} images / {len(all_groups)} groups into four fixed folds")
    print(f"split manifest: {manifest_path}")


if __name__ == "__main__":
    main()
