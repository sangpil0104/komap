"""Build alloy-grouped K-fold splits over train+valid (90 labeled images).

Images whose names share the `A{n}_` alloy prefix always land in the same
fold, so near-identical microstructures of one alloy never sit on both the
train and held-out side. By default the fold of each alloy is read from
`folds/alloy_to_fold.json` (the exact split used for the reported results);
with `--assignment none` a greedy size-balanced split is computed instead.

Writes symlinked `<output-root>/fold{i}/{train,valid}/{images,masks}` so each
fold can be passed to train.py / evaluate.py as `--data-root`.
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ALLOY_RE = re.compile(r"(A\d+)_")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-root", default="runs/folds")
    p.add_argument("--assignment", default="folds/alloy_to_fold.json", help="JSON alloy->fold map, or 'none'.")
    p.add_argument("--k", type=int, default=5, help="Only used with --assignment none.")
    p.add_argument("--seed", type=int, default=0, help="Only used with --assignment none.")
    return p.parse_args()


def collect(root: Path) -> dict[str, list[tuple[Path, Path]]]:
    by_alloy: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for split in ["train", "valid"]:
        for img in sorted((root / split / "images").glob("*_image.png")):
            stem = img.name[: -len("_image.png")]
            mask = root / split / "masks" / f"{stem}_mask.png"
            by_alloy[ALLOY_RE.match(stem).group(1)].append((img, mask))
    return by_alloy


def balanced_assignment(by_alloy: dict, k: int, seed: int) -> dict[str, int]:
    """Largest alloys first, each into the fold with the fewest images so far."""
    alloys = sorted(by_alloy)
    random.Random(seed).shuffle(alloys)
    alloys.sort(key=lambda a: len(by_alloy[a]), reverse=True)
    sizes = [0] * k
    out = {}
    for a in alloys:
        f = min(range(k), key=lambda i: sizes[i])
        out[a] = f
        sizes[f] += len(by_alloy[a])
    return out


def main() -> None:
    args = parse_args()
    root, out = Path(args.data_root), Path(args.output_root)
    by_alloy = collect(root)
    if args.assignment == "none":
        alloy_to_fold = balanced_assignment(by_alloy, args.k, args.seed)
        k = args.k
    else:
        spec = json.loads(Path(args.assignment).read_text())
        alloy_to_fold, k = spec["alloy_to_fold"], spec["k"]
        missing = sorted(set(by_alloy) - set(alloy_to_fold))
        if missing:
            raise ValueError(f"alloys missing from {args.assignment}: {missing}")

    counts = defaultdict(int)
    for alloy, pairs in by_alloy.items():
        held_out = alloy_to_fold[alloy]
        counts[held_out] += len(pairs)
        for fold in range(k):
            role = "valid" if fold == held_out else "train"
            for src_img, src_mask in pairs:
                for src, kind in [(src_img, "images"), (src_mask, "masks")]:
                    dst = out / f"fold{fold}" / role / kind / src.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if not dst.exists():
                        dst.symlink_to(src.resolve())
    print(f"{sum(counts.values())} images, {len(by_alloy)} alloys, held-out images per fold: {dict(sorted(counts.items()))}")
    print(f"folds written under {out}")


if __name__ == "__main__":
    main()
