#!/usr/bin/env python3
"""Build Model349 base entries without loading the auxiliary representation net.

The historical Model183 cache blends a separate representation-ASPP model
into Primary Si at gamma=.40. Model349 immediately algebraically undoes that
blend. This builder computes the pre-blend mixture directly from raw images,
then stores only the fields actually read by Model349 inference.
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
for path in (HERE / "historical_scripts", HERE):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import model183_metric_calibration as core  # noqa: E402
from model187_schedule_branch_blend import undo_model181_primary_blend  # noqa: E402


def make_preaux(item: dict, high: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """The gamma=0 branch of historical Model183.make_mixed, without extra."""
    params = item["params"]
    style_high = ((1.0 - params["style_weight"]) * high[0]
                  + params["style_weight"] * high[1])
    base = core.MATRIX * (1.0 - params["high_weight"] - params["pair_weight"])
    mixed = np.asarray([
        sum(base[c, m] * item["branches"][m][c] for m in range(3))
        + params["high_weight"] * style_high[c]
        + params["pair_weight"] * item["pair"][c]
        for c in range(4)
    ], dtype=np.float32)
    return mixed / np.maximum(mixed.sum(0, keepdims=True), 1e-7)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--pair-checkpoint", type=Path, default=None)
    parser.add_argument("--high-only", action="store_true",
                        help="omit the three cached base branches and pair network")
    for name in ("restart51", "model109", "seed456", "pairwise", "ema"):
        parser.add_argument(f"--source-{name}", type=Path, required=True)
    parser.add_argument("--alpha-1024", type=float, default=0.20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only-image", default=None,
                        help="image stem for one-image parity probe")
    parser.add_argument("--parity-reference", type=Path, default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.high_only and (args.cache_dir is None or args.pair_checkpoint is None):
        parser.error("--cache-dir and --pair-checkpoint are required without --high-only")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    global_z = core.fit_global_stats(args.data_dir)
    global_q = core.fit_percentile_stats(args.data_dir)
    train_stats = []
    for image_path, _ in core.collect_split(args.data_dir, "train"):
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        train_stats.append((float(gray.mean()), float(gray.std())))
    train_mean = float(np.mean([value[0] for value in train_stats]))
    train_std = float(np.mean([value[1] for value in train_stats]))
    pair = None if args.high_only else core.load_aspp(args.pair_checkpoint, device)
    source_args = argparse.Namespace(**{
        f"source_{name}": getattr(args, f"source_{name}")
        for name in ("restart51", "model109", "seed456", "pairwise", "ema")
    })
    original = None
    reference_names = None
    if args.parity_reference:
        with args.parity_reference.open("rb") as stream:
            original = pickle.load(stream)
        reference_names = [path.stem for path, _ in core.collect_split(args.data_dir, "valid")]
        if len(original) != len(reference_names):
            raise ValueError("parity reference and valid images differ in count")
    entries = []
    checks = []
    for image_path, mask_path in core.collect_split(args.data_dir, "valid"):
        if args.only_image and image_path.stem != args.only_image:
            continue
        branches = None
        if not args.high_only:
            with np.load(args.cache_dir / f"{image_path.stem}.npz") as loaded:
                branches = [loaded[f"p{i}"].astype(np.float32) for i in range(3)]
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        gray = raw[..., 0].astype(np.float32)
        mean, std = float(gray.mean()), float(gray.std())
        source = core.choose_source(mean, source_args)
        pair_value = None
        if not args.high_only:
            pair_value = core.predict_rgb(pair, image_path, 768, 576, device, amp, 1, "d4", "gaussian")
        adapted = core.adapt_model(source, raw, device, amp, 5, 1e-5)
        style = core.transform_image(raw, "standardize", train_mean, train_std)
        high768 = (core.predict_array(adapted, raw, 768, 576, device, amp, 1, "d4", "gaussian"),
                   core.predict_array(adapted, style, 768, 576, device, amp, 1, "d4", "gaussian"))
        high1024 = (core.predict_array(adapted, raw, 1024, 768, device, amp, 1, "d4", "gaussian"),
                    core.predict_array(adapted, style, 1024, 768, device, amp, 1, "d4", "gaussian"))
        high = tuple((1.0 - args.alpha_1024) * high768[index]
                     + args.alpha_1024 * high1024[index] for index in (0, 1))
        params = core.route_params(mean, std)
        if args.high_only:
            preaux = ((1.0 - params["style_weight"]) * high[0]
                      + params["style_weight"] * high[1]).astype(np.float32)
            preaux /= np.maximum(preaux.sum(axis=0, keepdims=True), 1e-7)
        else:
            item = {"branches": branches, "pair": pair_value.astype(np.float32),
                    "params": params}
            preaux = make_preaux(item, high)
        compact = {
            "preaux": preaux,
            "z": (gray - mean) / max(std, 1e-5),
            "q": core.percentile_image(gray),
            "prior_z_stats": global_z,
            "prior_q_stats": global_q,
            "params": params,
            "target": core.decode_mask(Image.open(mask_path).convert("RGB")),
        }
        entries.append(compact)
        if original is not None:
            reference = original[reference_names.index(image_path.stem)]
            old_base = undo_model181_primary_blend(reference)
            checks.append({"image": image_path.stem,
                           "preaux_max_abs": float(np.max(np.abs(preaux - old_base))),
                           "preaux_mean_abs": float(np.mean(np.abs(preaux - old_base))),
                           "target_equal": bool(np.array_equal(compact["target"], reference["target"])),
                           "selected_source": str(source)})
        print(f"built {image_path.stem} without representation model", flush=True)
        if not args.high_only:
            del item
        del adapted, high768, high1024, high, branches, pair_value
        gc.collect()
        if amp:
            torch.cuda.empty_cache()
    expected = 1 if args.only_image else 20
    if len(entries) != expected:
        raise ValueError(f"expected {expected} entries, got {len(entries)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump(entries, stream, protocol=pickle.HIGHEST_PROTOCOL)
    args.output.with_suffix(".json").write_text(json.dumps({
        "entries": len(entries), "representation_checkpoint_loaded": False,
        "representation_forward_calls": 0,
        "pair_checkpoint_loaded": not args.high_only,
        "branch_cache_read": not args.high_only,
        "base_mode": "high_only" if args.high_only else "rep_free_full_base",
        "alpha_1024": args.alpha_1024,
        "parity": checks,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
