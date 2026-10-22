#!/usr/bin/env python3
"""Label-free valid inference for the prelocked Model377 full-train seed pair."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_archive(path: Path, values: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    partial = path.with_name(path.name + ".incomplete")
    if partial.exists():
        raise FileExistsError(f"refusing to overwrite partial file {partial}")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED,
                         allowZip64=True) as archive:
        for name, value in values.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(value), allow_pickle=False)
            archive.writestr(f"{name}.npy", buffer.getvalue())
    partial.replace(path)


def average_archives(paths: list[Path], output: Path,
                     expected_names: set[str]) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    for path in paths:
        with np.load(path) as archive:
            if set(archive.files) != expected_names:
                raise ValueError(f"image keys differ in {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".incomplete")
    if partial.exists():
        raise FileExistsError(f"refusing to overwrite partial file {partial}")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED,
                         allowZip64=True) as destination:
        for name in sorted(expected_names):
            total = None
            reference_shape = None
            for path in paths:
                with np.load(path) as archive:
                    probability = np.asarray(archive[name], dtype=np.float32)
                if (probability.ndim != 3 or probability.shape[0] != 4
                        or not np.isfinite(probability).all()
                        or np.any(probability < -1e-5)
                        or not np.allclose(probability.sum(0), 1.0, atol=3e-3)):
                    raise ValueError(f"invalid probabilities in {path} for {name}")
                if reference_shape is None:
                    reference_shape = probability.shape
                    total = np.zeros(reference_shape, dtype=np.float32)
                elif probability.shape != reference_shape:
                    raise ValueError(f"shape mismatch in {path} for {name}")
                total += probability
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, (total / len(paths)).astype(np.float16),
                                      allow_pickle=False)
            destination.writestr(f"{name}.npy", buffer.getvalue())
    partial.replace(output)
    return {"seeds": [1, 2], "images": len(expected_names),
            "probabilities": str(output), "sha256": sha256(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=Path,
                        default=Path("/workspace/komap_model1/model336_hrnet_best"))
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/komap_model1/data"))
    parser.add_argument("--checkpoint-root", type=Path,
                        default=Path("/workspace/komap_model1/outputs/model377_hrnet_full_train_seeds"))
    parser.add_argument("--model376-result", type=Path,
                        default=Path("/workspace/komap_model1/outputs/model376_bagged_hrnet_valid_eval/results.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=170)
    parser.add_argument("--input-scale", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--expected-images", type=int, default=20)
    args = parser.parse_args()
    code_dir = args.code_dir.resolve()
    args.data_root = args.data_root.resolve()
    args.checkpoint_root = args.checkpoint_root.resolve()
    args.model376_result = args.model376_result.resolve()
    output_dir = args.output_dir.resolve()
    if not code_dir.is_dir():
        raise FileNotFoundError(code_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty {output_dir}")
    model376_result = json.loads(args.model376_result.read_text(encoding="utf-8"))
    if (model376_result.get("model") != 376
            or model376_result.get("metric")
            != "20-image mean of four class IoUs; empty union=0"):
        raise ValueError("Model377 fallback trigger must be the official Model376 result")
    model376_score = float(model376_result["candidate_image_mean_miou_empty0"])
    if not np.isfinite(model376_score) or model376_score >= 0.80:
        raise ValueError(f"Model377 fallback is allowed only if Model376 < 0.80; got {model376_score}")
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(code_dir))
    import torch
    from komap_seg.checkpoint import load_model
    from komap_seg.dataset import KomapFullDataset
    from komap_seg.inference import predict_probs_tta
    from komap_seg.palette import CLASS_NAMES

    valid_image_dir = args.data_root / "valid" / "images"
    dataset = KomapFullDataset(valid_image_dir, masks_dir=None)
    names = {sample["stem"] for sample in dataset.pairs}
    if len(names) != args.expected_images:
        raise ValueError(f"expected {args.expected_images} valid images, found {len(names)}")
    train_images = sorted((args.data_root / "train" / "images").glob("*_image.png"))
    train_masks = sorted((args.data_root / "train" / "masks").glob("*_mask.png"))
    if len(train_images) != 70 or len(train_masks) != 70:
        raise ValueError(f"expected 70 training images/masks, got {len(train_images)}/{len(train_masks)}")
    expected_train = {
        "architecture": "UnetPlusPlus", "encoder_name": "tu-hrnet_w48",
        "encoder_weights": "imagenet", "patch_size": 512, "input_scale": 2.0,
        "samples_per_image": 12, "batch_size": 4, "accumulation_steps": 2,
        "num_workers": 4,
        "epochs": 70, "lr": 3e-4, "encoder_lr_mult": 1.0,
        "weight_decay": 1e-4, "ce_class_weights": None,
        "init_checkpoint": None, "no_boundary_head": False,
        "boundary_head_weight": 1.0, "val_every": 0,
        "no_amp": False, "resume": True,
    }
    train_image_hashes = {path.name: sha256(path) for path in train_images}
    train_mask_hashes = {path.name: sha256(path) for path in train_masks}
    valid_image_hashes = {pair["stem"]: sha256(pair["image"]) for pair in dataset.pairs}
    checkpoint_records = []
    archives: dict[str, list[Path]] = {"flip4": [], "d4": []}
    archive_records: dict[str, list[dict[str, object]]] = {"flip4": [], "d4": []}
    device = torch.device(args.device)

    for seed in (1, 2):
        run_dir = args.checkpoint_root / f"seed{seed}"
        checkpoint = run_dir / "last.pt"
        args_path = run_dir / "args.json"
        train_log = run_dir / "train.log"
        if not checkpoint.is_file() or not args_path.is_file() or not train_log.is_file():
            raise FileNotFoundError(f"missing completed seed-{seed} training artifacts")
        train_args = json.loads(args_path.read_text(encoding="utf-8"))
        for key, expected in expected_train.items():
            if train_args.get(key) != expected:
                raise ValueError(f"seed {seed} training arg {key} mismatch")
        if (Path(train_args.get("data_root", "")).resolve() != args.data_root
                or Path(train_args.get("output_dir", "")).resolve() != run_dir.resolve()
                or train_args.get("seed") != seed
                or "done. final model:" not in train_log.read_text(
                    encoding="utf-8", errors="replace")):
            raise ValueError(f"seed {seed} was not trained to the fixed full-train recipe")
        model = load_model(checkpoint, device)
        probabilities = {mode: {} for mode in ("flip4", "d4")}
        for index, sample in enumerate(dataset, start=1):
            for mode in ("flip4", "d4"):
                probability = predict_probs_tta(
                    model, sample["image"], patch_size=args.patch_size,
                    overlap=args.overlap, batch_size=args.batch_size, tta=True,
                    input_scale=args.input_scale, device=device, tta_mode=mode,
                    downsample_antialias=False).numpy().astype(np.float32)
                if probability.shape[0] != 4 or not np.isfinite(probability).all():
                    raise ValueError(f"bad {mode} prediction for {sample['stem']}")
                probabilities[mode][sample["stem"]] = probability.astype(np.float16)
            print(f"seed {seed}/2: predicted {index}/{len(dataset)} images", flush=True)
        for mode in ("flip4", "d4"):
            path = output_dir / f"seed{seed}_{mode}.npz"
            if path.exists():
                raise FileExistsError(f"refusing to overwrite {path}")
            save_archive(path, probabilities[mode])
            archives[mode].append(path)
            archive_records[mode].append({"seed": seed, "path": str(path),
                                          "sha256": sha256(path)})
        checkpoint_records.append({
            "seed": seed, "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "training_args": str(args_path), "training_args_sha256": sha256(args_path),
            "training_log": str(train_log), "training_log_sha256": sha256(train_log),
        })
        del model, probabilities
        if device.type == "cuda":
            torch.cuda.empty_cache()

    averages = {
        mode: average_archives(paths, output_dir / f"seed_mean_{mode}.npz", names)
        for mode, paths in archives.items()
    }
    code_files = ("train.py", "evaluate_image_mean.py", *sorted(
        str(path.relative_to(code_dir)) for path in (code_dir / "komap_seg").rglob("*.py")))
    code_hashes = {name: sha256(code_dir / name) for name in code_files}
    runner_path = (Path(__file__).resolve().parent / "runners"
                   / "run_model377_hrnet_full_train_seeds.sh")
    manifest = {
        "model": 377,
        "purpose": "label-free inference; no valid masks accessed; fallback requires fixed Model376 score < 0.80",
        "fallback_trigger_model376_score": model376_score,
        "fallback_trigger_model376_result": str(args.model376_result),
        "fallback_trigger_model376_result_sha256": sha256(args.model376_result),
        "class_order": list(CLASS_NAMES), "image_count": len(names),
        "inference_code_dir": str(code_dir), "inference_code_sha256": code_hashes,
        "predictor_script_sha256": sha256(Path(__file__).resolve()),
        "training_runner_sha256": sha256(runner_path),
        "valid_image_dir": str(valid_image_dir.resolve()),
        "valid_image_sha256": valid_image_hashes,
        "training_image_sha256": train_image_hashes,
        "training_mask_sha256": train_mask_hashes,
        "checkpoints": checkpoint_records,
        "inference": {"patch_size": args.patch_size, "overlap": args.overlap,
                      "input_scale": args.input_scale, "batch_size": args.batch_size,
                      "tta_modes": ["flip4", "d4"], "downsample_antialias": False},
        "seed_archives": archive_records,
        "probability_averaging": "arithmetic mean of seed 1 and seed 2 probability maps",
        "averages": averages,
    }
    manifest_path = output_dir / "inference_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(json.dumps({"images": len(names), "averages": averages}, indent=2), flush=True)


if __name__ == "__main__":
    main()
