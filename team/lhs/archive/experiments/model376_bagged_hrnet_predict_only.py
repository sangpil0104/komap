#!/usr/bin/env python3
"""Generate label-free valid probability maps from the four Model374 folds.

This script intentionally opens validation images only (never masks) and does
not calculate a score. It is the inference-only half of a prelocked Model360
bagged-HRNet candidate; the fold probabilities are averaged after inference.
"""
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


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED,
                         allowZip64=True) as archive:
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
            archive.writestr(f"{name}.npy", buffer.getvalue())


def average_archives(paths: list[Path], output: Path,
                     expected_names: set[str]) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    for path in paths:
        with np.load(path) as archive:
            names = set(archive.files)
        if names != expected_names:
            raise ValueError(f"image-key mismatch in {path}")

    output.parent.mkdir(parents=True, exist_ok=True)
    incomplete = output.with_name(output.name + ".incomplete")
    if incomplete.exists():
        raise FileExistsError(f"refusing to overwrite partial archive {incomplete}")
    try:
        with zipfile.ZipFile(incomplete, "w", compression=zipfile.ZIP_DEFLATED,
                             allowZip64=True) as out_archive:
            for name in sorted(expected_names):
                total = None
                reference_shape = None
                for path in paths:
                    with np.load(path) as archive:
                        prob = np.asarray(archive[name], dtype=np.float32)
                    if (prob.ndim != 3 or prob.shape[0] != 4
                            or not np.isfinite(prob).all()):
                        raise ValueError(f"invalid probabilities in {path} for {name}")
                    if reference_shape is None:
                        reference_shape = prob.shape
                        total = np.zeros(reference_shape, dtype=np.float32)
                    elif prob.shape != reference_shape:
                        raise ValueError(f"fold probability shape mismatch for {name}")
                    if np.any(prob < -1e-5) or not np.allclose(prob.sum(0), 1.0, atol=3e-3):
                        raise ValueError(f"non-normalized probabilities in {path} for {name}")
                    total += prob
                mean_prob = (total / len(paths)).astype(np.float16)
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, mean_prob, allow_pickle=False)
                out_archive.writestr(f"{name}.npy", buffer.getvalue())
    except Exception:
        # Keep the .incomplete archive as a recovery/audit artifact.
        raise
    incomplete.replace(output)
    return {"images": len(expected_names), "folds": len(paths),
            "sha256": sha256(output), "probabilities": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=Path,
                        default=Path("/workspace/komap_model1/model336_hrnet_best"))
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/komap_model1/data"))
    parser.add_argument("--cv-data", type=Path,
                        default=Path("/workspace/komap_model1/data_model322_group_cv"))
    parser.add_argument("--checkpoint-root", type=Path,
                        default=Path("/workspace/komap_model1/outputs/model374_hrnet_group_oof"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=170)
    parser.add_argument("--input-scale", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--expected-images", type=int, default=20)
    args = parser.parse_args()
    code_dir = args.code_dir.resolve()
    if not code_dir.is_dir():
        raise FileNotFoundError(f"Model336 code directory not found: {code_dir}")
    sys.path.insert(0, str(code_dir))
    import torch
    from komap_seg.checkpoint import load_model
    from komap_seg.dataset import KomapFullDataset
    from komap_seg.inference import predict_probs_tta
    from komap_seg.palette import CLASS_NAMES

    args.data_root = args.data_root.resolve()
    args.cv_data = args.cv_data.resolve()
    args.checkpoint_root = args.checkpoint_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_dir = args.data_root / "valid" / "images"
    # No mask directory is passed: validation labels are not opened here.
    dataset = KomapFullDataset(image_dir, masks_dir=None)
    names = {sample["stem"] for sample in dataset.pairs}
    if len(names) != args.expected_images:
        raise ValueError(f"expected {args.expected_images} valid images, found {len(names)}")
    device = torch.device(args.device)
    modes = ("flip4", "d4")
    checkpoint_records = []
    fold_paths = {mode: [] for mode in modes}
    fold_archive_records = {mode: [] for mode in modes}
    expected_train_args = {
        "architecture": "UnetPlusPlus", "encoder_name": "tu-hrnet_w48",
        "encoder_weights": "imagenet", "patch_size": 512, "input_scale": 2.0,
        "samples_per_image": 12, "batch_size": 4, "accumulation_steps": 2,
        "epochs": 70, "lr": 3e-4, "encoder_lr_mult": 1.0,
        "weight_decay": 1e-4, "boundary_head_weight": 1.0,
        "ce_class_weights": None, "init_checkpoint": None,
        "no_boundary_head": False, "seed": 0, "val_every": 0,
        "no_amp": False, "resume": True,
    }

    gate_path = args.checkpoint_root / "fold1_kill_switch_official.json"
    if not gate_path.is_file():
        raise FileNotFoundError(f"official Model374 fold-1 OOF gate missing: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate_delta = (float(gate.get("candidate_oof_image_mean_miou_empty0", float("nan")))
                  - float(gate.get("model329_same_fold_oof_image_mean_miou_empty0",
                                   float("nan")))) * 100.0
    if (gate.get("candidate_model") != 374 or gate.get("fold") != 1
            or gate.get("continue") is not True
            or gate.get("model329_recomputed_from_grouped_train_oof_probabilities") is not True
            or gate.get("kill_switch_threshold_pp") != -2.0
            or not np.isfinite(gate_delta) or gate_delta <= -2.0
            or not np.isclose(gate_delta, float(gate.get("delta_pp", float("nan"))),
                              atol=1e-9, rtol=0)):
        raise ValueError("official Model374 fold-1 OOF gate is missing, failed, or inconsistent")
    fold1_summary_path = (args.checkpoint_root / "fold1_oof" / "valid_summary.json")
    if not fold1_summary_path.is_file():
        raise FileNotFoundError(f"Model374 fold-1 OOF summary missing: {fold1_summary_path}")
    fold1_summary = json.loads(fold1_summary_path.read_text(encoding="utf-8"))
    if not np.isclose(float(fold1_summary.get("image_miou_empty_zero", float("nan"))),
                      float(gate["candidate_oof_image_mean_miou_empty0"]),
                      atol=1e-10, rtol=0):
        raise ValueError("official fold-1 gate score does not match the OOF inference summary")

    for fold in range(1, 5):
        checkpoint = args.checkpoint_root / f"fold{fold}" / "last.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        run_dir = checkpoint.parent
        args_path = run_dir / "args.json"
        train_log = run_dir / "train.log"
        source_manifest_path = args.checkpoint_root / f"fold{fold}_oof" / "inference_manifest.json"
        if not args_path.is_file() or not train_log.is_file() or not source_manifest_path.is_file():
            raise FileNotFoundError(f"missing completed Model374 provenance for fold {fold}")
        train_args = json.loads(args_path.read_text(encoding="utf-8"))
        for key, expected in expected_train_args.items():
            if train_args.get(key) != expected:
                raise ValueError(f"fold {fold} training arg {key}={train_args.get(key)!r}; "
                                 f"expected {expected!r}")
        if Path(train_args.get("output_dir", "")).resolve() != run_dir.resolve():
            raise ValueError(f"fold {fold} training output_dir does not match checkpoint directory")
        if "done. final model:" not in train_log.read_text(encoding="utf-8", errors="replace"):
            raise ValueError(f"fold {fold} does not have a completed fixed-epoch train log")
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        checkpoint_hash = sha256(checkpoint)
        heldout_root = (args.cv_data / f"oof_fold{fold}").resolve()
        expected_data_root = (args.data_root.parent / f"data_model374_fold{fold}").resolve()
        if (Path(source_manifest.get("checkpoint", "")).resolve() != checkpoint.resolve()
                or Path(source_manifest.get("training_args", "")).resolve()
                != args_path.resolve()
                or Path(source_manifest.get("training_data_root", "")).resolve()
                != expected_data_root):
            raise ValueError(f"fold {fold} source manifest points at a different training run")
        if (source_manifest.get("fold") != fold
                or source_manifest.get("checkpoint_sha256") != checkpoint_hash
                or Path(source_manifest.get("heldout_data_root", "")).resolve()
                != heldout_root
                or Path(train_args.get("data_root", "")).resolve() != expected_data_root):
            raise ValueError(f"fold {fold} inference manifest does not match checkpoint/split")
        model = load_model(checkpoint, device)
        probabilities = {mode: {} for mode in modes}
        for index, sample in enumerate(dataset, start=1):
            for mode in modes:
                prob = predict_probs_tta(
                    model,
                    sample["image"],
                    patch_size=args.patch_size,
                    overlap=args.overlap,
                    batch_size=args.batch_size,
                    tta=True,
                    input_scale=args.input_scale,
                    device=device,
                    tta_mode=mode,
                    downsample_antialias=False,
                ).numpy().astype(np.float32)
                if prob.shape[0] != 4 or not np.isfinite(prob).all():
                    raise ValueError(f"bad {mode} prediction for {sample['stem']}")
                probabilities[mode][sample["stem"]] = prob.astype(np.float16)
            print(f"fold {fold}/4: predicted {index}/{len(dataset)} images", flush=True)
        for mode in modes:
            path = output_dir / f"fold{fold}_{mode}.npz"
            if path.exists():
                raise FileExistsError(f"refusing to overwrite {path}")
            save_npz(path, probabilities[mode])
            fold_paths[mode].append(path)
            fold_archive_records[mode].append({"fold": fold, "path": str(path),
                                               "sha256": sha256(path)})
        checkpoint_records.append({"fold": fold, "checkpoint": str(checkpoint),
                                   "checkpoint_sha256": checkpoint_hash,
                                   "training_args": str(args_path),
                                   "training_args_sha256": sha256(args_path),
                                   "source_manifest": str(source_manifest_path),
                                   "source_manifest_sha256": sha256(source_manifest_path)})
        del model
        del probabilities
        if device.type == "cuda":
            torch.cuda.empty_cache()

    averages = {}
    for mode in modes:
        averages[mode] = average_archives(
            fold_paths[mode], output_dir / f"fold_mean_{mode}.npz", names)
    code_files = ("train.py", "evaluate_image_mean.py", *sorted(
        str(path.relative_to(code_dir)) for path in (code_dir / "komap_seg").rglob("*.py")))
    code_sha256 = {relative: sha256(code_dir / relative) for relative in code_files}
    image_hashes = {pair["stem"]: sha256(pair["image"]) for pair in dataset.pairs}
    manifest = {
        "model": 376,
        "purpose": "label-free inference only; no valid masks or scores accessed",
        "image_count": len(names),
        "class_order": list(CLASS_NAMES),
        "class_palette_sha256": sha256(code_dir / "komap_seg" / "palette.py"),
        "inference_code_dir": str(code_dir),
        "inference_code_sha256": code_sha256,
        "predictor_script_sha256": sha256(Path(__file__).resolve()),
        "valid_image_dir": str(image_dir.resolve()),
        "valid_image_sha256": image_hashes,
        "folds": checkpoint_records,
        "fold1_official_kill_gate": {
            "path": str(gate_path), "sha256": sha256(gate_path),
            "candidate_oof_image_mean_miou_empty0":
                gate["candidate_oof_image_mean_miou_empty0"],
            "model329_same_fold_oof_image_mean_miou_empty0":
                gate["model329_same_fold_oof_image_mean_miou_empty0"],
            "delta_pp": gate["delta_pp"], "continue": gate["continue"],
        },
        "fold1_oof_summary": {
            "path": str(fold1_summary_path), "sha256": sha256(fold1_summary_path),
            "image_miou_empty_zero": fold1_summary["image_miou_empty_zero"],
        },
        "inference": {"patch_size": args.patch_size, "overlap": args.overlap,
                      "input_scale": args.input_scale, "batch_size": args.batch_size,
                      "tta_modes": list(modes), "downsample_antialias": False},
        "probability_averaging": "arithmetic mean of four fold probability maps",
        "fold_archives": fold_archive_records,
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
