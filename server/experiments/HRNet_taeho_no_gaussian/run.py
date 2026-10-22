#!/usr/bin/env python3
"""HRNet_taeho_no_gaussian / 0928: project adapter around the unchanged imported model."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from runtime import (
    ROOT,
    atomic,
    comparison,
    image_metrics,
    load_config,
    reserve_result,
    save_json,
    sha256,
    strict_mask,
    write_report,
)
from komap_seg.dataset import KomapFullDataset, KomapPatchDataset, seed_worker
from komap_seg.inference import predict_probs_tta
from komap_seg.losses import BoundaryHeadLoss, ComboLoss
from komap_seg.model import build_model, forward_with_boundary
from komap_seg.palette import PALETTE


def model_config(cfg):
    return {
        "architecture": cfg["architecture"],
        "encoder_name": cfg["encoder_name"],
        "boundary_head": cfg["boundary_head"],
    }


def preflight(cfg):
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    for row in manifest:
        if sha256(ROOT / "source" / row["path"]) != row["sha256"]:
            raise ValueError(f'Imported original source changed: {row["path"]}')
    data = Path(cfg["data_root"])
    rows = []
    for split, count in [("train", 70), ("valid", 20), ("test", 10)]:
        images = sorted((data / split / "images").glob("*_image.png"))
        if len(images) != count:
            raise ValueError(f"{split}: expected{count} images, found{len(images)}")
        if (
            split != "test"
            and len(list((data / split / "masks").glob("*_mask.png"))) != count
        ):
            raise ValueError(f"{split}: image/mask count mismatch")
        for image in tqdm(images, desc=f"preflight {split}"):
            arr = np.array(Image.open(image))
            h, w = arr.shape[:2]
            if (
                arr.ndim != 3
                or arr.shape[2] not in (3, 4)
                or not np.array_equal(arr[:, :, 0], arr[:, :, 1])
                or not np.array_equal(arr[:, :, 0], arr[:, :, 2])
            ):
                raise ValueError(f"Expected grayscale RGB/RGBA input: {image}")
            if arr.shape[2] == 4 and not (arr[:, :, 3] == 255).all():
                raise ValueError(f"Nonopaque input: {image}")
            row = {
                "image_id": image.stem.removesuffix("_image"),
                "split": split,
                "height": h,
                "width": w,
                "image_path": str(image),
                "image_sha256": sha256(image),
            }
            if split != "test":
                mask = (
                    data
                    / split
                    / "masks"
                    / image.name.replace("_image.png", "_mask.png")
                )
                labels = strict_mask(mask)
                if labels.shape != (h, w):
                    raise ValueError(f"Spatial mismatch: {image}")
                row.update(
                    mask_path=str(mask),
                    mask_sha256=sha256(mask),
                    class_counts=np.bincount(labels.ravel(), minlength=4).tolist(),
                )
            rows.append(row)
    ids = [r["image_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate image IDs across splits")
    packages = {
        k: importlib.metadata.version(k)
        for k in [
            "torch",
            "torchvision",
            "segmentation-models-pytorch",
            "timm",
            "albumentations",
            "opencv-python-headless",
            "numpy",
            "Pillow",
            "tqdm",
        ]
    }
    report = {
        "status": "passed",
        "model_name": "HRNet_taeho_no_gaussian",
        "model_created_date": "2026-09-28",
        "dataset_counts": {"train": 70, "valid": 20, "test": 10},
        "source_files_verified": len(manifest),
        "python": sys.version,
        "packages": packages,
        "config": cfg,
        "optimizer_updates": 0,
        "upstream_environment_exact_match": False,
        "dataset": rows,
    }
    save_json(ROOT / "reports/preflight.json", report)
    return report


def load_checkpoint(path, cfg):
    path = Path(path).resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "model_config" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError("Unsupported checkpoint format")
    if checkpoint["model_config"] != model_config(cfg):
        raise ValueError("Checkpoint model config mismatch")
    if "experiment_config" in checkpoint:
        previous = checkpoint["experiment_config"]
        for key in ["patch_size", "input_scale"]:
            if previous[key] != cfg[key]:
                raise ValueError(f"Train/inference {key} mismatch")
    model = build_model(**checkpoint["model_config"], encoder_weights=None)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(cfg["device"]).eval()


@torch.inference_mode()
def quick_validate(model, cfg):
    """Image mIoU at native1x; progress-only, distinct from final tile/flip inference."""
    ds = KomapFullDataset(
        Path(cfg["data_root"]) / "valid/images", Path(cfg["data_root"]) / "valid/masks"
    )
    scores = []
    model.eval()
    for sample in tqdm(ds, desc="quick valid native1x (no TTA)", leave=False):
        x = sample["image"][None].to(cfg["device"])
        h, w = x.shape[-2:]
        x = torch.nn.functional.pad(x, (0, (-w) % 32, 0, (-h) % 32), mode="reflect")
        pred = model(x)[0, :, :h, :w].argmax(0).cpu().numpy()
        scores.append(image_metrics(pred, sample["mask"].numpy())["image_miou"])
    model.train()
    return float(np.mean(scores))


@torch.inference_mode()
def evaluate(cfg, checkpoint, split="valid", output=None):
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    # Check all GT before reserving outputs; fail instead of nearest-color guessing.
    data = Path(cfg["data_root"])
    ds = KomapFullDataset(data / split / "images", data / split / "masks")
    for pair in ds.pairs:
        strict_mask(pair["mask"])
    model = load_checkpoint(checkpoint, cfg)
    result = (
        reserve_result(cfg["model_date_tag"])
        if output is None
        else Path(output).resolve()
    )
    if output is not None:
        result.mkdir(parents=True, exist_ok=False)
    (result / "predictions").mkdir()
    rows = []
    for sample in tqdm(
        ds, desc=f"{split} tile+flip4" if cfg["tta"] else f"{split} tile"
    ):
        prob = predict_probs_tta(
            model,
            sample["image"],
            patch_size=cfg["patch_size"],
            overlap=cfg["overlap"],
            batch_size=cfg["inference_batch_size"],
            input_scale=cfg["input_scale"],
            tta=cfg["tta"],
            device=torch.device(cfg["device"]),
        )
        pred = prob.argmax(0).numpy().astype(np.uint8)
        target = sample["mask"].numpy()
        row = {
            "image_id": sample["stem"],
            "height": sample["height"],
            "width": sample["width"],
            **image_metrics(pred, target),
        }
        rows.append(row)
        Image.fromarray(PALETTE[pred]).save(
            result / "predictions" / f'{sample["stem"]}_mask.png'
        )
        original = np.array(
            Image.open(data / split / "images" / f'{sample["stem"]}_image.png').convert(
                "RGB"
            )
        )
        comparison(
            result / "valid_mask_comparison" / f'{sample["stem"]}.png',
            original,
            pred,
            target,
            row["image_miou"],
        )
    summary = write_report(result, rows, checkpoint, cfg)
    print(
        f'HRNet_taeho_no_gaussian image mIoU={summary["image_miou"]*100:.6f}%; results={result}',
        flush=True,
    )
    return result


@torch.inference_mode()
def predict(cfg, checkpoint, output):
    output = Path(output).resolve()
    model = load_checkpoint(checkpoint, cfg)
    ds = KomapFullDataset(Path(cfg["data_root"]) / "test/images")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for sample in tqdm(ds, desc="test predictions"):
        prob = predict_probs_tta(
            model,
            sample["image"],
            patch_size=cfg["patch_size"],
            overlap=cfg["overlap"],
            batch_size=cfg["inference_batch_size"],
            input_scale=cfg["input_scale"],
            tta=cfg["tta"],
            device=torch.device(cfg["device"]),
        )
        pred = prob.argmax(0).numpy().astype(np.uint8)
        if pred.shape != (sample["height"], sample["width"]):
            raise ValueError("Native shape mismatch")
        path = output / f'{sample["stem"]}_mask.png'
        Image.fromarray(PALETTE[pred]).save(path)
        strict_mask(path)
        rows.append(
            {
                "image_id": sample["stem"],
                "height": sample["height"],
                "width": sample["width"],
                "prediction": path.name,
                "sha256": sha256(path),
            }
        )
    save_json(
        output / "manifest.json",
        {
            "model_name": "HRNet_taeho_no_gaussian",
            "model_created_date": "2026-09-28",
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "images": rows,
        },
    )
    return output


def train(cfg):
    audit = preflight(cfg)
    torch.manual_seed(cfg["seed"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = ROOT / "outputs/runs" / f'HRNet_taeho_no_gaussian_0928_seed{cfg["seed"]}_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "config.json", cfg)
    save_json(output / "preflight.json", audit)
    save_json(
        output / "model_metadata.json",
        json.loads((ROOT / "model_metadata.json").read_text()),
    )
    shutil.copytree(
        ROOT / "source", output / "source", ignore=shutil.ignore_patterns("__pycache__")
    )
    (output / "adapter").mkdir()
    for filename in ["run.py", "runtime.py", "run.sh", "requirements.txt"]:
        shutil.copy2(ROOT / filename, output / "adapter" / filename)
    save_json(
        ROOT / "outputs/latest_run.json",
        {"run_dir": str(output), "status": "initializing", "model_name": "HRNet_taeho_no_gaussian"},
    )
    model = build_model(**model_config(cfg), encoder_weights=cfg["encoder_weights"]).to(
        cfg["device"]
    )
    data = Path(cfg["data_root"])
    ds = KomapPatchDataset(
        data / "train/images",
        data / "train/masks",
        patch_size=cfg["patch_size"],
        input_scale=cfg["input_scale"],
        samples_per_image=cfg["samples_per_image"],
        seed=cfg["seed"],
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        worker_init_fn=seed_worker,
        drop_last=True,
        pin_memory=True,
    )
    if not len(loader):
        raise ValueError("Empty training loader")
    seg_loss = ComboLoss().to(cfg["device"])
    boundary_loss = BoundaryHeadLoss().to(cfg["device"])
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": cfg["lr"] * cfg["encoder_lr_mult"],
            },
            {
                "params": [p for p in model.parameters() if id(p) not in encoder_ids],
                "lr": cfg["lr"],
            },
        ],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]
    )
    start = time.monotonic()
    updates = 0
    print(
        f"Run: {output}; {len(ds.images)} images; {len(loader)} steps/epoch; FP32",
        flush=True,
    )
    with tqdm(
        total=cfg["epochs"] * len(loader), desc="HRNet_taeho_no_gaussian", unit="update"
    ) as progress:
        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            epoch_loss = 0.0
            for x, y in loader:
                x, y = x.to(cfg["device"], non_blocking=True), y.to(
                    cfg["device"], non_blocking=True
                )
                optimizer.zero_grad()
                if cfg["boundary_head"]:
                    logits, boundary = forward_with_boundary(model, x)
                    loss = seg_loss(logits, y) + cfg[
                        "boundary_head_weight"
                    ] * boundary_loss(boundary, y)
                else:
                    loss = seg_loss(model(x), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss: epoch{epoch}")
                loss.backward()
                optimizer.step()
                value = float(loss.detach())
                epoch_loss += value
                updates += 1
                progress.update()
                progress.set_postfix(
                    epoch=f'{epoch}/{cfg["epochs"]}', loss=f"{value:.4f}"
                )
                del x, y, loss
                if cfg["boundary_head"]:
                    del logits, boundary
            scheduler.step()
            record = {
                "epoch": epoch,
                "optimizer_updates": updates,
                "train_loss": epoch_loss / len(loader),
                "learning_rates": scheduler.get_last_lr(),
                "wall_seconds": time.monotonic() - start,
            }
            if cfg["val_every"] and (
                epoch % cfg["val_every"] == 0 or epoch == cfg["epochs"]
            ):
                record["quick_native1x_image_miou"] = quick_validate(model, cfg)
            with (output / "history.jsonl").open("a") as f:
                f.write(json.dumps(record, allow_nan=False) + "\n")
            state = {
                "model_config": model_config(cfg),
                "state_dict": model.state_dict(),
                "experiment_config": cfg,
                "epoch": epoch,
                "optimizer_updates": updates,
                "model_name": "HRNet_taeho_no_gaussian",
                "model_created_date": "2026-09-28",
            }
            atomic(output / "last.pt", lambda p: torch.save(state, p))
            save_json(
                output / "status.json",
                {
                    "status": "training",
                    "epoch": epoch,
                    "epochs": cfg["epochs"],
                    "optimizer_updates": updates,
                    "wall_seconds": time.monotonic() - start,
                },
            )
            tqdm.write(json.dumps(record, allow_nan=False))
    save_json(
        output / "status.json",
        {
            "status": "training_complete_evaluation_pending",
            "epoch": cfg["epochs"],
            "optimizer_updates": updates,
        },
    )
    del loader, model, optimizer, scheduler, state
    if str(cfg["device"]).startswith("cuda"):
        torch.cuda.empty_cache()
    result = evaluate(cfg, output / "last.pt")
    complete = {
        "status": "completed",
        "run_dir": str(output),
        "result_dir": str(result),
        "checkpoint": str(output / "last.pt"),
        "optimizer_updates": updates,
        "wall_seconds": time.monotonic() - start,
    }
    save_json(output / "completed.json", complete)
    save_json(output / "status.json", complete)
    save_json(ROOT / "outputs/latest_run.json", complete)
    return output


def smoke(cfg):
    """Configured-size synthetic forward/backward; zero optimizer updates."""
    batch, size = cfg["batch_size"], cfg["patch_size"]
    model = (
        build_model(**model_config(cfg), encoder_weights=None).to(cfg["device"]).train()
    )
    x = torch.randn(batch, 3, size, size, device=cfg["device"])
    y = torch.randint(0, 4, (batch, size, size), device=cfg["device"])
    boundary = None
    if cfg["boundary_head"]:
        logits, boundary = forward_with_boundary(model, x)
        loss = ComboLoss()(logits, y) + cfg[
            "boundary_head_weight"
        ] * BoundaryHeadLoss()(boundary, y)
    else:
        logits = model(x)
        loss = ComboLoss()(logits, y)
    loss.backward()
    if logits.shape != (batch, 4, size, size) or (
        boundary is not None and boundary.shape != (batch, 1, size, size)
    ):
        raise ValueError("Unexpected output shapes")
    finite = all(
        torch.isfinite(p.grad).all().item()
        for p in model.parameters()
        if p.grad is not None
    )
    if not finite:
        raise FloatingPointError("Nonfinite gradients")
    result = {
        "status": "passed",
        "model": "HRNet_taeho_no_gaussian",
        "data": "synthetic_random_configured_size",
        "input_shape": list(x.shape),
        "optimizer_updates": 0,
        "pretrained_weights_downloaded": False,
        "logits_shape": list(logits.shape),
        "boundary_shape": list(boundary.shape) if boundary is not None else None,
        "finite_gradients": finite,
        "parameters": sum(p.numel() for p in model.parameters()),
        "loss": float(loss.detach()),
        "device": cfg["device"],
    }
    if str(cfg["device"]).startswith("cuda"):
        result["peak_gpu_memory_GiB"] = torch.cuda.max_memory_allocated() / 2**30
    save_json(ROOT / "reports/smoke.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["preflight", "smoke", "prepare", "train", "evaluate", "predict"],
        nargs="?",
        default="preflight",
    )
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    torch.set_num_threads(cfg["cpu_threads"])
    if args.command == "preflight":
        result = preflight(cfg)
        print(json.dumps({k: v for k, v in result.items() if k != "dataset"}, indent=2))
    elif args.command == "smoke":
        print(json.dumps(smoke(cfg), indent=2))
    elif args.command == "prepare":
        preflight(cfg)
        model = build_model(**model_config(cfg), encoder_weights=cfg["encoder_weights"])
        print(
            "ImageNet encoder weights prepared; no training performed. Parameters:",
            sum(p.numel() for p in model.parameters()),
        )
    elif args.command == "train":
        if not args.execute:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "command": "./run.sh train --execute",
                        "config": cfg,
                    },
                    indent=2,
                )
            )
            return
        print(train(cfg))
    elif args.command in ["evaluate", "predict"]:
        if not args.checkpoint:
            parser.error("--checkpoint is required")
        if args.command == "evaluate":
            print(evaluate(cfg, args.checkpoint, output=args.output))
        else:
            if not args.output:
                parser.error("--output is required for predictions")
            print(predict(cfg, args.checkpoint, args.output))


if __name__ == "__main__":
    main()
