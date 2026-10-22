from pathlib import Path
import importlib.metadata
import json
import sys
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from runtime import ROOT, sha256, strict_mask, save_json
from komap_seg.model import build_model


def model_config(cfg):
    return {
        "architecture": cfg["architecture"],
        "encoder_name": cfg["encoder_name"],
        "boundary_head": cfg["boundary_head"],
    }


def preflight(cfg, output):
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
        "model_name": cfg["model_name"],
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
    save_json(output, report)
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
