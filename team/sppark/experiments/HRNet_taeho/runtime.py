"""Local paths, strict data/metric contracts and repository-format result export."""

from pathlib import Path
import csv
import hashlib
import json
import os
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "source"))
from komap_seg.palette import PALETTE, CLASS_NAMES


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        writer(Path(temp))
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def save_json(path, obj):
    atomic(
        path,
        lambda p: p.write_text(
            json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ),
    )


def load_config(path=None):
    path = Path(path).resolve() if path else ROOT / "configs/default.json"
    cfg = json.loads(path.read_text())
    schema = json.loads((ROOT / "configs/default.json").read_text())
    if set(cfg) != set(schema):
        raise ValueError(f"Config keys differ: {set(cfg)^set(schema)}")
    for key, value in schema.items():
        if type(cfg[key]) != type(value):
            raise TypeError(f"Invalid config type: {key}")
    if cfg["model_name"] != "HRNet_taeho" or cfg["model_date_tag"] != "0928":
        raise ValueError("Model identity must be HRNet_taeho / 0928")
    if cfg["primary_metric"] != "image_miou" or cfg["empty_union_policy"] != "one":
        raise ValueError("Project requires image mIoU with empty union=1")
    if cfg["checkpoint_selection"] != "fixed_last_epoch":
        raise ValueError("Keep upstream fixed-last checkpoint selection")
    if cfg["architecture"] != "UnetPlusPlus" or cfg["encoder_name"] != "tu-hrnet_w48":
        raise ValueError("This import preserves the HRNet-W48 U-Net++ architecture")
    if cfg["patch_size"] < 32 or cfg["patch_size"] % 32 or cfg["input_scale"] <= 0:
        raise ValueError("Invalid patch geometry")
    if not 0 <= cfg["overlap"] < round(cfg["patch_size"] / cfg["input_scale"]):
        raise ValueError("Overlap must be smaller than native tile size")
    for key in [
        "batch_size",
        "epochs",
        "samples_per_image",
        "inference_batch_size",
        "cpu_threads",
    ]:
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if cfg["num_workers"] < 0 or cfg["val_every"] < 0:
        raise ValueError("Invalid workers/validation interval")
    if (
        cfg["lr"] <= 0
        or cfg["encoder_lr_mult"] <= 0
        or cfg["weight_decay"] < 0
        or cfg["boundary_head_weight"] < 0
    ):
        raise ValueError("Invalid optimizer/loss config")
    cfg["data_root"] = str((path.parent / cfg["data_root"]).resolve())
    return cfg


def strict_mask(path):
    arr = np.array(Image.open(path))
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"Expected RGB/RGBA mask: {path}")
    if arr.shape[-1] == 4 and not (arr[..., 3] == 255).all():
        raise ValueError(f"Nonopaque mask: {path}")
    rgb = arr[..., :3]
    labels = np.full(rgb.shape[:2], 255, np.uint8)
    for i, color in enumerate(PALETTE):
        labels[np.all(rgb == color, -1)] = i
    if (labels == 255).any():
        colors, counts = np.unique(rgb[labels == 255], axis=0, return_counts=True)
        raise ValueError(
            f"Unknown mask colors: {list(zip(colors.tolist(), counts.tolist()))[:10]}"
        )
    return labels


def image_metrics(pred, target):
    pred, target = np.asarray(pred), np.asarray(target)
    if pred.shape != target.shape or target.ndim != 2:
        raise ValueError("Expected matching H,W masks")
    if not np.issubdtype(pred.dtype, np.integer) or not np.issubdtype(
        target.dtype, np.integer
    ):
        raise ValueError("Integer labels required")
    if ((pred < 0) | (pred > 3)).any() or ((target < 0) | (target > 3)).any():
        raise ValueError("Labels must be0..3")
    if not target.size:
        raise ValueError("Empty mask")
    cm = np.bincount(
        target.ravel().astype(np.int64) * 4 + pred.ravel(), minlength=16
    ).reshape(4, 4)
    tp = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - tp
    iou = np.ones(4, np.float64)
    np.divide(tp, union, out=iou, where=union > 0)
    return {
        "image_miou": float(iou.mean()),
        "class_iou": iou.tolist(),
        "pixel_error_rate": float(1 - tp.sum() / cm.sum()),
        "confusion_matrix": cm.tolist(),
    }


def summarize(rows):
    if not rows:
        raise ValueError("No evaluated images")
    if len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate evaluated image IDs")
    return {
        "model_name": "HRNet_taeho",
        "model_created_date": "2026-09-28",
        "primary_metric": "image_miou",
        "empty_union_policy": "one",
        "num_images": len(rows),
        "image_miou": float(np.mean([r["image_miou"] for r in rows])),
        "mean_class_iou": np.mean([r["class_iou"] for r in rows], axis=0).tolist(),
        "per_image": rows,
    }


def comparison(path, original, pred, target, score):
    h, w = target.shape
    margin = 16
    gap = 24
    header = 76
    canvas = Image.new(
        "RGB",
        (2 * w + 2 * margin + gap, 2 * h + 2 * margin + gap + 2 * header),
        "white",
    )
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = (
        ImageFont.truetype(font_path, 24)
        if Path(font_path).exists()
        else ImageFont.load_default()
    )
    draw = ImageDraw.Draw(canvas)
    x1 = margin
    x2 = margin + w + gap
    y1 = margin + header
    y2 = y1 + h + gap + header
    error = np.full((h, w, 3), 255, np.uint8)
    wrong = pred != target
    error[wrong] = PALETTE[target[wrong]]
    for x, y, arr in [
        (x1, y1, original),
        (x2, y1, error),
        (x1, y2, PALETTE[pred]),
        (x2, y2, PALETTE[target]),
    ]:
        canvas.paste(Image.fromarray(arr), (x, y))
    for xy, text in [
        ((x1, margin + 20), "Original image"),
        ((x2, margin + 4), f"image mIoU: {score*100:.4f}%"),
        ((x2, margin + 38), f"100% - image mIoU: {(1-score)*100:.4f}%"),
        ((x1, y2 - header + 20), "Predicted mask"),
        ((x2, y2 - header + 20), "Ground-truth mask"),
    ]:
        draw.text(xy, text, fill="black", font=font)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def reserve_result(date="0928"):
    root = REPO / "results"
    root.mkdir(exist_ok=True)
    for number in range(1, 100000):
        path = root / f"model_results_{date}_{number}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError("No unused result number")


def write_report(result, rows, checkpoint, cfg):
    summary = summarize(rows)
    summary.update(
        checkpoint=str(Path(checkpoint).resolve()),
        checkpoint_sha256=sha256(checkpoint),
        inference={
            "patch_size": cfg["patch_size"],
            "input_scale": cfg["input_scale"],
            "overlap": cfg["overlap"],
            "tta": "four_flip" if cfg["tta"] else "none",
        },
        source_commit=json.loads((ROOT / "model_metadata.json").read_text())[
            "source_commit"
        ],
    )
    save_json(result / "summary.json", summary)
    with (result / "per_image.csv").open("w", newline="") as f:
        fields = [
            "image_id",
            "height",
            "width",
            "image_miou",
            "pixel_error_rate",
            *CLASS_NAMES,
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    **{k: r[k] for k in fields[:5]},
                    **dict(zip(CLASS_NAMES, r["class_iou"])),
                }
            )
    with (result / "per_class.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "mean_image_iou"])
        writer.writerows(zip(CLASS_NAMES, summary["mean_class_iou"]))
    save_json(result / "config.json", cfg)
    title = f'# HRNet_taeho — 0928\n\n- 검증 이미지: {len(rows)}장\n- image mIoU: **{summary["image_miou"]*100:.6f}%**\n- 이미지마다4개 클래스 IoU 평균을 구하고 이미지별 동일 가중 평균. union=0이면1.\n- 체크포인트: `{checkpoint}`\n- 원본 구현의 global mIoU 수치와 구분합니다.\n\n'
    table = ["| 이미지 | image mIoU | 비교 그림 |", "|---|---:|---|"]
    for r in rows:
        table.append(
            f'| {r["image_id"]} | {r["image_miou"]*100:.4f}% | [보기](valid_mask_comparison/{r["image_id"]}.png) |'
        )
    (result / (result.name + ".md")).write_text(title + "\n".join(table) + "\n")
    return summary
