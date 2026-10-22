"""Shared local data validation, empty=0 image metrics and result export."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DATA = REPO / "dataset"
SOURCE = ROOT / "source"
_spec = importlib.util.spec_from_file_location(
    "lhs326_common", SOURCE / "experiments/model326_hybrid_fullstats/KoMaP/common.py")
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)
PALETTE = common.PALETTE
CLASSES = ["Al", "Al3Ni", "Primary Si", "Eutectic Si"]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def image_key(path):
    return Path(path).stem.removesuffix("_image")


def group(path):
    m = re.match(r"(A\d+)_", Path(path).name)
    if not m:
        raise ValueError(f"Invalid specimen name: {path}")
    return m[1]


def pairs(data=DATA, split="valid"):
    return common.list_split(str(data), split)


def metric(pred, target):
    pred, target = np.asarray(pred), np.asarray(target)
    if pred.shape != target.shape or pred.ndim != 2:
        raise ValueError("Prediction and mask must have the same H,W")
    if not (np.isin(pred, range(4)).all() and np.isin(target, range(4)).all()):
        raise ValueError("Only class labels 0..3 are supported")
    inter = np.array([np.count_nonzero((pred == c) & (target == c)) for c in range(4)])
    union = np.array([np.count_nonzero((pred == c) | (target == c)) for c in range(4)])
    iou = np.divide(inter, union, out=np.zeros(4, dtype=np.float64), where=union != 0)
    return {"image_miou": float(iou.mean()), "class_iou": dict(zip(CLASSES, iou.tolist()))}


def fold_pairs(data, record, heldout=False):
    selected = set(record["heldout_groups" if heldout else "train_groups"])
    return [(i, m) for i, m in pairs(data, "train") if group(i) in selected]


def audit(data=DATA, verify_source=True):
    info = {"data": str(Path(data).resolve()), "empty_union_policy": "zero", "splits": {}}
    for split, expected in (("train", 70), ("valid", 20), ("test", 10)):
        items = pairs(data, split)
        if len(items) != expected:
            raise ValueError(f"{split}: expected {expected} images, got {len(items)}")
        records = []
        for ip, mp in tqdm(items, desc=f"Check {split}", unit="image"):
            with Image.open(ip) as image:
                w, h = image.size
                image.verify()
            if split != "test":
                if mp is None:
                    raise FileNotFoundError(f"Missing mask for {ip}")
                target = common.load_mask(mp)
                if target.shape != (h, w):
                    raise ValueError(f"Mask dimensions differ: {mp}")
                with Image.open(mp) as mask:
                    if "A" in mask.getbands() and np.any(np.array(mask.getchannel("A")) != 255):
                        raise ValueError(f"Transparent mask: {mp}")
            records.append({"image": Path(ip).name, "height": h, "width": w})
        info["splits"][split] = records
    folds = read_json(ROOT / "configs/folds.json")["completed_folds"]
    all_groups = {group(i) for i, _ in pairs(data, "train")}
    seen = set()
    if sorted(r["fold"] for r in folds) != [1, 2, 3, 4]:
        raise ValueError("Exactly folds 1..4 are required")
    info["folds"] = []
    for r in folds:
        train, held = set(r["train_groups"]), set(r["heldout_groups"])
        if train & held or train | held != all_groups or seen & held:
            raise ValueError(f"Invalid group partition or repeated OOF group: fold {r['fold']}")
        seen |= held
        info["folds"].append({"fold": r["fold"], "train_images": len(fold_pairs(data, r)),
                              "oof_images": len(fold_pairs(data, r, True))})
    if seen != all_groups:
        raise ValueError("OOF must cover each training specimen exactly once")
    if verify_source:
        manifest = read_json(ROOT / "source_manifest.json")
        # The manifest records the original Git-tracked bytes, including historical reports.
        entries = manifest if isinstance(manifest, list) else manifest["files"]
        for row in entries:
            p = SOURCE / row["path"]
            if hashlib.sha256(p.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"Original source changed: {p}")
        info["source_files_verified"] = len(entries)
    return info


def new_result(results_root=REPO / "results"):
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%m%d")
    for number in range(1, 100000):
        path = results_root / f"model_results_{date}_{number}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError("No unused result directory")


def comparison(image_path, pred, target, score, out):
    """Native-size panels; error colors encode ground truth, correct pixels white."""
    original = Image.open(image_path).convert("RGB")
    w, h = original.size
    bar = 64
    canvas = Image.new("RGB", (2 * w, 2 * (h + bar)), "white")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    error = np.full((h, w, 3), 255, dtype=np.uint8)
    error[pred != target] = PALETTE[target[pred != target]]
    panels = [original, Image.fromarray(error), Image.fromarray(PALETTE[pred]), Image.fromarray(PALETTE[target])]
    titles = ["Original image", f"image mIoU: {score*100:.4f}%\n100% - image mIoU: {(1-score)*100:.4f}%",
              "Predicted mask", "Ground-truth mask"]
    draw = ImageDraw.Draw(canvas)
    for k, panel in enumerate(panels):
        x, y = (k % 2) * w, (k // 2) * (h + bar)
        draw.multiline_text((x + 10, y + 5), titles[k], fill="black", font=font, spacing=4)
        canvas.paste(panel, (x, y + bar))
    canvas.save(out)


def export_results(model, predictions, data=DATA, provenance=None, results_root=REPO / "results"):
    items = pairs(data, "valid")
    expected = {image_key(i) for i, _ in items}
    if len(items) != 20 or set(predictions) != expected:
        raise ValueError("Results require exactly the 20 matching validation images")
    # Validate all predictions before reserving a result folder.
    records = []
    for ip, mp in items:
        pred = predictions[image_key(ip)]
        target = common.load_mask(mp)
        records.append({"image": Path(ip).name, "height": target.shape[0], "width": target.shape[1],
                        **metric(pred, target)})
    result = new_result(results_root)
    for folder in ("valid_mask_comparison", "predicted_masks", "predicted_labels"):
        (result / folder).mkdir()
    for (ip, mp), row in tqdm(list(zip(items, records)), desc="Export masks", unit="image"):
        key = image_key(ip)
        pred = np.asarray(predictions[key], dtype=np.uint8)
        Image.fromarray(PALETTE[pred]).save(result / "predicted_masks" / f"{key}_mask.png")
        Image.fromarray(pred).save(result / "predicted_labels" / f"{key}.png")
        comparison(ip, pred, common.load_mask(mp), row["image_miou"],
                   result / "valid_mask_comparison" / f"{key}.png")
    overall = float(np.mean([r["image_miou"] for r in records]))
    summary = {"model": model, "image_miou": overall, "empty_union_policy": "zero",
               "class_order": CLASSES, "images": records, "provenance": provenance or {}}
    write_json(result / "metrics.json", summary)
    lines = [f"# {model}", "", f"- Validation image mIoU: **{overall*100:.6f}%** (20 images)",
             "- 이미지마다 4개 클래스 IoU를 평균한 뒤 이미지 20장을 평균합니다. union=0이면 IoU=0입니다.",
             "- 오류 패널: 정답 픽셀은 흰색, 오답 픽셀은 해당 위치의 정답 클래스 색입니다.",
             "- 오른쪽 위 숫자는 image mIoU 및 100% − image mIoU입니다. 후자는 픽셀 오류율과 다릅니다.",
             "", "| Image | (H, W) | image mIoU | Comparison |", "|---|---|---:|---|"]
    for row in records:
        key = image_key(row["image"])
        lines.append(f"| {row['image']} | ({row['height']}, {row['width']}) | {row['image_miou']*100:.4f}% | [mask](valid_mask_comparison/{key}.png) |")
    lines.extend(["", "## 실행 정보", "", "```json", json.dumps(provenance or {}, indent=2, ensure_ascii=False), "```", ""])
    (result / f"{result.name}.md").write_text("\n".join(lines))
    return result
