"""Re-render quantitative comparison figures from saved masks, preserving originals.

No model imports or training/inference. Pixel arrays keep their native resolution.
The original comparison directory is moved into comparison_history only after all
replacement figures pass pixel and metric verification.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[1]
PALETTE = np.array([[153, 127, 76], [25, 76, 153], [76, 178, 76], [204, 204, 204]], np.uint8)
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def labels(array):
    result = np.full(array.shape[:2], 255, np.uint8)
    for index, color in enumerate(PALETTE):
        result[np.all(array == color, axis=-1)] = index
    if np.any(result == 255):
        raise ValueError("Unknown mask color")
    return result


def render(path, original, predicted, target, row, seed):
    if original.shape != predicted.shape or original.shape != target.shape:
        raise ValueError("Image and mask dimensions differ")
    wrong = np.any(predicted != target, axis=-1)
    error_count, pixels = int(wrong.sum()), wrong.size
    error_rate = error_count / pixels * 100
    error = np.full_like(target, 255)
    error[wrong] = target[wrong]
    pred_ids, target_ids = labels(predicted), labels(target)
    cm = np.bincount((target_ids.astype(np.int64) * 4 + pred_ids).ravel(), minlength=16).reshape(4, 4)
    tp = cm.diagonal()
    union = cm.sum(0) + cm.sum(1) - tp
    score = np.divide(tp, union, out=np.ones(4), where=union != 0).mean()
    if not np.isclose(score, row["image_miou"], rtol=0, atol=1e-12):
        raise ValueError(f"Stored metric mismatch: seed {seed}, {row['image']}")
    h, w = target.shape[:2]
    margin, gap, header = 16, 24, 76
    title_font = ImageFont.truetype(str(FONT_DIR / "DejaVuSans-Bold.ttf"), 28)
    detail_font = ImageFont.truetype(str(FONT_DIR / "DejaVuSans.ttf"), 20)
    top = margin
    bottom = margin + header + h + gap
    left, right = margin, margin + w + gap
    canvas = Image.new("RGB", (2 * margin + 2 * w + gap, bottom + header + h + margin), "white")
    draw = ImageDraw.Draw(canvas)
    panels = [
        (original, "original", "Original image", f"seed {seed:02d} | {row['image']}", left, top),
        (error, "errors", f"Errors | error rate: {error_rate:.2f}%",
         "Wrong pixels / all pixels | GT colors; correct: white", right, top),
        (predicted, "predicted", "Predicted mask", f"image mIoU: {score*100:.4f}%", left, bottom),
        (target, "ground_truth", "Ground-truth mask",
         "Al / Al3Ni / Primary Si / Eutectic Si", right, bottom),
    ]
    boxes = {}
    for array, key, title, subtitle, x, y in panels:
        if draw.textlength(title, font=title_font) > w or draw.textlength(subtitle, font=detail_font) > w:
            raise ValueError(f"Title would exceed panel width: {title}")
        draw.text((x, y + 4), title, font=title_font, fill="black", anchor="lt")
        draw.text((x, y + 42), subtitle, font=detail_font, fill="#444444", anchor="lt")
        canvas.paste(Image.fromarray(array), (x, y + header))
        boxes[key] = [x, y + header, x + w, y + header + h]
    canvas.save(path)
    saved = rgb(path)
    for array, key, _, _, _, _ in panels:
        x0, y0, x1, y1 = boxes[key]
        if not np.array_equal(saved[y0:y1, x0:x1], array):
            raise ValueError(f"Pixel verification failed: {path}: {key}")
    if not np.all(saved[:, left+w:right] == 255):
        raise ValueError("Column gap not clear")
    if not np.all(saved[top+header+h:bottom] == 255):
        raise ValueError("Row gap not clear")
    return dict(seed=seed, image=row["image"], error_pixels=error_count, total_pixels=pixels,
                error_rate_percent=error_rate, error_rate_display=f"{error_rate:.2f}%",
                image_miou=score, panel_boxes=boxes, sha256=digest(path), pixel_verification="passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=PROJECT / "results/model_results_0926_2")
    args = parser.parse_args()
    result = args.result.resolve()
    suite = json.loads((result / "suite.json").read_text())
    if suite["state"] != "completed":
        raise ValueError("Wait until the suite finishes before replacing its figures")
    current = result / "valid_mask_comparison"
    metrics = sorted((result / "metrics").glob("seed_*.json"))
    plans = []
    protected = set(metrics) | {result / "suite.json", result / f"{result.name}.md", result / "seed_results.csv"}
    for metric in metrics:
        data = json.loads(metric.read_text())
        seed = data["seed"]
        rows = data["per_image"]
        if len(rows) != 20:
            raise ValueError(f"Expected 20 validation images for seed {seed}")
        for row in rows:
            relative = Path(f"seed_{seed:02d}") / row["image"]
            original = PROJECT / "dataset/valid/images" / row["image"]
            target = PROJECT / "dataset/valid/masks" / row["image"].replace("_image.png", "_mask.png")
            predicted = result / "predictions" / relative
            protected.update([original, target, predicted])
            plans.append((seed, row, relative, original, predicted, target))
    if len(plans) != 200 or {p.relative_to(current) for p in current.rglob("*.png")} != {p[2] for p in plans}:
        raise ValueError("Expected exactly the same 200 existing comparison images")
    source_hashes = {str(p): digest(p) for p in protected}
    original_hashes = {str(p.relative_to(current)): digest(p) for p in current.rglob("*") if p.is_file()}
    stamp = datetime.now(timezone.utc).strftime("layout_%Y%m%dT%H%M%S_%fZ")
    history = result / "comparison_history" / stamp
    stage = history / "new_valid_mask_comparison"
    stage.mkdir(parents=True, exist_ok=False)
    records = []
    for seed, row, relative, original, predicted, target in tqdm(plans, desc="Reformat valid comparisons", unit="image"):
        out = stage / relative
        out.parent.mkdir(exist_ok=True)
        records.append(render(out, rgb(original), rgb(predicted), rgb(target), row, seed))
    for path, before in source_hashes.items():
        if digest(Path(path)) != before:
            raise ValueError(f"Protected input changed: {path}")
    for relative, before in original_hashes.items():
        if digest(current / relative) != before:
            raise ValueError(f"Original figure changed during rendering: {relative}")
    backup = history / "valid_mask_comparison"
    shutil.copy2(Path(__file__), history / "rebuild.py")
    record = dict(created_utc=stamp, image_count=len(records), title_font_px=28, detail_font_px=20,
                  gap_px=24, margin_px=16, header_px=76, resized=False,
                  error_rate_definition="100 * count(predicted_class != ground_truth_class) / (height * width)",
                  original_backup=str(backup), preserved_inputs_sha256=source_hashes,
                  original_comparison_sha256=original_hashes, images=records)
    (history / "layout.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    with (history / "error_rates.csv").open("w", newline="") as stream:
        fields = ["seed", "image", "error_pixels", "total_pixels", "error_rate_percent", "error_rate_display", "image_miou"]
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    (stage / "README.md").write_text(
        "# 비교 이미지 표시\n\n"
        "- 우측 상단 error rate = 틀린 픽셀 수 / 전체 픽셀 수 × 100 (%).\n"
        "- 오류 위치는 정답 색, 맞게 예측한 위치는 흰색입니다.\n"
        "- 제목 28px, 패널 간격 24px. 이미지와 마스크는 원래 해상도 그대로입니다.\n"
        f"- 이전 이미지와 재생성 기록: `{backup.relative_to(result)}`\n")
    current.rename(backup)
    try:
        stage.rename(current)
    except BaseException:
        backup.rename(current)
        raise
    print(f"Updated {len(records)} figures: {current}")
    print(f"Originals preserved: {backup}")
    print(f"Pixel verification and stored image mIoU verification: {len(records)}/{len(records)} passed")


if __name__ == "__main__":
    main()
