"""Read-only audit of saved validation predictions; empty-union IoU is zero.

Run with /home/sppark/miniconda3/envs/komap/bin/python recalculate.py.
No training or inference is performed. All output stays beside this script.
"""
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PALETTE = np.array([[153, 127, 76], [25, 76, 153], [76, 178, 76], [204, 204, 204]], np.uint8)
CLASSES = ["Al", "Al3Ni", "Primary Si", "Eutectic Si"]


def read(path):
    return json.loads(path.read_text())


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def identifier(row):
    return row.get("image_id", row.get("image", "")).removesuffix("_image.png")


def score_cm(cm):
    cm = np.asarray(cm, dtype=np.int64)
    assert cm.shape == (4, 4) and np.all(cm >= 0)
    tp = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - tp
    iou = np.divide(tp, union, out=np.zeros(4), where=union > 0)
    return iou, np.flatnonzero(union == 0).tolist()


def label(rgb):
    ids = np.full(rgb.shape[:2], 255, np.uint8)
    for c, color in enumerate(PALETTE):
        ids[np.all(rgb == color, axis=-1)] = c
    assert np.all(ids < 4), "Unknown palette color"
    return ids


def rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def panel(image_id, original, pred, target, score):
    h, w = target.shape[:2]
    margin, gap, header, title_h = 16, 24, 52, 84
    errors = np.full_like(target, 255)
    wrong = np.any(pred != target, axis=-1)
    errors[wrong] = target[wrong]
    canvas = Image.new("RGB", (2*w + 2*margin + gap, 2*h + 2*header + 2*margin + gap + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    draw.text((margin, 8), f"EfficientNet-B4 U-Net seed 32 | {image_id}", font=font, fill="black")
    draw.text((margin, 44), f"image mIoU: {100*score:.6f}% | 100% - image mIoU: {100*(1-score):.6f}% | empty union = 0", font=font, fill="black")
    blocks = [(original, "Original image", margin, margin+title_h),
              (errors, "Errors: GT color; correct pixels: white", margin+w+gap, margin+title_h),
              (pred, "Predicted mask", margin, margin+title_h+header+h+gap),
              (target, "Ground-truth mask", margin+w+gap, margin+title_h+header+h+gap)]
    boxes = []
    for array, title, x, y in blocks:
        assert draw.textlength(title, font=font) <= w
        draw.text((x, y+8), title, font=font, fill="black")
        canvas.paste(Image.fromarray(array), (x, y+header))
        boxes.append([x, y+header, x+w, y+header+h])
    path = OUT / "valid_mask_comparison" / f"{image_id}.png"
    canvas.save(path)
    saved = rgb(path)
    for (array, *_), (x0, y0, x1, y1) in zip(blocks, boxes):
        assert np.array_equal(saved[y0:y1, x0:x1], array)
    return {"image_id": image_id, "image_miou_zero_percent": score*100,
            "complement_percent": (1-score)*100, "pixel_panels_verified": True,
            "sha256": digest(path)}


def main():
    manifest = list(csv.DictReader((ROOT / "data/processed/manifest.csv").open()))
    expected = {r["image_id"] for r in manifest if r["source_split"] == "valid"}
    assert len(expected) == 20
    # The user's example: three IoUs of 0.8 and an empty class => 0.6.
    example = np.array([[80, 10, 0, 0], [0, 80, 10, 0], [10, 0, 80, 0], [0, 0, 0, 0]])
    assert np.isclose(score_cm(example)[0].mean(), 0.6)
    paths = subprocess.check_output([
        "rg", "--files", "-uu", "experiments", "runs", "results",
        "-g", "*.json", "-g", "*.csv", "-g", "!**/.venv/**",
        "-g", "!**/source/**", "-g", "!**/vendor/**"], cwd=ROOT, text=True).splitlines()
    accepted = []
    rejected = []
    hashes = {}
    candidates = [p for p in paths if Path(p).name in {
        "per_image_metrics.csv", "per_image.csv", "per_image.json", "summary.json", "valid.json"}
        or ("results/model_results_0926_2/metrics/seed_" in p and p.endswith(".json"))]
    for relative in tqdm(sorted(candidates), desc="Recompute saved evaluations", unit="file"):
        p = ROOT / relative
        if OUT in p.parents or any(x in relative for x in ("smoke", "comparison_history", "report_history")):
            continue
        data = read(p) if p.suffix == ".json" else None
        if p.suffix == ".csv":
            rows = list(csv.DictReader(p.open()))
        else:
            rows = data if isinstance(data, list) else data.get("per_image", [])
        if not rows:
            continue
        if len(rows) != 20 or {identifier(r) for r in rows} != expected:
            rejected.append({"source": relative, "reason": "not the official 20-image validation set"})
            continue
        metric_rows = []
        methods = set()
        for row in sorted(rows, key=identifier):
            if "confusion_matrix" in row:
                cm = row["confusion_matrix"]
                iou, absent = score_cm(json.loads(cm) if isinstance(cm, str) else cm)
                methods.add("confusion_matrix")
            elif "absent_classes" in row and "class_iou" in row:
                iou = np.array(row["class_iou"], dtype=float)
                absent = row["absent_classes"]
                iou[absent] = 0
                methods.add("class_iou_and_absent_classes")
            else:
                break
            score = float(iou.mean()) * 100
            one = score + len(absent)*25
            for key, factor in [("mIoU_zero", 1), ("miou_zero", 1), ("image_miou_zero", 100)]:
                if key in row:
                    if key == "image_miou_zero" and "class_0_iou_zero" in row:
                        factor = 1  # Adaptive-threshold reports use percentages.
                    assert np.isclose(float(row[key])*factor, score, rtol=0, atol=1e-9), (relative, key)
            metric_rows.append({"image_id": identifier(row), "image_miou_zero_percent": score,
                                "image_miou_one_percent": one, "empty_classes": absent})
        if len(metric_rows) != 20:
            rejected.append({"source": relative, "reason": "duplicate display CSV lacks confusion matrices"})
            continue
        hashes[relative] = digest(p)
        result = {"source": relative, "image_miou_zero_percent": float(np.mean([r["image_miou_zero_percent"] for r in metric_rows])),
                  "image_miou_one_percent": float(np.mean([r["image_miou_one_percent"] for r in metric_rows])),
                  "empty_image_class_pairs": sum(len(r["empty_classes"]) for r in metric_rows),
                  "method": "+".join(sorted(methods)), "num_images": 20, "per_image": metric_rows}
        if isinstance(data, dict) and "checkpoint" in data:
            result["checkpoint"] = data["checkpoint"]
        accepted.append(result)
    accepted.sort(key=lambda r: r["image_miou_zero_percent"], reverse=True)
    assert accepted and np.isclose(accepted[0]["image_miou_zero_percent"], 78.97654566541209)
    dump("evaluation_records.json", accepted)
    with (OUT / "all_evaluations.csv").open("w", newline="") as f:
        fields = ["source", "image_miou_zero_percent", "image_miou_one_percent", "empty_image_class_pairs", "method", "num_images", "checkpoint"]
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(accepted)
    dump("excluded_records.json", rejected)
    seed32 = read(ROOT / "results/model_results_0926_2/metrics/seed_32.json")
    baseline = read(ROOT / "experiments/seonghyeon_upgrade_sangpil_v1/reports/seed32_baseline/confusion_matrices.json")
    (OUT / "valid_mask_comparison").mkdir(exist_ok=True)
    verified = []
    matrices = {}
    top_pixel_scores = {}
    # Independently verify the top three seed checkpoints using their saved masks.
    for seed in [32, 63, 11]:
        stored = read(ROOT / f"results/model_results_0926_2/metrics/seed_{seed:02d}.json")
        pixel_scores = []
        for row in tqdm(stored["per_image"], desc=f"Verify masks seed {seed:02d}", unit="image"):
            image_id = identifier(row)
            pred_path = ROOT / f"results/model_results_0926_2/predictions/seed_{seed:02d}" / row["image"]
            gt_path = ROOT / "dataset/valid/masks" / f"{image_id}_mask.png"
            pred_rgb, gt_rgb = rgb(pred_path), rgb(gt_path)
            pred, gt = label(pred_rgb), label(gt_rgb)
            cm = np.bincount((gt.astype(np.int64)*4+pred).ravel(), minlength=16).reshape(4, 4)
            iou, absent = score_cm(cm)
            score = float(iou.mean())
            assert np.isclose(score, row["image_miou_zero"], rtol=0, atol=1e-12)
            assert absent == row["absent_classes"]
            pixel_scores.append(score)
            for source in [pred_path, gt_path]:
                hashes[str(source.relative_to(ROOT))] = digest(source)
            if seed == 32:
                assert np.array_equal(cm, baseline[image_id])
                matrices[image_id] = cm.tolist()
                original = rgb(ROOT / "dataset/valid/images" / row["image"])
                verified.append(panel(image_id, original, pred_rgb, gt_rgb, score))
        top_pixel_scores[str(seed)] = float(np.mean(pixel_scores))*100
    dump("winner_confusion_matrices.json", matrices)
    dump("winner_panel_verification.json", verified)
    dump("source_sha256.json", hashes)
    assert all(digest(ROOT / name) == value for name, value in hashes.items())
    summary = {"metric": "mean over 20 images of four-class mean IoU; empty union = 0",
               "evaluation_records": len(accepted), "unique_metric_vectors": len({tuple(round(x["image_miou_zero_percent"], 10) for x in r["per_image"]) for r in accepted}),
               "records_by_family": dict(Counter("/".join(r["source"].split("/")[:2]) for r in accepted)),
               "winner": "EfficientNet-B4 U-Net seed 32, target_width=4000, D4 TTA",
               "winner_score_percent": top_pixel_scores["32"], "checkpoint": seed32["checkpoint"],
               "top_seed_pixel_verification": top_pixel_scores,
               "training_started": False, "existing_files_preserved": True,
               "limitations": ["Only saved official-validation evaluations were ranked; unscored intermediate checkpoints were not newly inferred.",
                               "Historical markdown-only uploads without per-image results or masks cannot be exactly recalculated."]}
    dump("summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
