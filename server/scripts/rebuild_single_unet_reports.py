"""Rebuild reports from saved validation masks using the layout in spec.md.

No model import, inference, training, or edits to existing results.
Every invocation allocates new numbered result directories.
"""
from __future__ import annotations

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
PALETTE = np.array([[76, 178, 76], [25, 76, 153], [204, 204, 204], [153, 127, 76]], dtype=np.uint8)
CLASS_NAMES = ["Primary Si", "Al3Ni", "Eutectic Si", "Al"]


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def allocate():
    parent = PROJECT / "results"
    prefix = datetime.now(timezone.utc).strftime("model_results_%m%d_")
    n = 1 + max((int(p.name[len(prefix):]) for p in parent.glob(prefix + "*")
                 if p.is_dir() and p.name[len(prefix):].isdigit()), default=0)
    while True:
        out = parent / f"{prefix}{n}"
        try:
            out.mkdir(exist_ok=False)
            return out
        except FileExistsError:
            n += 1


def rgb(path):
    with Image.open(path) as im:
        if "A" in im.getbands() and np.any(np.asarray(im.getchannel("A")) != 255):
            raise ValueError(f"Non-opaque input: {path}")
        return np.array(im.convert("RGB"))


def label_ids(array):
    matches = np.all(array[..., None, :] == PALETTE, axis=-1)
    if not matches.any(-1).all():
        raise ValueError("Mask contains colors outside the four-class palette")
    return matches.argmax(-1)


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def comparison(path, original, prediction, target, title, score):
    assert original.shape == prediction.shape == target.shape
    wrong = np.any(prediction != target, axis=-1)
    error_panel = np.full_like(target, 255)
    error_panel[wrong] = target[wrong]
    # Full native resolution; in particular do not erase isolated one-pixel errors by resizing.
    h, w = target.shape[:2]
    margin, gap, header, top, footer = 14, 16, 38, 52, 52
    x0, x1 = margin, margin + w + gap
    y0, y1 = top + header, top + header + h + gap + header
    canvas = Image.new("RGB", (2 * margin + 2 * w + gap, y1 + h + footer), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 12), f"{title} | image mIoU: {score:.4f}%", font=font(22), fill="black")
    panels = [(original, "Original image", x0, y0),
              (error_panel, "Errors: GT color / correct: white", x1, y0),
              (prediction, "Predicted mask", x0, y1),
              (target, "Ground-truth mask", x1, y1)]
    boxes = {}
    for array, label, x, y in panels:
        draw.text((x, y - 30), label, font=font(20), fill="black")
        canvas.paste(Image.fromarray(array), (x, y))
        boxes[label] = [x, y, x + w, y + h]
    legend_y = y1 + h + 16
    for i, (label, color) in enumerate(zip(CLASS_NAMES, PALETTE)):
        x = margin + i * ((canvas.width - 2 * margin) // 4)
        draw.rectangle((x, legend_y, x + 19, legend_y + 19), fill=tuple(map(int, color)))
        draw.text((x + 27, legend_y - 2), label, font=font(18), fill="black")
    canvas.save(path)
    # Verify actual saved pixels in all four panels, including every correct/error pixel.
    saved = rgb(path)
    for array, label, x, y in panels:
        assert np.array_equal(saved[y:y+h, x:x+w], array), (path, label)
    assert np.all(error_panel[~wrong] == 255)
    assert np.array_equal(error_panel[wrong], target[wrong])
    return dict(image_width=w, image_height=h, error_pixels=int(wrong.sum()),
                panel_boxes=boxes, comparison_sha256=digest(path), pixel_verification="passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT / "results/model_results_0924_4")
    args = parser.parse_args()
    source = args.source.resolve()
    manifest = {r["image_id"]: r for r in csv.DictReader((PROJECT / "data/processed/manifest.csv").open())}
    validation_ids = {i for i, r in manifest.items() if r["source_split"] == "valid"}
    entries = read_json(source / "provenance.json")
    plans = []
    for entry in entries:
        src = (source / entry["directory"]).resolve()
        request = read_json(src / "request.json")
        assert request["completed"] and request["split"] == "valid"
        records = {r["image_id"]: r for r in csv.DictReader((src / "per_image.csv").open())}
        assert set(records) == validation_ids and len(records) == 20
        plans.append(dict(entry=entry, source=src, request=request, rows=records,
                          summary=read_json(src / "summary.json"), output=allocate()))
    tracked = {PROJECT / "AGENTS.md", PROJECT / "spec.md", source / "provenance.json"}
    for plan in plans:
        tracked.update(plan["source"] / name for name in ["summary.json", "per_image.csv", "request.json"])
        for image_id in validation_ids:
            tracked.update([Path(manifest[image_id]["image_path"]), Path(manifest[image_id]["mask_path"]),
                            plan["source"] / "masks" / f"{image_id}_mask.png"])
    original_hashes = {str(p): digest(p) for p in tracked}
    progress = tqdm(total=sum(len(p["rows"]) for p in plans), desc="Rebuild 2x2 valid comparisons", unit="image")
    for plan in plans:
        src, out, cfg = plan["source"], plan["output"], plan["request"]["config"]
        name = cfg["experiment"]
        (out / "valid_mask_comparison").mkdir()
        (out / "masks").mkdir()
        for filename in ["summary.json", "per_image.csv", "request.json"]:
            shutil.copy2(src / filename, out / filename)
        rendered, scores = [], []
        for image_id, row in plan["rows"].items():
            info = manifest[image_id]
            pred_path = src / "masks" / f"{image_id}_mask.png"
            original, pred, target = rgb(Path(info["image_path"])), rgb(pred_path), rgb(Path(info["mask_path"]))
            pred_labels, target_labels = label_ids(pred), label_ids(target)
            cm = np.bincount((target_labels * 4 + pred_labels).ravel(), minlength=16).reshape(4, 4)
            assert np.array_equal(cm, json.loads(row["confusion_matrix"]))
            tp = cm.diagonal(); union = cm.sum(0) + cm.sum(1) - tp
            score = np.divide(tp, union, out=np.ones(4), where=union > 0).mean() * 100
            assert np.isclose(score, float(row["image_miou"]))
            scores.append(score)
            filename = f"{image_id}.png"
            verified = comparison(out / "valid_mask_comparison" / filename, original, pred, target,
                                  f"{name} x{cfg['scale']} | {image_id}", score)
            shutil.copy2(pred_path, out / "masks" / pred_path.name)
            assert digest(out / "masks" / pred_path.name) == original_hashes[str(pred_path)]
            rendered.append(dict(image_id=image_id, image_miou=float(score), file=f"valid_mask_comparison/{filename}", **verified))
            progress.update(1)
        assert np.isclose(np.mean(scores), plan["summary"]["image_miou"])
        record = dict(source_report=str(source), source_evaluation=str(src), preserved_request=plan["request"],
                      generator=str(Path(__file__).resolve()), generator_sha256=digest(Path(__file__).resolve()),
                      training_started=False, inference_started=False, resized=False,
                      layout=[["Original image", "GT colors only at wrong pixels; correct pixels white"],
                              ["Predicted mask", "Ground-truth mask"]],
                      image_miou_definition="mean over 20 images of the four-class IoU mean; empty union=1",
                      inputs_sha256=original_hashes, images=rendered)
        (out / "provenance.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        s = plan["summary"]
        lines = [f"# {out.name} — {name} ({cfg['scale']}배) 결과", "",
                 "AGENTS.md 및 spec.md의 비교 이미지 형식에 맞춰 기존 평가 결과를 다시 구성했다. 저장된 예측만 사용했으며 모델 재학습·추론은 수행하지 않았다.", "",
                 "## 학습 및 평가", "",
                 "- 모델: EfficientNet-B4 ImageNet encoder + 기본 U-Net, CE + Dice.",
                 f"- 설정: seed {cfg['seed']}, {plan['request']['update']:,} updates, 유효 배치 {cfg['train']['effective_batch']}.",
                 "- 평가: official valid 20장, 단일 모델, TTA 없음, tiled core512/halo64/50% overlap/Gaussian 결합.",
                 "- 주 지표: 각 이미지의 4클래스 IoU 평균을 먼저 계산한 뒤 20장 평균. 정답·예측 모두 없는 클래스는 IoU=1.",
                 f"- **image mIoU: {s['image_miou']:.4f}%**. 저장된 마스크에서 재계산해 기존 점수와 일치함을 확인했다.", "",
                 "| image mIoU | empty=0 참고 | 빈 클래스 제외 참고 | 경계 IoU 1px | 외곽 16px 오류율 |", "|---:|---:|---:|---:|---:|",
                 f"| {s['image_miou']:.4f} | {s['miou_zero']:.4f} | {s['miou_exclude']:.4f} | {s['boundary_iou_1px']:.4f} | {s['outer_error_16px']:.4f} |", "",
                 "## 검증 이미지 구성", "",
                 "모든 비교 이미지는 `valid_mask_comparison/`에 저장했다. 원본 해상도를 그대로 유지해 1픽셀 오차도 보존했다.", "",
                 "| 왼쪽 | 오른쪽 |", "|---|---|", "| Original image | 틀린 픽셀은 정답 클래스 색, 맞는 픽셀은 흰색 |", "| Predicted mask | Ground-truth mask |", "",
                 "우측 상단은 오차 위치의 **정답 색상**을 표시한다. Eutectic Si 정답은 회색 RGB(204,204,204), 맞게 예측한 배경은 흰색 RGB(255,255,255)이다. 원본 위에 빨간색을 덧씌우지 않는다.", "",
                 "| 검증 이미지 | image mIoU (%) | 2×2 비교 |", "|---|---:|---|"]
        for r in rendered:
            lines.append(f"| {r['image_id']} | {r['image_miou']:.4f} | [비교 이미지]({r['file']}) |")
        lines += ["", "## 모델 간 비교", "", "| 모델 | image mIoU (%) | 새 보고서 |", "|---|---:|---|"]
        for p in plans:
            dest = p["output"]
            lines.append(f"| {p['request']['config']['experiment']} | {p['summary']['image_miou']:.4f} | [{dest.name}](../{dest.name}/{dest.name}.md) |")
        lines += ["", "## 결과 해석", "",
                  "2배 확대 S1은 S0보다 +2.5129%p, 3배 확대 S2는 S1보다 +0.2501%p였다. S2는 20장 중 15장에서 S1보다 높지만, A71의 맨 왼쪽 Primary Si 2픽셀 오검출이 빈 클래스 규칙에 따라 전체 평균에 1.25%p 영향을 준다. 경계는 개선됐으나 외곽 오류와 Al3Ni/Eutectic Si 혼동은 남아 있다.",
                  "단일 seed 결과이므로 배율 우열의 안정성은 추가 검증이 필요하다. [상세 분석 보고서](../model_results_0924_4/model_results_0924_4.md)에 이미지별 비교와 진단을 보존했다.", "",
                  "## 보존 및 재현", "",
                  f"- 기존 평가: `{src}`.",
                  f"- 학습 run: `{plan['request']['run']}`.",
                  f"- checkpoint SHA256: `{plan['request']['checkpoint_sha256']}`.",
                  "- `masks/`: 기존 예측 마스크의 원본 복사본. `summary.json`, `per_image.csv`, `request.json`도 그대로 보존했다.",
                  "- `provenance.json`: 입력 hash, 패널 좌표, 픽셀 검증 결과, 생성 스크립트를 기록했다.",
                  "- `python -B scripts/rebuild_single_unet_reports.py`를 실행하면 새 번호의 결과 폴더를 생성한다. 기존 결과·학습 소스·가중치를 수정하지 않는다."]
        (out / f"{out.name}.md").write_text("\n".join(lines) + "\n")
    progress.close()
    for path, expected in original_hashes.items():
        assert digest(Path(path)) == expected, f"Input changed while rendering: {path}"
    for p in plans:
        out = p["output"]
        assert len(list((out / "valid_mask_comparison").glob("*.png"))) == 20
        (out / "verification.json").write_text(json.dumps(dict(status="passed", images=20, all_four_panels_pixel_exact=True,
                   incorrect_pixels_match_gt_color=True, correct_pixels_are_white=True, metrics_unchanged=True,
                   source_inputs_unchanged=True, training_started=False, inference_started=False), indent=2) + "\n")
        print(out / f"{out.name}.md")


if __name__ == "__main__":
    main()
