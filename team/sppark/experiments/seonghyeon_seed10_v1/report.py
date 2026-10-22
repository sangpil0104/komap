"""Image-first four-class metrics and native-resolution comparison panels."""
import csv
from datetime import datetime, timezone
from pathlib import Path
import statistics

import numpy as np
from PIL import Image, ImageDraw

from runtime import duration, read_json, write_json
from vendor.common import PALETTE, image_miou

CLASSES = ["Al", "Al3Ni", "Primary Si", "Eutectic Si"]


def scores(pred, target):
    excluded, one, iou = image_miou(pred, target)
    return dict(image_miou=float(one), image_miou_exclude=float(excluded),
                image_miou_zero=float(np.nan_to_num(iou).mean()),
                class_iou=np.nan_to_num(iou, nan=1.0).tolist(),
                absent_classes=np.flatnonzero(np.isnan(iou)).tolist())


def comparison(path, original, pred, target, score):
    h, w = target.shape
    if original.shape != (h, w, 3) or pred.shape != target.shape:
        raise ValueError("Comparison image shape mismatch")
    error = np.full((h, w, 3), 255, dtype=np.uint8)
    wrong = pred != target
    error[wrong] = PALETTE[target[wrong]]
    header = 36
    canvas = Image.new("RGB", (w * 2, (h + header) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    panels = [(original, "Original image", 0, 0),
              (error, "Errors: GT color; correct: white", w, 0),
              (PALETTE[pred], f"Predicted mask | image mIoU {score * 100:.4f}%", 0, h + header),
              (PALETTE[target], "Ground-truth mask", w, h + header)]
    for pixels, label, x, y in panels:
        draw.text((x + 8, y + 10), label, fill="black")
        canvas.paste(Image.fromarray(pixels), (x, y + header))
    canvas.save(path)


def allocate_results(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now(timezone.utc).strftime("model_results_%m%d_")
    n = 1 + max((int(p.name[len(prefix):]) for p in root.glob(prefix + "*")
                 if p.name[len(prefix):].isdigit()), default=0)
    while True:
        out = root / f"{prefix}{n}"
        try:
            out.mkdir()
            return out
        except FileExistsError:
            n += 1


def seed_report(result, seed, rows, run):
    if len(rows) != 20 or len({row["image"] for row in rows}) != 20:
        raise ValueError("Exactly 20 distinct validation images required")
    summary = dict(seed=seed, image_miou=statistics.mean(r["image_miou"] for r in rows),
                   image_miou_exclude=statistics.mean(r["image_miou_exclude"] for r in rows),
                   image_miou_zero=statistics.mean(r["image_miou_zero"] for r in rows),
                   per_class_iou=np.mean([r["class_iou"] for r in rows], axis=0).tolist(),
                   run=str(Path(run).resolve()), checkpoint=str(Path(run).resolve() / "last.pt"),
                   per_image=rows)
    write_json(Path(run) / "valid.json", summary)
    write_json(Path(result) / "metrics" / f"seed_{seed:02d}.json", summary)
    return summary


def suite_report(result, jobs, state, elapsed):
    result = Path(result)
    completed = []
    for job in jobs:
        metric = result / "metrics" / f"seed_{job['seed']:02d}.json"
        if job["state"] == "completed" and metric.exists():
            completed.append(read_json(metric))
    completed.sort(key=lambda r: r["image_miou"], reverse=True)
    lines = [f"# seonghyeon_v1 시드 10개 실험 — {state}", "",
             f"성공 {len(completed)}/{len(jobs)} · 전체 경과 시간 {duration(elapsed)}", "",
             "설정: U-Net + EfficientNet-B4 ImageNet, tw4000, crop1536, batch2, "
             "16,000 iterations, AdamW(lr=0.0003, wd=0.01), CE+Dice. train 70장만 사용.",
             "각 GPU에서 독립 시드 하나를 실행하며 GPU 0–5에서 최대 6개 병렬 실행.",
             "모든 시드는 마지막 체크포인트를 D4 TTA로 평가. valid로 중간 체크포인트를 선택하지 않음.", "",
             "주 지표 = 이미지별 4클래스 IoU 평균 → valid 20장 평균. empty union=1. "
             "빈 클래스 제외/0 처리도 참고 열로 제공. 모든 점수 단위는 %.",
             "이전 seonghyeon 보고서는 빈 클래스를 제외했으므로 비교할 때 '제외 참고' 열을 사용.", "",
             "| 순위 | seed | image mIoU | 제외 참고 | 0 참고 | Al | Al3Ni | Primary Si | Eutectic Si |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for rank, r in enumerate(completed, 1):
        values = [r[k] for k in ("image_miou", "image_miou_exclude", "image_miou_zero")]
        values += r["per_class_iou"]
        lines.append(f"| {rank} | {r['seed']} | " + " | ".join(f"{v*100:.4f}" for v in values) + " |")
    if completed:
        values = [r["image_miou"] * 100 for r in completed]
        sd = f"{statistics.stdev(values):.4f}" if len(values) > 1 else "N/A"
        lines += ["", f"완료 시드 평균 {statistics.mean(values):.4f}% · 표본 표준편차 {sd} · "
                  f"최소 {min(values):.4f}% · 최대 {max(values):.4f}%", "",
                  "동일 valid 20장으로 정렬한 개발 검증 순위이며 test 성능을 뜻하지 않음."]
    lines += ["", "## 시드별 상태", "", "| seed | 상태 | GPU | 체크포인트 / 로그 |",
              "|---:|---|---:|---|"]
    for j in jobs:
        lines.append(f"| {j['seed']} | {j['state']} | {j.get('gpu', '-')} | `{j['run']}` |")
    lines += ["", "## 저장 파일", "",
              "- `valid_mask_comparison/seed_XX/`: 각 시드의 valid 20장, 총 200장.",
              "- 좌상 Original image / 우상 오류 위치 / 좌하 predicted mask / 우하 ground-truth mask.",
              "- 정답 픽셀은 흰색, 오류 픽셀은 해당 위치의 정답 색. 초록색 오류는 실제 Primary Si.",
              "- `predictions/seed_XX/`: 원본 해상도의 RGB 예측 마스크.",
              "- `metrics/seed_XX.json`: 이미지별·클래스별 점수 및 체크포인트 경로.",
              "- `suite.json`: 설정, 실행 폴더와 상태. 실행 폴더에는 소스 스냅샷과 환경 기록이 있음.", ""]
    if completed:
        lines += ["## valid 20장 비교", "",
                  "각 링크는 해당 이미지의 image mIoU(%)와 2×2 비교 사진을 연결합니다.", ""]
        ordered = sorted(completed, key=lambda r: r["seed"])
        lines += ["| 이미지 | " + " | ".join(f"seed {r['seed']}" for r in ordered) + " |",
                  "|---|" + "---:|" * len(ordered)]
        indices = [{r["image"]: r for r in metric["per_image"]} for metric in ordered]
        for name in sorted(indices[0]):
            cells = []
            for metric, index in zip(ordered, indices):
                value = index.get(name)
                if value is None or "image_miou" not in value:
                    cells.append("—")
                    continue
                filename = Path(name).stem + ".png"
                cells.append(f"[{value['image_miou']*100:.2f}](valid_mask_comparison/"
                             f"seed_{metric['seed']:02d}/{filename})")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    path = result / f"{result.name}.md"
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines))
    tmp.replace(path)
    with (result / "seed_results.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "image_miou", "image_miou_exclude", "image_miou_zero", *CLASSES])
        for r in completed:
            writer.writerow([r["seed"], r["image_miou"], r["image_miou_exclude"],
                             r["image_miou_zero"], *r["per_class_iou"]])
    return path
