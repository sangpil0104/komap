"""Data-derived figures; mask panels remain at original pixel resolution."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core import CLASS_NAMES, PALETTE, gray_from_rgb


def font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def error_rgb(target, prediction):
    error = np.full((*target.shape, 3), 255, dtype=np.uint8)
    wrong = target != prediction
    error[wrong] = PALETTE[target[wrong]]
    return error


def comparison(path, original, target, prediction, image_id, score):
    height, width = target.shape
    margin, gap, header, top, footer = 14, 16, 38, 52, 76
    canvas_width = max(2 * width + 2 * margin + gap, 1050)
    canvas = Image.new("RGB", (canvas_width, top + 2 * (header + height) + gap + footer), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 12), f"{image_id} | image mIoU: {score:.4f}% (empty=1)", fill="black", font=font(22))
    positions = [(margin, top + header), (margin + width + gap, top + header),
                 (margin, top + 2 * header + height + gap),
                 (margin + width + gap, top + 2 * header + height + gap)]
    arrays = [original, error_rgb(target, prediction), PALETTE[prediction], PALETTE[target]]
    titles = ["Original image", "Errors: GT color; correct = white", "Predicted mask", "Ground-truth mask"]
    for position, array, title in zip(positions, arrays, titles):
        canvas.paste(Image.fromarray(array), position)
        draw.text((position[0], position[1] - 30), title, fill="black", font=font(19))
    y = canvas.height - footer + 12
    for class_id, name in enumerate(CLASS_NAMES):
        x = margin + 240 * class_id
        draw.rectangle((x, y, x + 18, y + 18), fill=tuple(PALETTE[class_id]))
        draw.text((x + 26, y - 2), name, fill="black", font=font(18))
    draw.text((margin, y + 30), "Gray errors = Eutectic Si (204); white = correct (255).", fill="black", font=font(17))
    canvas.save(path)
    # Verify the actual saved panels, including the frequently confused top-right panel.
    with Image.open(path) as reopened:
        for (x, y), expected in zip(positions, arrays):
            actual = np.asarray(reopened.crop((x, y, x + width, y + height)))
            if not np.array_equal(actual, expected):
                raise RuntimeError(f"Saved panel differs from source pixels: {path}")
    return {"image_id": image_id, "native_panel_size": [width, height],
            "panels_verified": 4, "error_rule": "wrong=GT palette, correct=white"}


def histogram(path, original, thresholds, mapping):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = np.bincount(gray_from_rgb(original).ravel(), minlength=256)
    figure, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
    edges = [-0.5, *thresholds, 255.5]
    for index, class_id in enumerate(mapping):
        axis.axvspan(edges[index], edges[index + 1], color=PALETTE[class_id] / 255,
                     alpha=0.28, label=f"Bin {index}: {CLASS_NAMES[class_id]}")
    axis.plot(np.arange(256), counts, color="black", linewidth=1)
    for cut in thresholds:
        axis.axvline(cut, color="crimson", linestyle="--", linewidth=1)
    axis.set(xlim=(-0.5, 255.5), xlabel="Image intensity (0-255)", ylabel="Pixel count",
             title="Image-specific thresholds: " + ", ".join(f"{t:g}" for t in thresholds))
    axis.legend(fontsize=8)
    figure.savefig(path, dpi=130)
    plt.close(figure)


def write_report(directory, model, summary, rows, model_path, elapsed):
    lines = ["# 이미지별 밝기 임계값 모델 결과", "",
             "## 모델과 평가 조건", "",
             "- 모델: **Adaptive Multi-Otsu (4개 밝기 구간)**. 신경망 및 학습 백본 없음, CPU 실행.",
             "- 원본 크기의 8-bit 명암 히스토그램에서 이미지마다 임계값 3개를 계산합니다.",
             "- train 70장만 사용해 밝기 구간→상 대응 24개 순열 중 train image mIoU가 가장 높은 하나를 고릅니다.",
             "- 검증 이미지의 정답은 점수와 비교 그림 생성에만 사용하며 임계값/대응 선택에는 사용하지 않습니다.",
             "- 명암값이 4종류 미만이면 train 이미지에서 구한 임계값의 중앙값으로 대체합니다.",
             "- 크기 변경, 딥러닝, 앙상블, TTA, 형태학적 후처리는 없습니다.",
             f"- 모델 경로: `{model_path}` (평가 시점 사본: [model.json](model.json)).",
             f"- 검증 이미지 수: {len(rows)}장. 평가·결과 저장 경과 시간: {elapsed:.1f}초.", "",
             "## 밝기 구간과 상의 대응", "", "| 밝기 구간 | 대응 상 |", "|---|---|"]
    for index, interval in enumerate(["I ≤ T1", "T1 < I ≤ T2", "T2 < I ≤ T3", "T3 < I"]):
        lines.append(f"| {interval} | {CLASS_NAMES[model['bin_to_class'][index]]} |")
    lines.extend(["", "T1, T2, T3는 각 입력 이미지에서 새로 계산하며, 구간→상 대응은 train에서 정한 것을 고정합니다.",
                  "", "## 검증 점수 (%)", "",
                  "이미지마다 4개 클래스 IoU를 평균한 뒤 20개 이미지에 대해 평균합니다. 상별 값 역시 이미지별 해당 상 IoU의 평균입니다. 전체 픽셀을 합친 global mIoU는 사용하지 않습니다.",
                  "", "기본값은 TP+FP+FN=0인 상의 IoU=1입니다. 정답은 없지만 예측이 있는 상은 IoU=0입니다. 비교용으로 0 처리와 제외 방식도 함께 저장합니다.",
                  "", "| 빈 클래스 처리 | image mIoU | Primary Si | Al3Ni | Eutectic Si | Al |",
                  "|---|---:|---:|---:|---:|---:|"])
    for policy, name in [("one", "IoU=1 (기본)"), ("zero", "IoU=0"), ("exclude", "평균에서 제외")]:
        values = [summary[f"image_miou_{policy}"]] + [summary[f"class_{c}_iou_{policy}"] for c in range(4)]
        lines.append("| " + name + " | " + " | ".join("N/A" if v is None else f"{v:.4f}" for v in values) + " |")
    lines.extend(["", "제외 방식의 상별 평균은 그 상의 union>0인 이미지에서만 계산됩니다.",
                  "", "## 이미지별 임계값 및 결과", "",
                  "| 이미지 | T1 | T2 | T3 | 임계값 계산 | image mIoU (%) | 비교 그림 | 히스토그램 |",
                  "|---|---:|---:|---:|---|---:|---|---|"])
    for row in rows:
        name = row["image_id"]
        lines.append(f"| {name} | {row['threshold_1']:g} | {row['threshold_2']:g} | {row['threshold_3']:g} | "
                     f"{row['threshold_source']} | {row['image_miou_one']:.4f} | "
                     f"[비교](valid_mask_comparison/{name}.png) | [분포](histograms/{name}.png) |")
    lines.extend(["", "## 비교 그림 읽는 방법", "",
                  "좌측 상단: 원본, 우측 상단: 맞는 픽셀은 흰색·틀린 픽셀은 정답 상의 색, 좌측 하단: 예측, 우측 하단: 정답입니다. 각 패널은 원본 해상도로 저장하고 저장된 픽셀을 재검사합니다.",
                  "", "## 해석과 한계", "",
                  "동일 이미지에서 명암이 같은 픽셀은 언제나 같은 상으로 분류됩니다. Primary Si와 Eutectic Si처럼 명암이 겹치는 상, 조명 편차, 한 상이 빠진 이미지는 오류 원인이 될 수 있습니다. 이 모델은 경계의 형태나 주변 조직을 학습하지 않으므로 테두리 문제가 해결된 것으로 해석하면 안 됩니다.",
                  "", "밝기 기준선 성능을 확인하는 실험입니다. 다른 모델과는 동일한 검증 20장, 이미지 평균, 빈 클래스 처리 조건에서만 비교해야 합니다.",
                  "", "- [요약 수치](summary.json), [이미지별 수치](per_image.csv), [픽셀 혼동행렬 및 임계값](per_image.json)",
                  "- [출처 기록](provenance.json), [그림 픽셀 검증](verification.json), `sources_at_creation/` 실행 소스 사본", ""])
    (directory / f"{directory.name}.md").write_text("\n".join(lines), encoding="utf-8")
