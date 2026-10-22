"""Export final Model364 predictions without rerunning the routing search."""
import json
import os
import shutil
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from local_image_metrics import class_ious

NAMES = ["Al3Ni", "Primary Si", "Al", "Eutectic Si"]
PALETTE = np.array([(25, 76, 153), (76, 178, 76), (153, 127, 76), (204, 204, 204)], dtype=np.uint8)


def comparison(image_path, prediction, target, score, output):
    with Image.open(image_path) as im:
        original = im.convert("RGB")
    w, h = original.size
    bar = 76
    canvas = Image.new("RGB", (2 * w, 2 * (h + bar)), "white")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    error = np.full((h, w, 3), 255, dtype=np.uint8)
    wrong = prediction != target
    error[wrong] = PALETTE[target[wrong]]
    panels = [original, Image.fromarray(error), Image.fromarray(PALETTE[prediction]), Image.fromarray(PALETTE[target])]
    titles = ["Original image", f"Error | image mIoU: {score * 100:.4f}%\n100% - image mIoU: {(1 - score) * 100:.4f}%",
              "Predicted mask", "Ground-truth mask"]
    draw = ImageDraw.Draw(canvas)
    for index, (panel, title) in enumerate(zip(panels, titles)):
        x, y = (index % 2) * w, (index // 2) * (h + bar)
        draw.multiline_text((x + 8, y + 6), title, fill="black", font=font, spacing=5)
        canvas.paste(panel, (x, y + bar))
    canvas.save(output)


def reserve_result(root):
    root.mkdir(parents=True, exist_ok=True)
    date = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%m%d")
    for number in range(1, 1000000):
        output = root / f"model_results_{date}_{number}"
        try:
            output.mkdir()
            return output
        except FileExistsError:
            continue
    raise RuntimeError("No unused result number")


def export_final(rows, names, predict, run_root, route_result):
    if len(rows) != 20 or len(names) != 20 or len(set(names)) != 20:
        raise ValueError("Exactly 20 distinct validation images required")
    predictions, metrics = [], []
    for name, row in tqdm(list(zip(names, rows)), desc="Model364 final masks", unit="image"):
        pred = predict(row).astype(np.uint8)
        target = row["item"]["target"]
        iou = class_ious(pred, target)
        predictions.append(pred)
        metrics.append({"image": name, "image_miou": float(iou.mean()), "class_iou": dict(zip(NAMES, iou.tolist()))})
    score = float(np.mean([r["image_miou"] for r in metrics]))
    expected = route_result["raw_best_valid_image_mean_miou_empty0"]
    if not np.isclose(score, expected, atol=1e-10, rtol=0):
        raise ValueError(f"Export/selected route score mismatch: {score} vs {expected}")
    root = Path(os.environ["KOMAP_RESULTS_ROOT"])
    output = reserve_result(root)
    for directory in ("valid_mask_comparison", "predicted_masks", "predicted_labels"):
        (output / directory).mkdir()
    for name, row, pred, metric in tqdm(list(zip(names, rows, predictions, metrics)), desc="Export comparison", unit="image"):
        stem = name.removesuffix("_image")
        image_path = Path(run_root) / "data/valid/images" / f"{stem}_image.png"
        Image.fromarray(pred).save(output / "predicted_labels" / f"{stem}.png")
        Image.fromarray(PALETTE[pred]).save(output / "predicted_masks" / f"{stem}_mask.png")
        comparison(image_path, pred, row["item"]["target"], metric["image_miou"],
                   output / "valid_mask_comparison" / f"{stem}.png")
    record = {"model": "Model364 local image-mIoU policy", "image_miou": score,
              "empty_union": 0, "class_order": NAMES, "run": str(run_root), "images": metrics,
              "route": route_result, "validation_used_for_selection": True, "final_gate_selection_in_this_run": False}
    (output / "metrics.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    for filename in ("provenance.json", "environment.freeze.txt", "local_patches.diff"):
        path = Path(run_root) / filename
        if path.exists():
            shutil.copy2(path, output / filename)
    lines = ["# Model364 — local image mIoU", "", f"Validation image mIoU: **{score * 100:.6f}%** (20 images)",
             "", "각 이미지의 4개 클래스 IoU 평균 후 20장 평균. union=0이면 IoU=0. global mIoU는 계산하지 않습니다.",
             "일부 선행 체크포인트 선택과 원본 조합 개발에 valid를 사용했습니다. 이번 최종 조합은 기록된 고정 설정이며 독립 테스트 점수가 아닙니다.",
             "원본 대비 중간 모델 선택도 image mIoU(empty=0)로 통일했습니다. 원본 기록 점수는 참고값이며 새 모델의 점수가 아닙니다.",
             "이번 재개 실행은 Model361 scale3 추론 overlap을 170→114로 수정했습니다(native_tile171, stride57). 원본 추론 인자와 다른 로컬 설정입니다.",
             f"실행 및 소스 스냅샷: `{run_root}`", "",
             "비교: 좌상 Original / 우상 Error / 좌하 Prediction / 우하 Ground truth.",
             "Error는 맞은 픽셀을 흰색, 틀린 픽셀을 정답 색으로 표시합니다. 100%−image mIoU는 픽셀 오류율이 아닙니다.",
             "", "| Image | image mIoU | Comparison |", "|---|---:|---|"]
    for metric in metrics:
        stem = metric["image"].removesuffix("_image")
        lines.append(f"| {stem} | {metric['image_miou'] * 100:.4f}% | [image](valid_mask_comparison/{stem}.png) |")
    (output / f"{output.name}.md").write_text("\n".join(lines) + "\n")
    (Path(run_root) / "result_path.txt").write_text(str(output) + "\n")
    print(f"RESULT_DIRECTORY={output}", flush=True)
    return output
