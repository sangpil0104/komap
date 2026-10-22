import csv
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch

from .common import (CLASS_NAMES, CLASSES, PALETTE, digest, eta, fresh_results,
                     read_json, records, snapshot, write_json)
from .data import gray, labels
from .engine import load_checkpoint
from .inference import predict
from .metrics import measure, summarize
from .model import SmallUNet
from .panels import comparison


DESCRIPTIONS = {
    "A": "ResNet18 ImageNet 초기값 → CE+Dice 지도학습",
    "B": "같은 초기값 → 원본 가림 복원 자기지도학습 → encoder만 이전 → A와 같은 CE+Dice 지도학습",
    "C": "A의 마지막 모델·optimizer를 이어 받아 추가 지도학습; B의 사전학습+분할학습 계산 시간에 맞춤",
}


def evaluate(checkpoint, cfg, device, arm, suite, results_root=None):
    started = time.monotonic()
    checkpoint = Path(checkpoint)
    state = load_checkpoint(checkpoint)
    if state.get("task") != "seg" or not state.get("finished"):
        raise ValueError("Evaluation requires a completed segmentation checkpoint")
    model = SmallUNet(freeze_bn=cfg["model"]["freeze_encoder_bn_stats"]).to(device)
    model.load_state_dict(state["model"])
    del state
    model.eval()
    directory = fresh_results(results_root)
    print(f"[evaluation {arm}] RESULTS={directory}", flush=True)
    snapshot(directory)
    for name in ("masks", "valid_mask_comparison"):
        (directory / name).mkdir()
    rows, checks = [], []
    for index, row in enumerate(records(cfg)["valid"]):
        prediction = predict(model, gray(row["image_path"]).astype(np.float32) / 255, cfg, device).argmax(0).astype(np.uint8)
        target = labels(row["mask_path"])
        metrics = {"image_id": row["image_id"], **measure(target, prediction, cfg["evaluation"])}
        rows.append(metrics)
        with Image.open(row["image_path"]) as image:
            original = np.asarray(image.convert("RGB"))
        name = row["image_id"] + ".png"
        Image.fromarray(PALETTE[prediction]).convert("RGBA").save(directory / "masks" / name)
        checks.append(comparison(directory / "valid_mask_comparison" / name, original, target,
                                 prediction, row["image_id"], metrics["image_miou"]))
        elapsed = time.monotonic() - started
        remaining = elapsed / (index + 1) * (19 - index)
        print(f"[valid {arm}] {index+1}/20 image_mIoU={metrics['image_miou']:.4f}% "
              f"elapsed={elapsed:.1f}s remaining~{remaining:.1f}s finish~{eta(remaining)}", flush=True)
    summary = summarize(rows, cfg["evaluation"]["empty_union"])
    write_json(directory / "summary.json", summary)
    write_json(directory / "per_image.json", rows)
    with (directory / "per_image.csv").open("x", newline="") as handle:
        fields = [key for key in rows[0] if key != "confusion_matrix"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_json(directory / "verification.json", {"saved_panels_verified": True, "images": checks})
    write_json(directory / "provenance.json", {"arm": arm, "suite": str(suite), "checkpoint": str(checkpoint),
               "checkpoint_sha256": digest(checkpoint), "config": cfg,
               "train_ids": [r["image_id"] for r in records(cfg)["train"]],
               "valid_ids": [r["image_id"] for r in records(cfg)["valid"]],
               "ssl_uses_valid_or_test": False})
    lines = [f"# 자기지도학습 비교 실험 {arm}", "", f"- 학습 백본: **ResNet18 (ImageNet IMAGENET1K_V1)**.",
             f"- 적용 내용: {DESCRIPTIONS[arm]}.",
             "- 단일 U-Net, 원해상도 skip, encoder BN 통계 고정, decoder GroupNorm. 입력 확대·TTA·경계 손실 없음.",
             "- 원본 중심 512px + halo64px 패치(아래 config가 최종 기준), D4 학습 증강. 마스크는 원본 크기로 평가.",
             "- train 70장만 학습에 사용. 자기지도 단계는 원본만 읽고, 가려진 실제 중심 픽셀의 L1 복원 손실로 학습.",
             "- B 분할 단계에서는 encoder만 이전하고 decoder·분류 head는 A와 같은 초기값에서 시작.",
             "- valid20은 평가에만 사용. 마지막 checkpoint를 고정하며 검증 점수로 선택하지 않음.",
             f"- 체크포인트: `{checkpoint}`. 실행 설정·소스는 provenance.json과 source/에 저장.", "",
             "## 점수 (%)", "",
             "이미지마다 4개 상 IoU를 평균하고 20개 이미지에 대해 다시 평균합니다. 상별 값도 이미지 평균입니다. global mIoU는 사용하지 않습니다.",
             "", "| image mIoU | Primary Si | Al3Ni | Eutectic Si | Al |", "|---:|---:|---:|---:|---:|",
             "| " + " | ".join(f"{summary[k]:.4f}" for k in ["image_miou"] + ["iou_" + c for c in CLASSES]) + " |", "",
             "정답과 예측이 모두 없는 상은 IoU=1, 오검출이 있으면 IoU=0입니다.",
             f"빈 클래스 0 처리 mIoU: {summary['miou_zero']:.4f}%. 제외 처리 mIoU: {summary['miou_exclude']:.4f}%.", "",
             "## 경계와 외곽 진단", "", "| 보조 지표 | 값 (%) |", "|---|---:|"]
    for key in summary:
        if key.startswith(("boundary_", "outer_", "inner_")):
            value = summary[key]
            lines.append(f"| {key} | {value:.4f} |" if value is not None else f"| {key} | N/A |")
    lines += ["", "boundary_error는 정답 상 경계에서 지정한 거리 이내의 오분류율입니다. outer_error는 이미지 외곽 띠의 오분류율입니다. 오류율은 낮을수록 좋습니다. boundary IoU는 클래스 내부 경계 띠(영상 프레임 포함)의 겹침이며, 양쪽에 없는 상은 이 보조 지표에서 제외합니다.",
              "", "## 이미지별 결과", "", "| 이미지 | image mIoU (%) | 비교 사진 |", "|---|---:|---|"]
    for row in rows:
        name = row["image_id"]
        lines.append(f"| {name} | {row['image_miou']:.4f} | [비교](valid_mask_comparison/{name}.png) |")
    lines += ["", "비교 사진: 좌상단 원본, 우상단 오류(정답 픽셀은 흰색·오답 픽셀은 정답 상의 색), 좌하단 예측, 우하단 정답. 저장된 네 패널을 픽셀 단위로 재검사했습니다.",
              "", "한 seed의 비교이므로 성능 차이의 안정성을 확정할 수 없습니다. 가림 복원 자체가 상 경계 개선을 보장하지 않습니다.", ""]
    (directory / f"{directory.name}.md").write_text("\n".join(lines))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"directory": str(directory), "wall_seconds": time.monotonic() - started, "summary": summary}


def comparison_report(suite, evaluations, stages, cfg, results_root=None):
    directory = fresh_results(results_root)
    write_json(directory / "summary.json", {a: e["summary"] for a, e in evaluations.items()})
    compute = {"A": stages["A_supervised"]["compute_seconds"],
               "B": stages["B_ssl"]["compute_seconds"] + stages["B_finetune"]["compute_seconds"],
               "C": stages["A_supervised"]["compute_seconds"] + stages["C_extra"]["compute_seconds"]}
    write_json(directory / "budget.json", {"compute_seconds": compute,
               "gpu_seconds": {a: v * stages['A_supervised']['world_size'] for a, v in compute.items()},
               "C_minus_B_seconds": compute["C"] - compute["B"],
               "definition": "sum of slowest-rank update compute intervals; excludes checkpoint/evaluation/data prefetch"})
    write_json(directory / "provenance.json", {"suite": str(suite), "evaluations": evaluations, "stages": stages, "config": cfg})
    lines = ["# 자기지도학습 A·B·C 종합 결과", "",
             "동일한 ResNet18 U-Net과 ImageNet 초기값으로 자기지도 사전학습의 효과를 비교했습니다. B의 복원 decoder는 버리고 encoder만 이전했습니다. C는 A를 이어 추가 지도학습하며, 학습 시간 비교에는 A의 계산 시간도 포함합니다.", "",
             "| 모델 | 방법 | image mIoU (%) | Primary Si | Al3Ni | Eutectic Si | Al | 학습 계산 시간 (s) |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ("A", "B", "C"):
        summary = evaluations[arm]["summary"]
        report = Path(evaluations[arm]["directory"])
        link = os.path.relpath(report / f"{report.name}.md", directory)
        values = [summary['image_miou']] + [summary['iou_' + c] for c in CLASSES]
        lines.append(f"| [{arm}]({link}) | {DESCRIPTIONS[arm]} | " + " | ".join(f"{v:.4f}" for v in values) + f" | {compute[arm]:.1f} |")
    diff = evaluations['B']['summary']['image_miou'] - evaluations['A']['summary']['image_miou']
    diff_control = evaluations['B']['summary']['image_miou'] - evaluations['C']['summary']['image_miou']
    lines += ["", f"- B−A image mIoU: {diff:+.4f}%p. B−C: {diff_control:+.4f}%p.",
              f"- C−B 학습 계산 시간: {compute['C']-compute['B']:+.3f}초. 실제 GPU 수를 곱한 GPU·초도 budget.json에 저장했습니다.",
              "- C는 A의 모델·optimizer를 이어 받고 추가 구간에서 cosine LR을 다시 시작합니다. B와 같게 만드는 것은 측정한 계산 시간이며, 업데이트 수·FLOPs·학습 경로까지 동일한 것은 아닙니다.",
              "- 계산 시간은 각 업데이트에서 가장 느린 rank의 전송·forward·backward·optimizer 구간 합계입니다. 데이터 prefetch, 체크포인트 저장, 평가 시간은 제외합니다. 목표 초과는 마지막 업데이트만큼 발생할 수 있습니다.",
              "- image mIoU는 각 이미지의 4개 클래스 IoU 평균을 valid20에서 평균합니다. 빈 클래스 IoU=1, 나머지 두 정책도 개별 보고서에 저장합니다.",
              "- 자기지도학습의 이득은 B−A뿐 아니라 B−C, Eutectic Si, 경계·외곽 오류를 함께 확인해야 합니다. 단일 seed 결과입니다.", "",
              "| 모델 | 경계 2px 오류 (%) | 외곽 32px 오류 (%) |", "|---|---:|---:|"]
    for arm in ("A", "B", "C"):
        s = evaluations[arm]["summary"]
        values = [s.get('boundary_error_2px'), s.get('outer_error_32px')]
        lines.append(f"| {arm} | " + " | ".join(f"{v:.4f}" if v is not None else "N/A" for v in values) + " |")
    lines += ["", "각 모델 보고서에 검증 20장의 원해상도 2×2 비교 사진이 있습니다. validation/test 원본은 자기지도 사전학습에서도 제외했습니다.", ""]
    (directory / f"{directory.name}.md").write_text("\n".join(lines))
    return directory
