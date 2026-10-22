from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .common import PROJECT, read_json, write_json


def allocate_results():
    parent = PROJECT / "results"
    parent.mkdir(exist_ok=True)
    prefix = datetime.now(timezone.utc).strftime("model_results_%m%d_")
    existing = []
    for p in parent.glob(prefix + "*"):
        suffix = p.stem[len(prefix):]
        if suffix.isdigit():
            existing.append(int(suffix))
    number = max(existing, default=0) + 1
    while True:
        folder = parent / f"{prefix}{number}"
        try:
            folder.mkdir(exist_ok=False)
            return folder
        except FileExistsError:
            number += 1


def make_report(evaluations):
    items = []
    for path in evaluations:
        path = Path(path).resolve()
        request = read_json(path / "request.json")
        if not request["completed"]:
            raise ValueError(f"Incomplete evaluation: {path}")
        items.append((path, request, read_json(path / "summary.json")))
    if not items:
        raise ValueError("No evaluation results to report")
    folder = allocate_results()
    lines = [f"# {folder.name} — single_unet_v3", "",
             "EfficientNet-B4 (torchvision ImageNet) + 기본 U-Net, 독립 학습한 단일 체크포인트.",
             "학습: CE + Dice, seed별 고정 예산 마지막 가중치. 검증으로 체크포인트를 선택하지 않음.",
             "S0/S1/S2 입력 배율은 1/2/3. S3는 선택 배율에서 경계 CE 가중치만 0.2 추가.",
             "빈 클래스 기본값은 IoU=1. 각 행의 주 지표 정책을 함께 기록하며 세 정책을 모두 보고함.",
             "모든 점수는 원본 해상도에서 이미지별 계산 후 평균. Global mIoU는 사용하지 않음.",
             "Boundary IoU는 클래스가 양쪽 모두 없으면 제외하며 원본 1/2/4px 폭으로 계산.",
             "test는 정답이 없으므로 마스크만 생성하고 점수는 보고하지 않음.", "",
             "| 실행 | split | 배율 | 추론 | 주 정책 | image mIoU | empty=1 | empty=0 | exclude |",
             "|---|---|---:|---|---|---:|---:|---:|---:|"]
    manifest, flat = [], []
    for index, (source, request, result) in enumerate(items, 1):
        cfg, inference = request["config"], request["config"]["inference"]
        label = f"{index:02d}_{cfg['experiment']}_seed{cfg['seed']}"
        dest = folder / label
        shutil.copytree(source, dest)
        policy = cfg["evaluation"]["empty_union"]
        mode = f"{inference['mode']}, pad={inference['pad']}, halo={inference['halo']}, offset={inference['offset']}, D4={inference['d4']}"
        def val(key):
            value = result.get(key)
            return "—" if value is None else f"{value:.4f}"
        lines.append(f"| {label} | {cfg['data']['split']}/{request['split']} | {cfg['scale']} | {mode} | {policy} | "
                     f"{val('image_miou')} | {val('miou_one')} | {val('miou_zero')} | {val('miou_exclude')} |")
        manifest.append(dict(directory=label, source=str(source), **request))
        flat.append(dict(run=label, mode=mode, policy=policy, **result))
    groups = {}
    for _, request, result in items:
        if "image_miou" not in result:
            continue
        cfg = request["config"]
        key = json.dumps({k: v for k, v in cfg.items() if k != "seed"}, sort_keys=True)
        groups.setdefault(key, []).append((cfg, result))
    repeated = [v for v in groups.values() if len({x[0]["seed"] for x in v}) > 1]
    if repeated:
        lines += ["", "## Seed 반복 (예측을 합치지 않은 독립 모델)", "",
                  "| 설정 | seeds | image mIoU 평균 | 표본 표준편차 |", "|---|---|---:|---:|"]
        for group in repeated:
            seeds = [cfg["seed"] for cfg, _ in group]
            if len(set(seeds)) != len(seeds):
                lines.append("동일 seed의 중복 평가가 포함된 설정은 반복 평균에서 제외했다.")
                continue
            values = [r["image_miou"] for _, r in group]
            cfg = group[0][0]
            lines.append(f"| {cfg['experiment']} x{cfg['scale']} {cfg['data']['split']} / {cfg['inference']} | "
                         f"{seeds} | {np.mean(values):.4f} | {np.std(values, ddof=1):.4f} |")
    lines += ["", "## 마스크 비교 및 진단", "",
              "각 실행 폴더의 `comparisons/*.png`: 원본 / 오류(빨강) / 예측 / 정답.",
              "`masks/*.png`: 원본 크기 RGBA 팔레트 마스크. `per_image.csv`: 이미지별 점수·클래스 혼동·경계 및 외곽 진단.",
              "`summary.json`: 20장(official valid)의 이미지 평균. fold 실험은 해당 fold의 실제 이미지 수로 평균.",
              "`outer/inner_*_16/32/64px`: 이미지 외곽과 내부를 분리한 점수.",
              "`boundary_error_*`: GT 경계 주변 오류율. `boundary_iou_*`: GT/예측 클래스별 내부 경계 띠 IoU.",
              "", "## 재현 정보", "",
              "실행별 `request.json`에 checkpoint SHA256, 전체 설정, 배율, seed, 평가 조건을 저장했다.",
              "학습 run에는 소스 snapshot, 데이터 hash, 업데이트별 checkpoint와 training.jsonl을 보존한다.",
              "기존 보고서의 exclude 점수와 empty=1 점수를 직접 비교하지 않는다. Plain과 D4도 구분한다.",
              "GPU 수 변경 재개는 모델/optimizer/update를 보존하지만 난수 분배와 BN 실행 순서가 달라 bitwise 재현은 아니다.",
              "공식 split 일부 A-ID 중복과 반복 validation 선택 영향을 고려해 최종 후보는 독립 seed/group fold로 확인한다."]
    # A reference is explicitly marked as a different input/training/inference recipe.
    reference = PROJECT / "experiments/accuracy_v1/outputs/evaluations/baseline/official/valid/plain_1/metrics_summary.json"
    if reference.exists():
        saved = read_json(reference)
        lines += ["", "## 기존 PhaseCNN 참고값", "",
                  "기존 보존된 Plain 평가: 입력 gray/zscore/CDF, full-frame 추론. 새 실험과 동일 조건의 통제 비교는 아님.",
                  f"empty=1: {saved.get('mIoU_one', '미기록')}, empty=0: {saved.get('mIoU_zero', '미기록')}, "
                  f"exclude: {saved.get('mIoU_exclude', '미기록')} (%)"]
    (folder / f"{folder.name}.md").write_text("\n".join(lines) + "\n")
    write_json(folder / "provenance.json", manifest)
    fields = sorted(set().union(*(r.keys() for r in flat)))
    with (folder / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    return folder
