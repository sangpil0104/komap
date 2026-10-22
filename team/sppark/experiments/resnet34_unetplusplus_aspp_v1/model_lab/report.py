from __future__ import annotations

from pathlib import Path
from model_lab.workspace import RESULTS_DIR, OUTPUTS_DIR, read_json


def generate_report(output_file: Path | None = None) -> Path:
    """Generates or updates model_results_0921_1.md in results/."""
    if output_file is None:
        output_file = RESULTS_DIR / "model_results_0921_1.md"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plain_summary_path = OUTPUTS_DIR / "evaluations/plain/summary.json"
    d4_summary_path = OUTPUTS_DIR / "evaluations/d4/summary.json"
    best_pt_path = OUTPUTS_DIR / "runs/best.pt"

    plain_metrics = read_json(plain_summary_path) if plain_summary_path.is_file() else None
    d4_metrics = read_json(d4_summary_path) if d4_summary_path.is_file() else None

    # Status detection
    if d4_metrics is not None and plain_metrics is not None:
        status_str = "완료 (D4 및 Plain 검증 평가 완료)"
    elif best_pt_path.is_file():
        status_str = "학습 완료 (체크포인트 생성됨, 최종 평가 대기)"
    else:
        status_str = "모델 빌드 완료 (학습 준비 완료 — 사용자가 직접 실행)"

    lines = [
        "# model_results_0921_1 — ResNet34 U-Net++ + residual ASPP 실험 결과",
        "",
        f"**상태**: {status_str}",
        "",
        "## 1. 실험 개요 및 모델 아키텍처",
        "",
        "- **모델명**: ResNet34 U-Net++ + Residual ASPP (`resnet34_unetplusplus_aspp_v1`)",
        "- **문제 정의** ([problem.pdf](../../problem.pdf)): 100장 광학현미경(OM) 이미지 분할 (Train 70장, Valid 20장, Test 10장).",
        "- **클래스 정의 (4 Phase)**:",
        "  - `0 Al` (알루미늄 모상 / 갈색)",
        "  - `1 Primary Si, Sip` (초정 실리콘 / 초록색)",
        "  - `2 Al3Ni` (Al3Ni 금속간화합물 / 파란색)",
        "  - `3 Eutectic Si, Sie` (공정 실리콘 / 흰색)",
        "- **구조적 특징**:",
        "  1. **ResNet34 Backbone**: ImageNet 사전학습 가중치 활용, Stride-1 Native Resolution Stem(32채널)을 결합하여 미세 상 경계 보존.",
        "  2. **Residual ASPP Bottleneck**: 최하단 인코더(Stage 5, 512채널)에 Dilation Rate $d \\in [1, 2, 4, 8]$의 Atrous Conv 분기 및 Global Context Pooling, 잔차 연결($x + \\text{ASPP}(x)$) 적용.",
        "  3. **Nested Dense Decoder (U-Net++)**: $X^{i, j}$ 격자 구조로 밀집된 Dense Skip Connection을 연결하여 인코더와 디코더 간 semantic gap 완화.",
        "  4. **총 파라미터 수**: 약 27,481,396개 (27.48M).",
        "",
        "## 2. 학습 환경 및 하이퍼파라미터",
        "",
        "| 항목 | 설정값 |",
        "|---|---|",
        "| 입력 채널 | 3채널 (`[raw gray, z-score, empirical CDF]`) |",
        "| 최적화 알고리즘 | AdamW (차등 학습률: Encoder `1e-4`, Decoder/ASPP `3e-4`) |",
        "| 손실 함수 | Combined Loss: CrossEntropy (1.0) + SoftDice (1.0) |",
        "| 정밀도 | BF16 Mixed Precision (`torch.autocast`) |",
        "| 유효 배치 크기 | 8 (GPU 8장: 각 1개, GPU 1장: 8회 Gradient Accumulation) |",
        "| 지수이동평균 (EMA) | Decay 0.99 |",
        "| 데이터 증강 | D4 (8-way Dihedral 회전/반전) + Photometric (대비, 밝기, 감마) |",
        "| 단계별 일정 | Stage 0 (512패치 2,000 updates) $\\to$ Stage 1 (768패치 1,500 updates) $\\to$ Stage 2 (원해상도 1,000 updates) |",
        "",
        "## 3. 검증 평가 결과 (Validation 20장)",
        "",
    ]

    if plain_metrics or d4_metrics:
        lines += [
            "| 평가 방식 | 이미지 평균 mIoU | Global mIoU | Primary Si IoU | Al3Ni IoU | Eutectic Si IoU | Al IoU |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        if plain_metrics:
            c = plain_metrics["class_ious"]
            lines.append(
                f"| Plain (Single pass) | **{plain_metrics['image_mean_miou']:.2f}%** | {plain_metrics['global_miou']:.2f}% | "
                f"{c['Primary_Si']:.2f}% | {c['Al3Ni']:.2f}% | {c['Eutectic_Si']:.2f}% | {c['Al']:.2f}% |"
            )
        if d4_metrics:
            c = d4_metrics["class_ious"]
            lines.append(
                f"| D4 TTA (8-way) | **{d4_metrics['image_mean_miou']:.2f}%** | {d4_metrics['global_miou']:.2f}% | "
                f"{c['Primary_Si']:.2f}% | {c['Al3Ni']:.2f}% | {c['Eutectic_Si']:.2f}% | {c['Al']:.2f}% |"
            )
        lines.append("")
    else:
        lines += [
            "현재 모델 구현 및 smoke 테스트가 완료되었으며, 사용자가 학습을 실행하면 아래 표에 측정값이 갱신됩니다.",
            "",
            "| 평가 방식 | 이미지 평균 mIoU | Global mIoU | Primary Si IoU | Al3Ni IoU | Eutectic Si IoU | Al IoU | 상태 |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
            "| Plain (Single pass) | - | - | - | - | - | - | 학습 실행 대기 |",
            "| D4 TTA (8-way) | - | - | - | - | - | - | 학습 실행 대기 |",
            "",
        ]

    lines += [
        "## 4. 실행 및 재현 방법",
        "",
        "### 4.1 원클릭 전체 실행 (학습 $\\to$ 검증 $\\to$ 테스트 예측 $\\to$ 결과 리포트)",
        "```bash",
        "cd /Data3/sppark/komap",
        "conda activate komap",
        "",
        "# GPU 8장 사용 시",
        "bash experiments/resnet34_unetplusplus_aspp_v1/run_suite.sh 8",
        "",
        "# GPU 1장 사용 시",
        "bash experiments/resnet34_unetplusplus_aspp_v1/run_suite.sh 1",
        "```",
        "",
        "### 4.2 개별 단계 실행",
        "```bash",
        "# 1) 학습만 실행",
        "python experiments/resnet34_unetplusplus_aspp_v1/run.py train --gpus 8",
        "",
        "# 2) 검증만 실행 (Plain 및 D4 TTA)",
        "python experiments/resnet34_unetplusplus_aspp_v1/run.py evaluate",
        "python experiments/resnet34_unetplusplus_aspp_v1/run.py evaluate --d4",
        "",
        "# 3) 테스트 10장 예측 PNG 생성",
        "python experiments/resnet34_unetplusplus_aspp_v1/run.py predict",
        "",
        "# 4) 리포트 파일 갱신",
        "python experiments/resnet34_unetplusplus_aspp_v1/run.py report",
        "```",
        "",
        "## 5. 산출물 저장 경로",
        "",
        "- **체크포인트**: `experiments/resnet34_unetplusplus_aspp_v1/outputs/runs/best.pt` (최고 검증 모델), `last.pt`",
        "- **검증 세부 지표 (CSV)**: `experiments/resnet34_unetplusplus_aspp_v1/outputs/evaluations/`",
        "- **테스트 예측 PNG**: `experiments/resnet34_unetplusplus_aspp_v1/outputs/predictions/*_mask.png` (정확한 4색 팔레트, RGBA 255)",
        "- **학습 로그**: `experiments/resnet34_unetplusplus_aspp_v1/outputs/runs/resolved_config.json`",
        "",
        "---",
        "*본 실험은 [spec.md](../../spec.md) 및 [AGENTS.md](../../AGENTS.md) 지침에 따라 기존 모델, 소스코드 및 이전 결과를 일절 수정하지 않고 독립 환경에서 구축되었습니다.*",
    ]

    output_file.write_text("\n".join(lines), encoding="utf-8")
    return output_file

