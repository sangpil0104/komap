# model_results_0924_2 — S1 (2배) 결과

AGENTS.md 및 spec.md의 비교 이미지 형식에 맞춰 기존 평가 결과를 다시 구성했다. 저장된 예측만 사용했으며 모델 재학습·추론은 수행하지 않았다.

## 학습 및 평가

- 모델: EfficientNet-B4 ImageNet encoder + 기본 U-Net, CE + Dice.
- 설정: seed 2026, 8,000 updates, 유효 배치 8.
- 평가: official valid 20장, 단일 모델, TTA 없음, tiled core512/halo64/50% overlap/Gaussian 결합.
- 주 지표: 각 이미지의 4클래스 IoU 평균을 먼저 계산한 뒤 20장 평균. 정답·예측 모두 없는 클래스는 IoU=1.
- **image mIoU: 73.1106%**. 저장된 마스크에서 재계산해 기존 점수와 일치함을 확인했다.

| image mIoU | empty=0 참고 | 빈 클래스 제외 참고 | 경계 IoU 1px | 외곽 16px 오류율 |
|---:|---:|---:|---:|---:|
| 73.1106 | 71.8606 | 72.3748 | 40.0461 | 11.7153 |

## 검증 이미지 구성

모든 비교 이미지는 `valid_mask_comparison/`에 저장했다. 원본 해상도를 그대로 유지해 1픽셀 오차도 보존했다.

| 왼쪽 | 오른쪽 |
|---|---|
| Original image | 틀린 픽셀은 정답 클래스 색, 맞는 픽셀은 흰색 |
| Predicted mask | Ground-truth mask |

우측 상단은 오차 위치의 **정답 색상**을 표시한다. Eutectic Si 정답은 회색 RGB(204,204,204), 맞게 예측한 배경은 흰색 RGB(255,255,255)이다. 원본 위에 빨간색을 덧씌우지 않는다.

| 검증 이미지 | image mIoU (%) | 2×2 비교 |
|---|---:|---|
| A117_step4_sample2 | 86.6005 | [비교 이미지](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 60.9174 | [비교 이미지](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 66.5117 | [비교 이미지](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 76.9672 | [비교 이미지](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 63.9441 | [비교 이미지](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 78.2318 | [비교 이미지](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 75.8253 | [비교 이미지](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 80.4243 | [비교 이미지](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 82.5324 | [비교 이미지](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 77.7516 | [비교 이미지](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 66.1615 | [비교 이미지](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 88.0438 | [비교 이미지](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 80.1635 | [비교 이미지](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 77.9506 | [비교 이미지](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 68.9799 | [비교 이미지](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 71.1284 | [비교 이미지](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 55.8540 | [비교 이미지](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 84.9546 | [비교 이미지](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 49.8177 | [비교 이미지](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 69.4516 | [비교 이미지](valid_mask_comparison/A7_step1_sample4.png) |

## 모델 간 비교

| 모델 | image mIoU (%) | 새 보고서 |
|---|---:|---|
| S0 | 70.5977 | [model_results_0924_1](../model_results_0924_1/model_results_0924_1.md) |
| S1 | 73.1106 | [model_results_0924_2](../model_results_0924_2/model_results_0924_2.md) |
| S2 | 73.3606 | [model_results_0924_3](../model_results_0924_3/model_results_0924_3.md) |

## 결과 해석

2배 확대 S1은 S0보다 +2.5129%p, 3배 확대 S2는 S1보다 +0.2501%p였다. S2는 20장 중 15장에서 S1보다 높지만, A71의 맨 왼쪽 Primary Si 2픽셀 오검출이 빈 클래스 규칙에 따라 전체 평균에 1.25%p 영향을 준다. 경계는 개선됐으나 외곽 오류와 Al3Ni/Eutectic Si 혼동은 남아 있다.
단일 seed 결과이므로 배율 우열의 안정성은 추가 검증이 필요하다. [상세 분석 보고서](../model_results_0924_4/model_results_0924_4.md)에 이미지별 비교와 진단을 보존했다.

## 보존 및 재현

- 기존 평가: `/Data3/sppark/komap/results/model_results_0924_4/02_S1_seed2026`.
- 학습 run: `/Data3/sppark/komap/experiments/single_unet_v3/outputs/runs/S1_x2_seed2026_official_20260924T060537_447445Z`.
- checkpoint SHA256: `3c610157bf4fe5bf8604e882bae5fcccf7d26c5990c52c5fd93ae0a67b41a3d6`.
- `masks/`: 기존 예측 마스크의 원본 복사본. `summary.json`, `per_image.csv`, `request.json`도 그대로 보존했다.
- `provenance.json`: 입력 hash, 패널 좌표, 픽셀 검증 결과, 생성 스크립트를 기록했다.
- `python -B scripts/rebuild_single_unet_reports.py`를 실행하면 새 번호의 결과 폴더를 생성한다. 기존 결과·학습 소스·가중치를 수정하지 않는다.
