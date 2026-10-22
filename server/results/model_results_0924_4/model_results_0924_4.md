# 이미지별 밝기 임계값 모델 결과

## 모델과 평가 조건

- 모델: **Adaptive Multi-Otsu (4개 밝기 구간)**. 신경망 및 학습 백본 없음, CPU 실행.
- 원본 크기의 8-bit 명암 히스토그램에서 이미지마다 임계값 3개를 계산합니다.
- train 70장만 사용해 밝기 구간→상 대응 24개 순열 중 train image mIoU가 가장 높은 하나를 고릅니다.
- 검증 이미지의 정답은 점수와 비교 그림 생성에만 사용하며 임계값/대응 선택에는 사용하지 않습니다.
- 명암값이 4종류 미만이면 train 이미지에서 구한 임계값의 중앙값으로 대체합니다.
- 크기 변경, 딥러닝, 앙상블, TTA, 형태학적 후처리는 없습니다.
- 모델 경로: `/Data3/sppark/komap/experiments/adaptive_threshold_v1/runs/20260924_110356_808dc62f/model.json` (평가 시점 사본: [model.json](model.json)).
- 검증 이미지 수: 20장. 평가·결과 저장 경과 시간: 13.0초.

## 밝기 구간과 상의 대응

| 밝기 구간 | 대응 상 |
|---|---|
| I ≤ T1 | Primary Si |
| T1 < I ≤ T2 | Al3Ni |
| T2 < I ≤ T3 | Eutectic Si |
| T3 < I | Al |

T1, T2, T3는 각 입력 이미지에서 새로 계산하며, 구간→상 대응은 train에서 정한 것을 고정합니다.

## 검증 점수 (%)

이미지마다 4개 클래스 IoU를 평균한 뒤 20개 이미지에 대해 평균합니다. 상별 값 역시 이미지별 해당 상 IoU의 평균입니다. 전체 픽셀을 합친 global mIoU는 사용하지 않습니다.

기본값은 TP+FP+FN=0인 상의 IoU=1입니다. 정답은 없지만 예측이 있는 상은 IoU=0입니다. 비교용으로 0 처리와 제외 방식도 함께 저장합니다.

| 빈 클래스 처리 | image mIoU | Primary Si | Al3Ni | Eutectic Si | Al |
|---|---:|---:|---:|---:|---:|
| IoU=1 (기본) | 45.9125 | 53.6458 | 51.8622 | 1.9420 | 76.1999 |
| IoU=0 | 45.9125 | 53.6458 | 51.8622 | 1.9420 | 76.1999 |
| 평균에서 제외 | 45.9125 | 53.6458 | 51.8622 | 1.9420 | 76.1999 |

제외 방식의 상별 평균은 그 상의 union>0인 이미지에서만 계산됩니다.

## 이미지별 임계값 및 결과

| 이미지 | T1 | T2 | T3 | 임계값 계산 | image mIoU (%) | error rate (%) | 비교 그림 | 히스토그램 |
|---|---:|---:|---:|---|---:|---:|---|---|
| A117_step4_sample2 | 114 | 166.5 | 206 | multi_otsu | 46.5559 | 30.6346 | [비교](valid_mask_comparison/A117_step4_sample2.png) | [분포](histograms/A117_step4_sample2.png) |
| A11_step1_sample4 | 65.5 | 136 | 200.5 | multi_otsu | 45.2838 | 26.5772 | [비교](valid_mask_comparison/A11_step1_sample4.png) | [분포](histograms/A11_step1_sample4.png) |
| A11_step2_sample4 | 55.5 | 126.5 | 199 | multi_otsu | 27.6650 | 42.4441 | [비교](valid_mask_comparison/A11_step2_sample4.png) | [분포](histograms/A11_step2_sample4.png) |
| A125_step1_sample4 | 116 | 162.5 | 202.5 | multi_otsu | 40.2995 | 35.2699 | [비교](valid_mask_comparison/A125_step1_sample4.png) | [분포](histograms/A125_step1_sample4.png) |
| A153_step3_sample3 | 85.5 | 149.5 | 193.5 | multi_otsu | 48.2720 | 38.6186 | [비교](valid_mask_comparison/A153_step3_sample3.png) | [분포](histograms/A153_step3_sample3.png) |
| A170_step4_sample3 | 80.5 | 148 | 195 | multi_otsu | 53.3291 | 25.0676 | [비교](valid_mask_comparison/A170_step4_sample3.png) | [분포](histograms/A170_step4_sample3.png) |
| A181_step1_sample1 | 128.5 | 171.5 | 212.5 | multi_otsu | 41.3092 | 30.7404 | [비교](valid_mask_comparison/A181_step1_sample1.png) | [분포](histograms/A181_step1_sample1.png) |
| A186_step2_sample2 | 72.5 | 143.5 | 198 | multi_otsu | 62.0474 | 13.1000 | [비교](valid_mask_comparison/A186_step2_sample2.png) | [분포](histograms/A186_step2_sample2.png) |
| A191_step4_sample3 | 120 | 169 | 207.5 | multi_otsu | 45.8178 | 28.5878 | [비교](valid_mask_comparison/A191_step4_sample3.png) | [분포](histograms/A191_step4_sample3.png) |
| A40_step3_sample3 | 79.5 | 142.5 | 197.5 | multi_otsu | 50.6383 | 26.8757 | [비교](valid_mask_comparison/A40_step3_sample3.png) | [분포](histograms/A40_step3_sample3.png) |
| A42_step2_sample4 | 133.5 | 177.5 | 211.5 | multi_otsu | 47.0513 | 23.2168 | [비교](valid_mask_comparison/A42_step2_sample4.png) | [분포](histograms/A42_step2_sample4.png) |
| A44_step4_sample3 | 69.5 | 130.5 | 186.5 | multi_otsu | 51.9383 | 22.8982 | [비교](valid_mask_comparison/A44_step4_sample3.png) | [분포](histograms/A44_step4_sample3.png) |
| A55_step2_sample1 | 75 | 131 | 192 | multi_otsu | 46.6302 | 29.2032 | [비교](valid_mask_comparison/A55_step2_sample1.png) | [분포](histograms/A55_step2_sample1.png) |
| A59_step3_sample1 | 76 | 150 | 206 | multi_otsu | 54.4729 | 23.2370 | [비교](valid_mask_comparison/A59_step3_sample1.png) | [분포](histograms/A59_step3_sample1.png) |
| A63_step1_sample4 | 134.5 | 181.5 | 215.5 | multi_otsu | 53.7601 | 23.1910 | [비교](valid_mask_comparison/A63_step1_sample4.png) | [분포](histograms/A63_step1_sample4.png) |
| A67_step1_sample1 | 90.5 | 148.5 | 204.5 | multi_otsu | 42.5442 | 32.8389 | [비교](valid_mask_comparison/A67_step1_sample1.png) | [분포](histograms/A67_step1_sample1.png) |
| A71_step2_sample1 | 67.5 | 133.5 | 190.5 | multi_otsu | 25.7944 | 42.4614 | [비교](valid_mask_comparison/A71_step2_sample1.png) | [분포](histograms/A71_step2_sample1.png) |
| A72_step4_sample4 | 75 | 139 | 198 | multi_otsu | 35.7363 | 32.8948 | [비교](valid_mask_comparison/A72_step4_sample4.png) | [분포](histograms/A72_step4_sample4.png) |
| A74_step3_sample4 | 57 | 126.5 | 190 | multi_otsu | 51.9569 | 27.4216 | [비교](valid_mask_comparison/A74_step3_sample4.png) | [분포](histograms/A74_step3_sample4.png) |
| A7_step1_sample4 | 66.5 | 143.5 | 207.5 | multi_otsu | 47.1470 | 27.7046 | [비교](valid_mask_comparison/A7_step1_sample4.png) | [분포](histograms/A7_step1_sample4.png) |

## 비교 그림 읽는 방법

좌측 상단: 원본, 우측 상단: 맞는 픽셀은 흰색·틀린 픽셀은 정답 상의 색, 좌측 하단: 예측, 우측 하단: 정답입니다. 각 패널은 원본 해상도로 저장하고 저장된 픽셀을 재검사합니다.

우측 상단 제목에 **error rate = 틀린 픽셀 수 / 전체 픽셀 수 × 100**을 표시합니다. 최근 결과와 동일하게 제목 28px, 부제목 20px, 패널 간격 24px, 여백 16px를 적용했습니다. 원본·예측·정답은 축소하지 않았습니다.

[표시 변경 검증 기록](comparison_history/layout_20260927_1/layout.json) · [오류율 CSV](comparison_history/layout_20260927_1/error_rates.csv) · [기존 비교 사진](comparison_history/layout_20260927_1/valid_mask_comparison/) · [기존 보고서](comparison_history/layout_20260927_1/model_results_0924_4.md)

## 해석과 한계

동일 이미지에서 명암이 같은 픽셀은 언제나 같은 상으로 분류됩니다. Primary Si와 Eutectic Si처럼 명암이 겹치는 상, 조명 편차, 한 상이 빠진 이미지는 오류 원인이 될 수 있습니다. 이 모델은 경계의 형태나 주변 조직을 학습하지 않으므로 테두리 문제가 해결된 것으로 해석하면 안 됩니다.

밝기 기준선 성능을 확인하는 실험입니다. 다른 모델과는 동일한 검증 20장, 이미지 평균, 빈 클래스 처리 조건에서만 비교해야 합니다.

- [요약 수치](summary.json), [이미지별 수치](per_image.csv), [픽셀 혼동행렬 및 임계값](per_image.json)
- [출처 기록](provenance.json), [최초 평가의 그림 검증](verification.json), `sources_at_creation/` 실행 소스 사본
