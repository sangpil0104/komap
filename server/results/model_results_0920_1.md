# accuracy_v1 실험 결과 — model_results_0920_1

기존 저장 결과를 정리한 보고서입니다. 새 학습이나 추론을 실행하지 않았으며, 기존 모델·소스·결과를 변경하지 않았습니다.

## 1. 핵심 결과

- 학습 실험 4개(`fullframe`, `boundary`, `seed2027`, `seed2028`) 모두 4,500 updates 완료. 기존 모델과 3-seed 앙상블까지 비교했습니다.
- 이미지 평균 mIoU 최고: **seed2028 + D4, 78.7858%**. 기존 모델 + D4의 78.6865% 대비 **+0.0993%p**입니다.
- 최고 모델의 Global mIoU는 **81.5082%**입니다. Global mIoU 최고는 앙상블의 **81.6370%**로, 평가 방식에 따라 순위가 다릅니다.
- seed2028은 20장 중 7장에서 기존 D4보다 개선됐습니다. A71을 제외한 동일 19장에서는 평균이 낮아져, 전반적인 개선으로 단정하기 어렵습니다.

## 2. 모델과 실험 조건

공통 모델은 `cnn_unet`(`PhaseCNN`)으로, 잔차 블록을 사용하는 자체 구현 U-Net입니다. 입력은 gray/image_zscore/image_cdf 3채널, 출력은 primary_si/Al3Ni/eutectic_si/Al 4개 클래스입니다. 학습 70장, 검증 20장, effective batch size 8을 사용했습니다. 네 실험은 기존 가중치를 이어 학습한 것이 아니라 각각 새로 초기화해 학습했습니다.

| 실험 | 변경 사항 | 학습 일정(512 패치 / 768 패치 / 전체 영상) |
|---|---|---|
| baseline | 보존한 기존 CNN, seed 2026 | 2,500 / 1,500 / 500 |
| fullframe | 전체 영상 학습 비중 증가 | 2,000 / 1,000 / 1,500 |
| boundary | 경계 보조 손실 가중치 0.05 추가 | 2,500 / 1,500 / 500 |
| seed2027 | 기존 설정에서 난수 시드만 2027 | 2,500 / 1,500 / 500 |
| seed2028 | 기존 설정에서 난수 시드만 2028 | 2,500 / 1,500 / 500 |
| seed_ensemble | baseline + seed2027 + seed2028의 동일 가중치 확률 평균, D4 적용 | 추가 학습 없음 |

`seed2028`은 모델 구조 이름이 아니라 난수 시드 2028로 학습한 실험 이름입니다. DINOv3 설정은 준비되어 있지만 기본 suite 및 아래 완료 결과에 포함되지 않습니다.

## 3. 평가 기준과 전체 성능

주 지표는 검증 20장의 **이미지별 mIoU 평균**이며 픽셀 accuracy가 아닙니다. 정답과 예측 모두 없는 클래스는 해당 이미지 평균에서 제외(`empty_union=exclude`)합니다. Global mIoU는 모든 이미지의 혼동 행렬을 합산한 후 계산한 별도 보조 지표입니다. D4는 회전·반전 8개 변환의 추론 결과를 결합합니다. 아래 변화량 및 개선 장수는 모두 **baseline + D4** 기준입니다. 기존 comparison.csv의 변화량은 baseline plain 기준이므로 수치가 다릅니다.

| 실험 | Plain mIoU (%) | D4 mIoU (%) | D4 Global mIoU (%) | 기존 D4 대비 (%p) | 개선 장수 |
|---|---:|---:|---:|---:|---:|
| baseline | 78.4729 | 78.6865 | 81.4989 | +0.0000 | 0/20 |
| fullframe | 78.5860 | 78.7710 | 81.2688 | +0.0845 | 6/20 |
| boundary | 78.3139 | 78.4957 | 81.5788 | -0.1908 | 11/20 |
| seed2027 | 77.7883 | 78.0455 | 81.2474 | -0.6409 | 6/20 |
| seed2028 | 78.5990 | 78.7858 | 81.5082 | +0.0993 | 7/20 |
| seed_ensemble | — | 78.5617 | 81.6370 | -0.1248 | 8/20 |

## 4. 클래스별 IoU — D4

아래는 클래스별 이미지 평균 IoU입니다. 빈 클래스 제외로 클래스마다 평균에 포함되는 이미지 수가 다를 수 있습니다.

| 실험 | 초정 Si (%) | Al3Ni (%) | 공정 Si (%) | Al (%) |
|---|---:|---:|---:|---:|
| baseline | 86.3947 | 74.4029 | 59.8481 | 94.1002 |
| fullframe | 90.7758 | 73.1799 | 59.0625 | 94.0124 |
| boundary | 86.6266 | 73.5841 | 59.7558 | 94.0162 |
| seed2027 | 86.2982 | 72.8138 | 59.1427 | 93.9275 |
| seed2028 | 91.2817 | 72.6949 | 59.1609 | 93.9065 |
| seed_ensemble | 86.6333 | 73.6410 | 59.9182 | 94.0541 |

## 5. 해석 시 주의할 점

**A71 한 장의 영향이 큽니다.** `A71_step2_sample1`에는 초정 Si 정답이 없습니다. 기존 D4는 초정 Si를 780픽셀 오검출하여 해당 클래스 IoU 0을 평균에 포함하지만, seed2028과 fullframe은 이 클래스를 예측하지 않아 평균 대상에서 제외합니다. 정해진 평가 규칙에 따른 결과이며 계산 오류는 아닙니다.

- A71 이미지 mIoU: 기존 D4 **40.7059% → seed2028 D4 53.2620%**. 이 한 장의 전체 평균 기여는 **+0.6278%p**입니다.
- 나머지 19장: 기존 D4 **80.6855% → seed2028 D4 80.1291%**, **−0.5563%p**입니다. 이는 민감도 분석이며 공식 검증셋을 변경한 결과가 아닙니다.
- 초정 Si의 클래스 평균 86.3947% → 91.2817%는 각각 20장/19장 평균입니다. 동일 19장에서는 **90.9418% → 91.2817%**, **+0.3399%p**입니다.
- seed2028의 Al3Ni와 공정 Si 평균 IoU는 기존 D4보다 낮습니다. 주요 악화 이미지는 A67_step1_sample1(−7.3242%p), A11_step1_sample4(−3.0857%p)입니다.

**D4는 모든 단일 모델에서 개선됐습니다.** Plain 대비 +0.1818~+0.2573%p입니다. 반면 boundary와 seed 앙상블은 기존 D4 대비 주 지표가 낮습니다. 앙상블의 Global mIoU 개선과 이미지 평균 mIoU 하락은 함께 기록해야 합니다.

기존 분석의 이미지 A-prefix 그룹 단위 paired bootstrap 95% 구간은 seed2028의 기존 D4 대비 차이에 대해 **[−1.2203, +1.8128]%p**입니다. 검증셋을 통한 모델 선택과 다중 비교를 보정하지 않은 탐색적 구간입니다. 현재 작은 차이만으로 일반화 성능 향상을 확정할 수 없습니다. 테스트 10장은 정답이 없어 정확도를 산출하지 않았습니다.

## 6. 학습 시간과 선택된 체크포인트

아래 시간은 상태 로그에 기록된 학습 시간입니다. 평가와 프로세스 시작 비용을 포함한 전체 실행 시간과는 다릅니다. 새 실험 4개의 학습 시간 합계는 **38.87분**입니다.

| 실험 | 학습 시간(분) | 최고 검증 update | 선택 가중치 | 선택 당시 Plain mIoU (%) |
|---|---:|---:|---|---:|
| baseline | 9.11 | 4250 | raw | 78.4729 |
| fullframe | 11.25 | 4250 | raw | 78.5860 |
| boundary | 9.29 | 4250 | raw | 78.3139 |
| seed2027 | 9.21 | 4500 | ema | 77.7883 |
| seed2028 | 9.11 | 4250 | raw | 78.5990 |

## 7. 모델 위치와 후속 실험의 관계

- 최고 seed2028 모델: [best.pt](../experiments/accuracy_v1/outputs/runs/seed2028/official/best.pt)
- seed2028 학습 설정: [seed2028.yaml](../experiments/accuracy_v1/configs/seed2028.yaml)
- 나머지 학습 모델: `../experiments/accuracy_v1/outputs/runs/{fullframe,boundary,seed2027}/official/best.pt`
- 기존 기준 모델: [baseline/run/best.pt](../experiments/accuracy_v1/baseline/run/best.pt)

`hundred_v1`의 11번 `m011_seed2028_parent`는 위 seed2028 best.pt에서 출발해 전체 영상 600 updates를 추가 학습한 별도 후속 실험입니다. 이 보고서의 **78.7858%는 추가 학습 전 accuracy_v1 모델 점수**이며, hundred_v1의 78.8004%와 구분합니다.

## 8. 근거 파일

- [실험 계획](../experiments/accuracy_v1/configs/plan.yaml), [suite 완료 상태](../experiments/accuracy_v1/outputs/suite_status.json)
- [원본 비교 CSV](../experiments/accuracy_v1/outputs/reports/comparison.csv)
- [지표·Global mIoU·민감도 분석 CSV](../experiments/accuracy_v1/outputs/analysis_20260920/metric_analysis.csv)
- [학습 시간·체크포인트 분석 CSV](../experiments/accuracy_v1/outputs/analysis_20260920/training_analysis.csv)
- [기존 상세 분석 보고서](../experiments/accuracy_v1/outputs/analysis_20260920/REPORT.md)

전체 11개 평가 행의 이미지 평균 mIoU와 Global mIoU를 원본 per_image_metrics.csv의 점수 및 혼동 행렬에서 다시 계산하여 기존 분석 CSV와 일치하는지 확인했습니다.
