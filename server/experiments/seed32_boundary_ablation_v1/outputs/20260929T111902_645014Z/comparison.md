# Seed 32 구조 기반 단일 요소 제거·추가 실험

학습 실험은 모두 ImageNet 인코더 + 새 디코더로 시작합니다. seed 32 분할 가중치는 R 실험의 추론에만 사용합니다.
주 지표: valid20 image mIoU, 네 클래스 고정 평균, 빈 클래스 IoU=0.
T 실험은 유효 배치 8 / SyncBatchNorm의 공통 조건입니다. 원본 seed32(배치2) 점수는 참고값이며 직접적인 단일 요소 대조군이 아닙니다.

| 실험 | 변경 | 대조군 | image mIoU | 대조군 대비 %p | 결과 |
|---|---|---|---:|---:|---|
| R00_seed32 | Seed 32 원본 / D4 | — | 78.976546% | — | [보고서](/Data3/sppark/komap/results/model_results_0929_6/model_results_0929_6.md) |
| R01_no_tta | 추론 D4만 제거 | R00_seed32 | 78.729805% | -0.246741 | [보고서](/Data3/sppark/komap/results/model_results_0929_7/model_results_0929_7.md) |
| R02_bilinear_restore | 최종 원본 복원만 bilinear | R00_seed32 | 79.147809% | +0.171263 | [보고서](/Data3/sppark/komap/results/model_results_0929_8/model_results_0929_8.md) |
| T00_control | B4 U-Net 신규 학습 대조군 | — | 78.066012% | — | [보고서](/Data3/sppark/komap/results/model_results_0929_190/model_results_0929_190.md) |
| T01_native_geometry | 원본·확대 좌표가 일치하는 crop 전처리 | T00_control | 77.538279% | -0.527733 | [보고서](/Data3/sppark/komap/results/model_results_0929_533/model_results_0929_533.md) |
| T02_native_loss | 원본 해상도 NLL+Dice 손실 추가 | T01_native_geometry | 77.672078% | +0.133799 | [보고서](/Data3/sppark/komap/results/model_results_0929_534/model_results_0929_534.md) |
| T03_boundary_sampling | 35% 경계 중심 crop | T01_native_geometry | 77.510049% | -0.028231 | [보고서](/Data3/sppark/komap/results/model_results_0929_535/model_results_0929_535.md) |
| T04_boundary_weight | 경계 띠 CE 가중치 1→2 | T01_native_geometry | 77.626274% | +0.087995 | [보고서](/Data3/sppark/komap/results/model_results_0929_536/model_results_0929_536.md) |
| T05_detail_branch | 모델 입력 해상도 특징 경로 추가 | T00_control | 78.029059% | -0.036953 | [보고서](/Data3/sppark/komap/results/model_results_0929_537/model_results_0929_537.md) |
| T06_no_gaussian | Gaussian blur만 제거; noise 유지 | T00_control | 77.955470% | -0.110542 | [보고서](/Data3/sppark/komap/results/model_results_0929_538/model_results_0929_538.md) |
| T07_native_refiner | 동일 native loss에 원본 영상 보정 모듈 추가 | T02_native_loss | 77.875722% | +0.203644 | [보고서](/Data3/sppark/komap/results/model_results_0929_539/model_results_0929_539.md) |
| T08_refiner_no_gray | 보정 모듈의 원본 영상 입력만 0으로 차단 | T07_native_refiner | 77.846010% | -0.029712 | [보고서](/Data3/sppark/komap/results/model_results_0929_540/model_results_0929_540.md) |
| T09_bilinear_decoder | 디코더 보간만 nearest→bilinear | T00_control | 78.139163% | +0.073151 | [보고서](/Data3/sppark/komap/results/model_results_0929_541/model_results_0929_541.md) |
