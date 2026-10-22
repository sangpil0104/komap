# Seed 32 구조 기반 단일 요소 제거·추가 실험

학습 실험은 모두 ImageNet 인코더 + 새 디코더로 시작합니다. seed 32 분할 가중치는 R 실험의 추론에만 사용합니다.
주 지표: valid20 image mIoU, 네 클래스 고정 평균, 빈 클래스 IoU=0.
T 실험은 유효 배치 8 / SyncBatchNorm의 공통 조건입니다. 원본 seed32(배치2) 점수는 참고값이며 직접적인 단일 요소 대조군이 아닙니다.

| 실험 | 변경 | 대조군 | image mIoU | 대조군 대비 %p | 결과 |
|---|---|---|---:|---:|---|
| R00_seed32 | Seed 32 원본 / D4 | — | 78.976546% | — | [보고서](/Data3/sppark/komap/results/model_results_0929_3/model_results_0929_3.md) |
| R01_no_tta | 추론 D4만 제거 | R00_seed32 | 78.729805% | -0.246741 | [보고서](/Data3/sppark/komap/results/model_results_0929_4/model_results_0929_4.md) |
| R02_bilinear_restore | 최종 원본 복원만 bilinear | R00_seed32 | 79.147809% | +0.171263 | [보고서](/Data3/sppark/komap/results/model_results_0929_5/model_results_0929_5.md) |
