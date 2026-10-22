# 빈 클래스 IoU=0 기준 재계산 결과 — 2026-09-29

현재 작업 폴더에서 **공식 valid 20장에 대한 이미지별 결과를 확인할 수 있는 평가 기록**을 비교한 최고 모델은 **EfficientNet-B4 U-Net seed 32 (D4 TTA)**이며, **image mIoU 78.9765456654%**다.

## 평가 기준

각 이미지에서 4개 클래스의 IoU를 계산한다. 정답과 예측에 모두 없는 클래스는 IoU=0이다. 4개 IoU를 더해 항상 4로 나누고, 이렇게 얻은 이미지별 mIoU를 valid 20장에 동일 가중치로 평균한다. Global mIoU는 순위에 사용하지 않았다.

사용자 예시 `(0.8 + 0.8 + 0.8 + 0) / 4 = 0.6`도 재계산 코드에서 검증했다.

## 최상위 결과

| 순위 | 모델 / 추론 | 빈 클래스=0 | 빈 클래스=1 (이전) |
|---:|---|---:|---:|
| 1 | EfficientNet-B4 U-Net seed 32 / D4 TTA | 78.976546% | 80.226546% |
| 2 | EfficientNet-B4 U-Net seed 63 / D4 TTA | 78.959910% | 80.209910% |
| 3 | EfficientNet-B4 U-Net seed 11 / D4 TTA | 78.941877% | 80.191877% |
| 4 | HRNet_taeho_no_gaussian / D4 TTA | 78.917288% | 80.167288% |
| 5 | EfficientNet-B4 U-Net seed 80 / D4 TTA | 78.828614% | 80.078614% |
| 6 | EfficientNet-B4 U-Net seed 37 / D4 TTA | 78.811772% | 80.061772% |
| 7 | EfficientNet-B4 U-Net seed 61 / D4 TTA | 78.747686% | 79.997686% |

위 7개는 전체 평가 기록에서도 상위 7개 결과다. 동일 모델의 복제된 보고서는 중복 순위로 세지 않았다.

그 외 비교: `photo_mild + D4`는 78.720305%, `HRNet_taeho`는 78.664330%다. seed 32와 HRNet(no Gaussian)의 차이는 **0.059258%p**다.

## 최고 모델과 계산 검산

- 체크포인트: [last.pt](/Data3/sppark/komap/experiments/seonghyeon_seed10_99_v1/outputs/20260926T082111_741695Z/seed_32/last.pt)
- 구조: U-Net + EfficientNet-B4, ImageNet 정규화, 흑백 입력을 3채널로 복사.
- 설정: seed 32, target_width=4000, crop=1536, batch=2, 16,000 iterations, 마지막 체크포인트, D4 8방향 TTA.
- 기존 결과: [model_results_0926_2](../model_results_0926_2/model_results_0926_2.md), [seed 32 원본 지표](../model_results_0926_2/metrics/seed_32.json).

빈 클래스가 발생하는 것은 `A71_step2_sample1`의 Primary Si 한 쌍이다. 이 이미지에서 Primary Si의 TP, FP, FN이 모두 0임을 저장된 예측/정답 마스크로 확인했다.

```text
A71의 새 image mIoU
= (0.9186927760984799 + 0.4440667305618785 + 0 + 0.4827805986955321) / 4
= 46.1385026339%

20장 평균 감소량 = 100% / 4 / 20 = 1.25%p
seed 32: 80.2265456654% - 1.25%p = 78.9765456654%
```

## 확인 범위와 한계

- 공식 검증 이미지 20개의 ID를 manifest와 대조한 평가 기록 **1,359건**을 재계산했다. 복제 보고서·추론 설정·중간 검증 기록이 포함되므로 모델 개수와 같지 않다.
- 포함: CNN baseline, accuracy, augmentation, refinement, hundred, evolve, evolve_accuracy, ResNet34 U-Net++ ASPP, single_unet, selfsup, adaptive threshold, seed 0–99, 0927 개선 모델 12개, 0928 HRNet 2개.
- confusion matrix에서 TP와 union을 직접 계산했다. seed 결과는 클래스 IoU와 absent_classes를 이용했다. 상위 seed 32·63·11의 60개 저장 예측 마스크를 정답과 다시 비교하여 점수가 일치함을 확인했다.
- 과거 Markdown만 있는 업로드 중 원본 체크포인트/예측 마스크/이미지별 지표가 이 폴더에 없는 항목(예: 0919·0922 현석 보고서의 외부 경로 모델, 0923 성현 보고서의 과거 해상도/앙상블 실험)은 정확한 새 점수를 계산할 수 없다. 특히 빈 클래스를 제외한 과거 점수에 일괄적으로 1.25%p를 빼면 안 된다.
- 평가 기록이 없는 중간 체크포인트를 새로 추론하지 않았다. 따라서 결과는 **재계산 가능한 저장 평가 결과 기준 최고**이며, 자료가 없는 모든 과거 모델까지 수치로 검증했다는 의미는 아니다.
- 학습·새 모델 추론 없이 저장된 지표와 예측 마스크만 사용했다. 기존 소스·모델·결과를 변경하지 않았으며, 읽은 지표 및 검산 마스크의 SHA-256이 처리 전후 일치한다.

## 최고 모델의 이미지별 결과

비교 그림은 원본 해상도로 좌상 원본, 우상 오류 위치, 좌하 예측, 우하 정답을 배치했다. 맞은 픽셀은 흰색, 틀린 픽셀은 정답 색이다. 상단에 새 image mIoU와 `100% - image mIoU`를 표시했다. 후자는 픽셀 오류율과 다른 값이다.

| 이미지 | image mIoU (빈 클래스=0) | 100% - image mIoU | 비교 그림 |
|---|---:|---:|---|
| A117_step4_sample2 | 85.217373% | 14.782627% | [보기](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 75.239134% | 24.760866% | [보기](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 76.918223% | 23.081777% | [보기](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 78.682318% | 21.317682% | [보기](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 73.567578% | 26.432422% | [보기](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 87.835140% | 12.164860% | [보기](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 78.764411% | 21.235589% | [보기](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 79.770162% | 20.229838% | [보기](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 81.803501% | 18.196499% | [보기](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 82.498933% | 17.501067% | [보기](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 82.615737% | 17.384263% | [보기](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 89.470180% | 10.529820% | [보기](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 81.547733% | 18.452267% | [보기](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 82.344210% | 17.655790% | [보기](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 82.424300% | 17.575700% | [보기](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 74.973898% | 25.026102% | [보기](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 46.138503% | 53.861497% | [보기](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 86.827044% | 13.172956% | [보기](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 71.395810% | 28.604190% | [보기](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 81.496725% | 18.503275% | [보기](valid_mask_comparison/A7_step1_sample4.png) |

## 전체 계산 자료

- [전체 평가 기록 CSV](all_evaluations.csv)
- [이미지별 재계산 기록 JSON](evaluation_records.json)
- [최고 모델 confusion matrix](winner_confusion_matrices.json)
- [검산 요약](summary.json)
- [재계산 코드](recalculate.py)
- [읽은 원본의 SHA-256](source_sha256.json)
