# 자기지도학습 비교 실험 C

- 학습 백본: **ResNet18 (ImageNet IMAGENET1K_V1)**.
- 적용 내용: A의 마지막 모델·optimizer를 이어 받아 추가 지도학습; B의 사전학습+분할학습 계산 시간에 맞춤.
- 단일 U-Net, 원해상도 skip, encoder BN 통계 고정, decoder GroupNorm. 입력 확대·TTA·경계 손실 없음.
- 원본 중심 512px + halo64px 패치(아래 config가 최종 기준), D4 학습 증강. 마스크는 원본 크기로 평가.
- train 70장만 학습에 사용. 자기지도 단계는 원본만 읽고, 가려진 실제 중심 픽셀의 L1 복원 손실로 학습.
- B 분할 단계에서는 encoder만 이전하고 decoder·분류 head는 A와 같은 초기값에서 시작.
- valid20은 평가에만 사용. 마지막 checkpoint를 고정하며 검증 점수로 선택하지 않음.
- 체크포인트: `/Data3/sppark/komap/experiments/selfsup_unet_v1/outputs/suite_20260924T113719_442712Z/C_extra/checkpoints/update_001037.pt`. 실행 설정·소스는 provenance.json과 source/에 저장.

## 점수 (%)

이미지마다 4개 상 IoU를 평균하고 20개 이미지에 대해 다시 평균합니다. 상별 값도 이미지 평균입니다. global mIoU는 사용하지 않습니다.

| image mIoU | Primary Si | Al3Ni | Eutectic Si | Al |
|---:|---:|---:|---:|---:|
| 76.3541 | 84.7592 | 70.9850 | 56.1490 | 93.5231 |

정답과 예측이 모두 없는 상은 IoU=1, 오검출이 있으면 IoU=0입니다.
빈 클래스 0 처리 mIoU: 76.3541%. 제외 처리 mIoU: 76.3541%.

## 경계와 외곽 진단

| 보조 지표 | 값 (%) |
|---|---:|
| boundary_error_1px | 16.2060 |
| boundary_iou_1px | 42.5708 |
| boundary_error_2px | 13.1996 |
| boundary_iou_2px | 56.6360 |
| boundary_error_4px | 10.3839 |
| boundary_iou_4px | 67.3506 |
| outer_error_16px | 8.8320 |
| outer_miou_16px | 74.8678 |
| inner_error_16px | 7.8124 |
| inner_miou_16px | 77.7250 |
| outer_error_32px | 8.5334 |
| outer_miou_32px | 74.3041 |
| inner_error_32px | 7.7726 |
| inner_miou_32px | 77.7510 |
| outer_error_64px | 8.3055 |
| outer_miou_64px | 74.9880 |
| inner_error_64px | 7.7031 |
| inner_miou_64px | 77.6868 |

boundary_error는 정답 상 경계에서 지정한 거리 이내의 오분류율입니다. outer_error는 이미지 외곽 띠의 오분류율입니다. 오류율은 낮을수록 좋습니다. boundary IoU는 클래스 내부 경계 띠(영상 프레임 포함)의 겹침이며, 양쪽에 없는 상은 이 보조 지표에서 제외합니다.

## 이미지별 결과

| 이미지 | image mIoU (%) | 비교 사진 |
|---|---:|---|
| A117_step4_sample2 | 87.0628 | [비교](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 68.9582 | [비교](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 72.3540 | [비교](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 76.5344 | [비교](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 70.1439 | [비교](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 84.3320 | [비교](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 76.8661 | [비교](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 78.2860 | [비교](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 80.0943 | [비교](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 81.4090 | [비교](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 81.1318 | [비교](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 86.6718 | [비교](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 79.8665 | [비교](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 80.0638 | [비교](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 80.6143 | [비교](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 72.5047 | [비교](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 39.6570 | [비교](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 79.5354 | [비교](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 70.7403 | [비교](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 80.2549 | [비교](valid_mask_comparison/A7_step1_sample4.png) |

비교 사진: 좌상단 원본, 우상단 오류(정답 픽셀은 흰색·오답 픽셀은 정답 상의 색), 좌하단 예측, 우하단 정답. 저장된 네 패널을 픽셀 단위로 재검사했습니다.

한 seed의 비교이므로 성능 차이의 안정성을 확정할 수 없습니다. 가림 복원 자체가 상 경계 개선을 보장하지 않습니다.
