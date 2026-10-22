# T02_native_loss — 원본 해상도 NLL+Dice 손실 추가

**image mIoU: 77.672078%** (valid 20 images, empty union=0).

이미지별 4개 클래스 IoU 평균을 구한 뒤 이미지별 동일 가중치로 평균합니다. Global mIoU는 계산하지 않습니다.

- 비교 대조군: `T01_native_geometry`
- 변경 요소: `{'native_weight': 0.5}`
- 초기화: `/Data3/sppark/komap/experiments/seonghyeon_seed10_v1/assets/efficientnet_b4_imagenet_encoder.pt`
- 학습 방식: ImageNet 인코더 + seed 32로 새로 초기화한 디코더; 다른 분할 체크포인트를 이어 학습하지 않음
- 체크포인트 선택: 고정 마지막 update; valid 성능에 따른 checkpoint 선택 없음
- crop=1536, updates=16000, global batch=8, D4=True
- provenance: `provenance.json` / weights SHA-256: `b2b6594242c17a596bd1c5f62767f5868de668ada9ca617031d1aec99db59379`

각 비교 사진은 좌상 원본, 우상 오류(정답 색), 좌하 예측, 우하 정답입니다. 맞은 픽셀은 흰색입니다.
`100% − image mIoU`는 픽셀 오류율이 아닙니다. 경계 띠 오류율은 원본 픽셀 단위 진단값입니다.

| 이미지 | image mIoU | 100%−image mIoU | 비교 |
|---|---:|---:|---|
| A117_step4_sample2 | 85.9407% | 14.0593% | [보기](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 75.9026% | 24.0974% | [보기](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 75.4658% | 24.5342% | [보기](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 77.7951% | 22.2049% | [보기](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 69.2324% | 30.7676% | [보기](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 88.1885% | 11.8115% | [보기](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 76.6884% | 23.3116% | [보기](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 72.4082% | 27.5918% | [보기](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 80.6600% | 19.3400% | [보기](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 81.3170% | 18.6830% | [보기](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 81.8172% | 18.1828% | [보기](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 88.7746% | 11.2254% | [보기](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 80.8142% | 19.1858% | [보기](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 81.7547% | 18.2453% | [보기](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 81.1086% | 18.8914% | [보기](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 74.5579% | 25.4421% | [보기](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 39.5308% | 60.4692% | [보기](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 85.3431% | 14.6569% | [보기](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 76.2200% | 23.7800% | [보기](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 79.9218% | 20.0782% | [보기](valid_mask_comparison/A7_step1_sample4.png) |
