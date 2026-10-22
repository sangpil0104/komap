# T04_boundary_weight — 경계 띠 CE 가중치 1→2

**image mIoU: 77.626274%** (valid 20 images, empty union=0).

이미지별 4개 클래스 IoU 평균을 구한 뒤 이미지별 동일 가중치로 평균합니다. Global mIoU는 계산하지 않습니다.

- 비교 대조군: `T01_native_geometry`
- 변경 요소: `{'boundary_ce_extra': 1.0}`
- 초기화: `/Data3/sppark/komap/experiments/seonghyeon_seed10_v1/assets/efficientnet_b4_imagenet_encoder.pt`
- 학습 방식: ImageNet 인코더 + seed 32로 새로 초기화한 디코더; 다른 분할 체크포인트를 이어 학습하지 않음
- 체크포인트 선택: 고정 마지막 update; valid 성능에 따른 checkpoint 선택 없음
- crop=1536, updates=16000, global batch=8, D4=True
- provenance: `provenance.json` / weights SHA-256: `706b69dfe3b8a578cc4032dbad673fc500de049039f36d1f6900fb894ccc649d`

각 비교 사진은 좌상 원본, 우상 오류(정답 색), 좌하 예측, 우하 정답입니다. 맞은 픽셀은 흰색입니다.
`100% − image mIoU`는 픽셀 오류율이 아닙니다. 경계 띠 오류율은 원본 픽셀 단위 진단값입니다.

| 이미지 | image mIoU | 100%−image mIoU | 비교 |
|---|---:|---:|---|
| A117_step4_sample2 | 85.6810% | 14.3190% | [보기](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 74.0014% | 25.9986% | [보기](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 76.6472% | 23.3528% | [보기](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 77.3291% | 22.6709% | [보기](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 69.7249% | 30.2751% | [보기](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 87.8609% | 12.1391% | [보기](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 77.1421% | 22.8579% | [보기](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 74.1386% | 25.8614% | [보기](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 80.6630% | 19.3370% | [보기](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 80.9728% | 19.0272% | [보기](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 81.9513% | 18.0487% | [보기](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 88.8294% | 11.1706% | [보기](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 80.6396% | 19.3604% | [보기](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 81.6753% | 18.3247% | [보기](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 81.2693% | 18.7307% | [보기](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 73.1758% | 26.8242% | [보기](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 40.0645% | 59.9355% | [보기](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 84.1372% | 15.8628% | [보기](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 76.2977% | 23.7023% | [보기](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 80.3246% | 19.6754% | [보기](valid_mask_comparison/A7_step1_sample4.png) |
