# T01_native_geometry — 원본·확대 좌표가 일치하는 crop 전처리

**image mIoU: 77.538279%** (valid 20 images, empty union=0).

이미지별 4개 클래스 IoU 평균을 구한 뒤 이미지별 동일 가중치로 평균합니다. Global mIoU는 계산하지 않습니다.

- 비교 대조군: `T00_control`
- 변경 요소: `{'geometry': 'aligned_native_roi_v1'}`
- 초기화: `/Data3/sppark/komap/experiments/seonghyeon_seed10_v1/assets/efficientnet_b4_imagenet_encoder.pt`
- 학습 방식: ImageNet 인코더 + seed 32로 새로 초기화한 디코더; 다른 분할 체크포인트를 이어 학습하지 않음
- 체크포인트 선택: 고정 마지막 update; valid 성능에 따른 checkpoint 선택 없음
- crop=1536, updates=16000, global batch=8, D4=True
- provenance: `provenance.json` / weights SHA-256: `607f39beac6d9ed711c22bdc1e536fe00420953874e16ba658037757c1dfb513`

각 비교 사진은 좌상 원본, 우상 오류(정답 색), 좌하 예측, 우하 정답입니다. 맞은 픽셀은 흰색입니다.
`100% − image mIoU`는 픽셀 오류율이 아닙니다. 경계 띠 오류율은 원본 픽셀 단위 진단값입니다.

| 이미지 | image mIoU | 100%−image mIoU | 비교 |
|---|---:|---:|---|
| A117_step4_sample2 | 85.6509% | 14.3491% | [보기](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 73.8442% | 26.1558% | [보기](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 75.7769% | 24.2231% | [보기](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 77.0562% | 22.9438% | [보기](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 68.6723% | 31.3277% | [보기](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 87.9856% | 12.0144% | [보기](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 75.4066% | 24.5934% | [보기](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 73.7478% | 26.2522% | [보기](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 80.8409% | 19.1591% | [보기](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 81.5139% | 18.4861% | [보기](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 83.2815% | 16.7185% | [보기](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 88.6115% | 11.3885% | [보기](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 80.6404% | 19.3596% | [보기](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 81.8221% | 18.1779% | [보기](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 81.0404% | 18.9596% | [보기](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 73.0672% | 26.9328% | [보기](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 39.5124% | 60.4876% | [보기](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 85.8730% | 14.1270% | [보기](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 76.2005% | 23.7995% | [보기](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 80.2210% | 19.7790% | [보기](valid_mask_comparison/A7_step1_sample4.png) |
