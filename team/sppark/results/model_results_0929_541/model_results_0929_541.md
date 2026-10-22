# T09_bilinear_decoder — 디코더 보간만 nearest→bilinear

**image mIoU: 78.139163%** (valid 20 images, empty union=0).

이미지별 4개 클래스 IoU 평균을 구한 뒤 이미지별 동일 가중치로 평균합니다. Global mIoU는 계산하지 않습니다.

- 비교 대조군: `T00_control`
- 변경 요소: `{'decoder_interpolation': 'bilinear'}`
- 초기화: `/Data3/sppark/komap/experiments/seonghyeon_seed10_v1/assets/efficientnet_b4_imagenet_encoder.pt`
- 학습 방식: ImageNet 인코더 + seed 32로 새로 초기화한 디코더; 다른 분할 체크포인트를 이어 학습하지 않음
- 체크포인트 선택: 고정 마지막 update; valid 성능에 따른 checkpoint 선택 없음
- crop=1536, updates=16000, global batch=8, D4=True
- provenance: `provenance.json` / weights SHA-256: `a95f92329cf4986b7d594c763828c4564e4073bc785170db660e3947dee73518`

각 비교 사진은 좌상 원본, 우상 오류(정답 색), 좌하 예측, 우하 정답입니다. 맞은 픽셀은 흰색입니다.
`100% − image mIoU`는 픽셀 오류율이 아닙니다. 경계 띠 오류율은 원본 픽셀 단위 진단값입니다.

| 이미지 | image mIoU | 100%−image mIoU | 비교 |
|---|---:|---:|---|
| A117_step4_sample2 | 86.0520% | 13.9480% | [보기](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 74.9854% | 25.0146% | [보기](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 76.1133% | 23.8867% | [보기](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 77.2266% | 22.7734% | [보기](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 69.4011% | 30.5989% | [보기](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 87.3076% | 12.6924% | [보기](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 77.9084% | 22.0916% | [보기](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 79.8474% | 20.1526% | [보기](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 80.9283% | 19.0717% | [보기](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 81.7577% | 18.2423% | [보기](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 83.3399% | 16.6601% | [보기](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 89.0397% | 10.9603% | [보기](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 81.3334% | 18.6666% | [보기](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 81.8530% | 18.1470% | [보기](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 81.6190% | 18.3810% | [보기](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 73.7285% | 26.2715% | [보기](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 40.4955% | 59.5045% | [보기](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 83.6719% | 16.3281% | [보기](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 76.1946% | 23.8054% | [보기](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 79.9800% | 20.0200% | [보기](valid_mask_comparison/A7_step1_sample4.png) |
