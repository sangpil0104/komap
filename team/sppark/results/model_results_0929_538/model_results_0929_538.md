# T06_no_gaussian — Gaussian blur만 제거; noise 유지

**image mIoU: 77.955470%** (valid 20 images, empty union=0).

이미지별 4개 클래스 IoU 평균을 구한 뒤 이미지별 동일 가중치로 평균합니다. Global mIoU는 계산하지 않습니다.

- 비교 대조군: `T00_control`
- 변경 요소: `{'gaussian_blur': False}`
- 초기화: `/Data3/sppark/komap/experiments/seonghyeon_seed10_v1/assets/efficientnet_b4_imagenet_encoder.pt`
- 학습 방식: ImageNet 인코더 + seed 32로 새로 초기화한 디코더; 다른 분할 체크포인트를 이어 학습하지 않음
- 체크포인트 선택: 고정 마지막 update; valid 성능에 따른 checkpoint 선택 없음
- crop=1536, updates=16000, global batch=8, D4=True
- provenance: `provenance.json` / weights SHA-256: `6344aeeeb04f9b8e7770ef4086f36db31de93c8db68883960dc6df5682a893e9`

각 비교 사진은 좌상 원본, 우상 오류(정답 색), 좌하 예측, 우하 정답입니다. 맞은 픽셀은 흰색입니다.
`100% − image mIoU`는 픽셀 오류율이 아닙니다. 경계 띠 오류율은 원본 픽셀 단위 진단값입니다.

| 이미지 | image mIoU | 100%−image mIoU | 비교 |
|---|---:|---:|---|
| A117_step4_sample2 | 85.7968% | 14.2032% | [보기](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 74.7063% | 25.2937% | [보기](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 75.5916% | 24.4084% | [보기](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 77.7615% | 22.2385% | [보기](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 70.4550% | 29.5450% | [보기](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 87.8563% | 12.1437% | [보기](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 78.4923% | 21.5077% | [보기](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 77.8484% | 22.1516% | [보기](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 81.0777% | 18.9223% | [보기](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 82.1907% | 17.8093% | [보기](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 81.9378% | 18.0622% | [보기](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 89.0462% | 10.9538% | [보기](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 81.0792% | 18.9208% | [보기](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 81.4897% | 18.5103% | [보기](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 81.7757% | 18.2243% | [보기](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 73.8954% | 26.1046% | [보기](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 39.7400% | 60.2600% | [보기](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 82.5610% | 17.4390% | [보기](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 76.0149% | 23.9851% | [보기](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 79.7927% | 20.2073% | [보기](valid_mask_comparison/A7_step1_sample4.png) |
