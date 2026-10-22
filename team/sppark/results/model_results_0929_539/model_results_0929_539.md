# T07_native_refiner — 동일 native loss에 원본 영상 보정 모듈 추가

**image mIoU: 77.875722%** (valid 20 images, empty union=0).

이미지별 4개 클래스 IoU 평균을 구한 뒤 이미지별 동일 가중치로 평균합니다. Global mIoU는 계산하지 않습니다.

- 비교 대조군: `T02_native_loss`
- 변경 요소: `{'native_refiner': True}`
- 초기화: `/Data3/sppark/komap/experiments/seonghyeon_seed10_v1/assets/efficientnet_b4_imagenet_encoder.pt`
- 학습 방식: ImageNet 인코더 + seed 32로 새로 초기화한 디코더; 다른 분할 체크포인트를 이어 학습하지 않음
- 체크포인트 선택: 고정 마지막 update; valid 성능에 따른 checkpoint 선택 없음
- crop=1536, updates=16000, global batch=8, D4=True
- provenance: `provenance.json` / weights SHA-256: `92f24bf523e490fd63edb8acb714af731cec60df33470a7cc9c4e98e5593d4ae`

각 비교 사진은 좌상 원본, 우상 오류(정답 색), 좌하 예측, 우하 정답입니다. 맞은 픽셀은 흰색입니다.
`100% − image mIoU`는 픽셀 오류율이 아닙니다. 경계 띠 오류율은 원본 픽셀 단위 진단값입니다.

| 이미지 | image mIoU | 100%−image mIoU | 비교 |
|---|---:|---:|---|
| A117_step4_sample2 | 85.8445% | 14.1555% | [보기](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 77.7653% | 22.2347% | [보기](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 76.1719% | 23.8281% | [보기](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 77.0662% | 22.9338% | [보기](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 68.5909% | 31.4091% | [보기](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 87.9626% | 12.0374% | [보기](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 76.1202% | 23.8798% | [보기](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 73.2522% | 26.7478% | [보기](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 80.8811% | 19.1189% | [보기](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 81.7141% | 18.2859% | [보기](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 82.3420% | 17.6580% | [보기](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 88.7488% | 11.2512% | [보기](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 80.8657% | 19.1343% | [보기](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 81.7240% | 18.2760% | [보기](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 80.8293% | 19.1707% | [보기](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 75.0138% | 24.9862% | [보기](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 39.8451% | 60.1549% | [보기](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 85.5016% | 14.4984% | [보기](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 76.3396% | 23.6604% | [보기](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 80.9354% | 19.0646% | [보기](valid_mask_comparison/A7_step1_sample4.png) |
