# T00_control — B4 U-Net 신규 학습 대조군

**image mIoU: 78.066012%** (valid 20 images, empty union=0).

이미지별 4개 클래스 IoU 평균을 구한 뒤 이미지별 동일 가중치로 평균합니다. Global mIoU는 계산하지 않습니다.

- 비교 대조군: `reference / control`
- 변경 요소: `{}`
- 초기화: `/Data3/sppark/komap/experiments/seonghyeon_seed10_v1/assets/efficientnet_b4_imagenet_encoder.pt`
- 학습 방식: ImageNet 인코더 + seed 32로 새로 초기화한 디코더; 다른 분할 체크포인트를 이어 학습하지 않음
- 체크포인트 선택: 고정 마지막 update; valid 성능에 따른 checkpoint 선택 없음
- crop=1536, updates=16000, global batch=8, D4=True
- provenance: `provenance.json` / weights SHA-256: `d4e726dc607508c949ad32c3696b92c5715e52193d852101df214aa4eb142bda`

각 비교 사진은 좌상 원본, 우상 오류(정답 색), 좌하 예측, 우하 정답입니다. 맞은 픽셀은 흰색입니다.
`100% − image mIoU`는 픽셀 오류율이 아닙니다. 경계 띠 오류율은 원본 픽셀 단위 진단값입니다.

| 이미지 | image mIoU | 100%−image mIoU | 비교 |
|---|---:|---:|---|
| A117_step4_sample2 | 86.1188% | 13.8812% | [보기](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 74.0796% | 25.9204% | [보기](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 75.9554% | 24.0446% | [보기](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 77.6626% | 22.3374% | [보기](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 70.7606% | 29.2394% | [보기](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 87.7772% | 12.2228% | [보기](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 78.3095% | 21.6905% | [보기](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 78.0052% | 21.9948% | [보기](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 80.8225% | 19.1775% | [보기](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 81.6511% | 18.3489% | [보기](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 82.3869% | 17.6131% | [보기](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 88.5219% | 11.4781% | [보기](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 81.0945% | 18.9055% | [보기](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 81.6421% | 18.3579% | [보기](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 81.8112% | 18.1888% | [보기](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 74.0096% | 25.9904% | [보기](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 40.0663% | 59.9337% | [보기](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 84.7428% | 15.2572% | [보기](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 75.8734% | 24.1266% | [보기](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 80.0292% | 19.9708% | [보기](valid_mask_comparison/A7_step1_sample4.png) |
