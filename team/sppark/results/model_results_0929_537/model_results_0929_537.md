# T05_detail_branch — 모델 입력 해상도 특징 경로 추가

**image mIoU: 78.029059%** (valid 20 images, empty union=0).

이미지별 4개 클래스 IoU 평균을 구한 뒤 이미지별 동일 가중치로 평균합니다. Global mIoU는 계산하지 않습니다.

- 비교 대조군: `T00_control`
- 변경 요소: `{'detail_branch': True}`
- 초기화: `/Data3/sppark/komap/experiments/seonghyeon_seed10_v1/assets/efficientnet_b4_imagenet_encoder.pt`
- 학습 방식: ImageNet 인코더 + seed 32로 새로 초기화한 디코더; 다른 분할 체크포인트를 이어 학습하지 않음
- 체크포인트 선택: 고정 마지막 update; valid 성능에 따른 checkpoint 선택 없음
- crop=1536, updates=16000, global batch=8, D4=True
- provenance: `provenance.json` / weights SHA-256: `3dc57fa2854a8bd7030a5689a4f8ae99d99fa465fb1341bf07e4d7729f7cfbc7`

각 비교 사진은 좌상 원본, 우상 오류(정답 색), 좌하 예측, 우하 정답입니다. 맞은 픽셀은 흰색입니다.
`100% − image mIoU`는 픽셀 오류율이 아닙니다. 경계 띠 오류율은 원본 픽셀 단위 진단값입니다.

| 이미지 | image mIoU | 100%−image mIoU | 비교 |
|---|---:|---:|---|
| A117_step4_sample2 | 86.5478% | 13.4522% | [보기](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 73.6071% | 26.3929% | [보기](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 75.8687% | 24.1313% | [보기](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 77.6988% | 22.3012% | [보기](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 69.8297% | 30.1703% | [보기](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 87.5367% | 12.4633% | [보기](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 78.8040% | 21.1960% | [보기](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 75.9714% | 24.0286% | [보기](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 80.9366% | 19.0634% | [보기](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 82.2837% | 17.7163% | [보기](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 83.7999% | 16.2001% | [보기](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 88.2696% | 11.7304% | [보기](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 81.0742% | 18.9258% | [보기](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 81.7783% | 18.2217% | [보기](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 81.4022% | 18.5978% | [보기](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 74.4214% | 25.5786% | [보기](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 39.8685% | 60.1315% | [보기](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 85.0878% | 14.9122% | [보기](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 76.0843% | 23.9157% | [보기](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 79.7103% | 20.2897% | [보기](valid_mask_comparison/A7_step1_sample4.png) |
