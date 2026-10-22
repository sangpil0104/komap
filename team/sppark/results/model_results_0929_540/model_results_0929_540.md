# T08_refiner_no_gray — 보정 모듈의 원본 영상 입력만 0으로 차단

**image mIoU: 77.846010%** (valid 20 images, empty union=0).

이미지별 4개 클래스 IoU 평균을 구한 뒤 이미지별 동일 가중치로 평균합니다. Global mIoU는 계산하지 않습니다.

- 비교 대조군: `T07_native_refiner`
- 변경 요소: `{'refiner_use_gray': False}`
- 초기화: `/Data3/sppark/komap/experiments/seonghyeon_seed10_v1/assets/efficientnet_b4_imagenet_encoder.pt`
- 학습 방식: ImageNet 인코더 + seed 32로 새로 초기화한 디코더; 다른 분할 체크포인트를 이어 학습하지 않음
- 체크포인트 선택: 고정 마지막 update; valid 성능에 따른 checkpoint 선택 없음
- crop=1536, updates=16000, global batch=8, D4=True
- provenance: `provenance.json` / weights SHA-256: `74b2b6d675aee28f7c879e4f13e9e5a2b81fb9ba6329b41afce8af33f0b232d0`

각 비교 사진은 좌상 원본, 우상 오류(정답 색), 좌하 예측, 우하 정답입니다. 맞은 픽셀은 흰색입니다.
`100% − image mIoU`는 픽셀 오류율이 아닙니다. 경계 띠 오류율은 원본 픽셀 단위 진단값입니다.

| 이미지 | image mIoU | 100%−image mIoU | 비교 |
|---|---:|---:|---|
| A117_step4_sample2 | 86.0572% | 13.9428% | [보기](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 76.1714% | 23.8286% | [보기](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 75.6205% | 24.3795% | [보기](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 77.3658% | 22.6342% | [보기](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 68.2956% | 31.7044% | [보기](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 87.9534% | 12.0466% | [보기](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 77.6608% | 22.3392% | [보기](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 74.3058% | 25.6942% | [보기](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 80.7102% | 19.2898% | [보기](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 81.4738% | 18.5262% | [보기](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 82.3906% | 17.6094% | [보기](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 89.1878% | 10.8122% | [보기](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 80.9754% | 19.0246% | [보기](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 81.4104% | 18.5896% | [보기](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 81.2739% | 18.7261% | [보기](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 74.5206% | 25.4794% | [보기](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 39.5688% | 60.4312% | [보기](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 85.4950% | 14.5050% | [보기](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 76.3180% | 23.6820% | [보기](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 80.1651% | 19.8349% | [보기](valid_mask_comparison/A7_step1_sample4.png) |
