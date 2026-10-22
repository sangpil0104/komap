# R01_no_tta — 추론 D4만 제거

**image mIoU: 78.729805%** (valid 20 images, empty union=0).

이미지별 4개 클래스 IoU 평균을 구한 뒤 이미지별 동일 가중치로 평균합니다. Global mIoU는 계산하지 않습니다.

- 비교 대조군: `R00_seed32`
- 변경 요소: `{'tta': False}`
- 초기화: `/Data3/sppark/komap/experiments/seonghyeon_seed10_99_v1/outputs/20260926T082111_741695Z/seed_32/last.pt`
- 학습 방식: 보존된 seed 32의 추론 전용 참고 결과; 학습 없음
- 체크포인트 선택: 고정 마지막 update; valid 성능에 따른 checkpoint 선택 없음
- crop=1536, updates=16000, global batch=8, D4=False
- provenance: `provenance.json` / weights SHA-256: `1a97954c9655cb1e9d3a3b7c95a79dcda07807dd34fc979c57960b1b6cce1615`

각 비교 사진은 좌상 원본, 우상 오류(정답 색), 좌하 예측, 우하 정답입니다. 맞은 픽셀은 흰색입니다.
`100% − image mIoU`는 픽셀 오류율이 아닙니다. 경계 띠 오류율은 원본 픽셀 단위 진단값입니다.

| 이미지 | image mIoU | 100%−image mIoU | 비교 |
|---|---:|---:|---|
| A117_step4_sample2 | 84.8449% | 15.1551% | [보기](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 75.2444% | 24.7556% | [보기](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 76.3209% | 23.6791% | [보기](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 78.4940% | 21.5060% | [보기](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 73.0711% | 26.9289% | [보기](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 87.6404% | 12.3596% | [보기](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 77.9673% | 22.0327% | [보기](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 80.1693% | 19.8307% | [보기](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 81.5861% | 18.4139% | [보기](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 82.1265% | 17.8735% | [보기](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 81.7785% | 18.2215% | [보기](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 89.2324% | 10.7676% | [보기](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 81.5271% | 18.4729% | [보기](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 82.1571% | 17.8429% | [보기](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 82.4037% | 17.5963% | [보기](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 74.8327% | 25.1673% | [보기](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 46.2576% | 53.7424% | [보기](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 86.3823% | 13.6177% | [보기](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 71.6953% | 28.3047% | [보기](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 80.8645% | 19.1355% | [보기](valid_mask_comparison/A7_step1_sample4.png) |
