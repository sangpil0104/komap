# R02_bilinear_restore — 최종 원본 복원만 bilinear

**image mIoU: 79.147809%** (valid 20 images, empty union=0).

이미지별 4개 클래스 IoU 평균을 구한 뒤 이미지별 동일 가중치로 평균합니다. Global mIoU는 계산하지 않습니다.

- 비교 대조군: `R00_seed32`
- 변경 요소: `{'restore': 'bilinear'}`
- 초기화: `/Data3/sppark/komap/experiments/seonghyeon_seed10_99_v1/outputs/20260926T082111_741695Z/seed_32/last.pt`
- 학습 방식: 보존된 seed 32의 추론 전용 참고 결과; 학습 없음
- 체크포인트 선택: 고정 마지막 update; valid 성능에 따른 checkpoint 선택 없음
- crop=1536, updates=16000, global batch=8, D4=True
- provenance: `provenance.json` / weights SHA-256: `1a97954c9655cb1e9d3a3b7c95a79dcda07807dd34fc979c57960b1b6cce1615`

각 비교 사진은 좌상 원본, 우상 오류(정답 색), 좌하 예측, 우하 정답입니다. 맞은 픽셀은 흰색입니다.
`100% − image mIoU`는 픽셀 오류율이 아닙니다. 경계 띠 오류율은 원본 픽셀 단위 진단값입니다.

| 이미지 | image mIoU | 100%−image mIoU | 비교 |
|---|---:|---:|---|
| A117_step4_sample2 | 85.4293% | 14.5707% | [보기](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 75.4761% | 24.5239% | [보기](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 77.0386% | 22.9614% | [보기](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 78.8713% | 21.1287% | [보기](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 73.5274% | 26.4726% | [보기](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 88.0027% | 11.9973% | [보기](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 78.9104% | 21.0896% | [보기](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 79.8433% | 20.1567% | [보기](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 81.9262% | 18.0738% | [보기](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 82.7770% | 17.2230% | [보기](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 82.6932% | 17.3068% | [보기](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 89.6406% | 10.3594% | [보기](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 81.7389% | 18.2611% | [보기](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 82.5340% | 17.4660% | [보기](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 82.4798% | 17.5202% | [보기](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 75.2367% | 24.7633% | [보기](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 45.8641% | 54.1359% | [보기](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 86.9773% | 13.0227% | [보기](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 72.2665% | 27.7335% | [보기](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 81.7227% | 18.2773% | [보기](valid_mask_comparison/A7_step1_sample4.png) |
