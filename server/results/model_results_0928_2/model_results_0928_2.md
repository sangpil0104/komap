# HRNet_taeho_no_gaussian — 0928 / multi-GPU

- 검증 이미지: 20장
- image mIoU: **80.167288%**
- 이미지마다4개 클래스 IoU 평균을 구하고 이미지별 동일 가중 평균. union=0이면1.
- 체크포인트: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`
- 원본 구현의 global mIoU 수치와 구분합니다.

| 이미지 | image mIoU | 비교 그림 |
|---|---:|---|
| A117_step4_sample2 | 87.7842% | [보기](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 69.4713% | [보기](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 75.2186% | [보기](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 80.0031% | [보기](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 72.5256% | [보기](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 86.8358% | [보기](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 78.5399% | [보기](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 80.6506% | [보기](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 82.9790% | [보기](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 82.6775% | [보기](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 82.2717% | [보기](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 89.9300% | [보기](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 82.0405% | [보기](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 82.0950% | [보기](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 83.5712% | [보기](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 75.5502% | [보기](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 66.7722% | [보기](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 86.7229% | [보기](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 75.8658% | [보기](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 81.8405% | [보기](valid_mask_comparison/A7_step1_sample4.png) |
