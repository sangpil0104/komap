# HRNet_taeho — 0928 / multi-GPU

- 검증 이미지: 20장
- image mIoU: **79.914330%**
- 이미지마다4개 클래스 IoU 평균을 구하고 이미지별 동일 가중 평균. union=0이면1.
- 체크포인트: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_0928_ddp6_20260928T122637_062971Z/last.pt`
- 원본 구현의 global mIoU 수치와 구분합니다.

| 이미지 | image mIoU | 비교 그림 |
|---|---:|---|
| A117_step4_sample2 | 87.5927% | [보기](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 72.0433% | [보기](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 76.0490% | [보기](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 80.0530% | [보기](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 71.7886% | [보기](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 85.6125% | [보기](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 77.8156% | [보기](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 79.3800% | [보기](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 82.4242% | [보기](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 82.3320% | [보기](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 82.6969% | [보기](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 89.8895% | [보기](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 82.3063% | [보기](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 82.0245% | [보기](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 83.1017% | [보기](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 74.9631% | [보기](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 65.4940% | [보기](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 86.8138% | [보기](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 75.7519% | [보기](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 80.1540% | [보기](valid_mask_comparison/A7_step1_sample4.png) |
