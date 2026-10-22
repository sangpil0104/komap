# Model397 — local valid20 evaluation

image mIoU: **79.430160%**

원본 OOF 전용 번들에 추가한 로컬 검증 평가입니다. 각 family의 4-fold 확률 평균을 0.50 B4 + 0.25 HR seed1 + 0.25 HR seed2로 결합합니다. 원본 기록의 공식 valid 결과가 아닙니다. 빈 클래스 IoU=0.

Run: `/Data3/sppark/komap/experiments/lhs_349_397_gpu4567_v1/model397/outputs/run_20261006T050601_265645Z`

| Image | image mIoU | Comparison |
|---|---:|---|
| A117_step4_sample2 | 86.8988% | [image](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 71.6898% | [image](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 78.5294% | [image](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 79.5569% | [image](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 72.3397% | [image](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 88.8782% | [image](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 80.4071% | [image](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 80.6228% | [image](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 82.1414% | [image](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 83.0953% | [image](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 83.0626% | [image](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 89.7125% | [image](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 81.9248% | [image](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 83.3147% | [image](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 84.4093% | [image](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 75.8884% | [image](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 39.9409% | [image](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 87.3601% | [image](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 76.6329% | [image](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 82.1976% | [image](valid_mask_comparison/A7_step1_sample4.png) |