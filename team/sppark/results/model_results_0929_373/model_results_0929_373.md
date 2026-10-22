# W14_full_factor — HRNet_taeho_no_gaussian inference

image mIoU: **78.270811%** (valid20, empty union=0)

Control: `W00_full_extended_control`
Change: `{'scale': 1.875}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "W14_full_factor",
  "contrast": "W00_full_extended_control",
  "change": {
    "scale": 1.875
  },
  "engine": "full_v2",
  "config": {
    "scale": 1.875,
    "tta": "d4",
    "precision": "fp32"
  }
}
```

Panels: original top-left, errors in ground-truth colors top-right (correct pixels white), prediction bottom-left, ground truth bottom-right.
The header reports image mIoU and 100%−image mIoU; the latter is not pixel error rate.

| Image | image mIoU | Comparison |
|---|---:|---|
| A117_step4_sample2 | 87.1403% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 68.7386% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 75.9120% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 79.2814% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 71.9733% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 86.6038% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 76.7199% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 80.6018% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 81.6145% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 81.7985% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 82.5768% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 89.2316% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 81.5606% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 81.2506% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 82.3810% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 74.5677% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 40.7651% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 86.5008% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 74.8428% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 81.3551% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
