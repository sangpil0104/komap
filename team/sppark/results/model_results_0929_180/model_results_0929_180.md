# F_full_1.5 — HRNet_taeho_no_gaussian inference

image mIoU: **76.511774%** (valid20, empty union=0)

Control: `F_full2`
Change: `{'scale': 1.5}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "F_full_1.5",
  "contrast": "F_full2",
  "change": {
    "scale": 1.5
  },
  "config": {
    "scale": 1.5,
    "tta": "flip4",
    "precision": "fp32"
  }
}
```

Panels: original top-left, errors in ground-truth colors top-right (correct pixels white), prediction bottom-left, ground truth bottom-right.
The header reports image mIoU and 100%−image mIoU; the latter is not pixel error rate.

| Image | image mIoU | Comparison |
|---|---:|---|
| A117_step4_sample2 | 86.2261% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 67.5359% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 69.7808% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 78.1091% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 71.8168% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 85.4151% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 76.8326% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 76.1584% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 80.7703% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 81.1858% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 82.2707% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 89.2183% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 80.0159% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 77.4967% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 78.5320% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 73.1172% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 40.2589% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 83.8130% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 71.0298% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 80.6522% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
