# EF2_0.25 — HRNet_taeho_no_gaussian inference

image mIoU: **79.096495%** (valid20, empty union=0)

Control: `C640_d4_bicubic`
Change: `{'same_checkpoint_context_ensemble': True, 'second_weight': 0.25}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "EF2_0.25",
  "contrast": "C640_d4_bicubic",
  "change": {
    "same_checkpoint_context_ensemble": true,
    "second_weight": 0.25
  },
  "members": [
    "C640_d4_bicubic",
    "C1024_d4_bicubic"
  ],
  "weights": [
    0.75,
    0.25
  ]
}
```

Panels: original top-left, errors in ground-truth colors top-right (correct pixels white), prediction bottom-left, ground truth bottom-right.
The header reports image mIoU and 100%−image mIoU; the latter is not pixel error rate.

| Image | image mIoU | Comparison |
|---|---:|---|
| A117_step4_sample2 | 87.8681% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 70.2938% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 77.1435% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 79.9952% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 72.3974% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 86.6552% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 77.8466% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 81.0981% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 82.2916% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 82.6463% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 82.9673% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 89.9317% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 82.1878% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 82.3942% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 83.6028% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 75.7259% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 42.2650% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 87.0724% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 75.9049% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 81.6419% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
