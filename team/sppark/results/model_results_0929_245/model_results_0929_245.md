# FR_full225 — HRNet_taeho_no_gaussian inference

image mIoU: **78.388360%** (valid20, empty union=0)

Control: `FR_full2`
Change: `{'scale': 2.25}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "FR_full225",
  "contrast": "FR_full2",
  "change": {
    "scale": 2.25
  },
  "config": {
    "scale": 2.25,
    "tta": "flip4",
    "precision": "fp32"
  },
  "allocator": "expandable_segments:True",
  "memory_fraction": 0.36
}
```

Panels: original top-left, errors in ground-truth colors top-right (correct pixels white), prediction bottom-left, ground truth bottom-right.
The header reports image mIoU and 100%−image mIoU; the latter is not pixel error rate.

| Image | image mIoU | Comparison |
|---|---:|---|
| A117_step4_sample2 | 87.1656% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 69.7817% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 77.0161% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 79.3733% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 72.1533% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 86.5948% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 76.3459% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 81.8751% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 79.9253% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 82.3704% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 79.7795% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 88.8115% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 82.2133% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 82.0394% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 82.6331% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 74.8534% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 42.4473% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 85.5829% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 75.9453% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 80.8600% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
