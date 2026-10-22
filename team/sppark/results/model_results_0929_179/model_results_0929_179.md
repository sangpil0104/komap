# F_full_1.0 — HRNet_taeho_no_gaussian inference

image mIoU: **70.296785%** (valid20, empty union=0)

Control: `F_full2`
Change: `{'scale': 1.0}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "F_full_1.0",
  "contrast": "F_full2",
  "change": {
    "scale": 1.0
  },
  "config": {
    "scale": 1.0,
    "tta": "flip4",
    "precision": "fp32"
  }
}
```

Panels: original top-left, errors in ground-truth colors top-right (correct pixels white), prediction bottom-left, ground truth bottom-right.
The header reports image mIoU and 100%−image mIoU; the latter is not pixel error rate.

| Image | image mIoU | Comparison |
|---|---:|---|
| A117_step4_sample2 | 81.0262% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 65.8857% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 55.2265% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 72.6926% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 67.0347% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 73.4869% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 77.3538% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 70.3113% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 80.1895% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 79.1939% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 82.0015% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 86.8411% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 71.5217% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 67.5783% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 62.4455% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 60.8294% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 38.0693% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 76.0518% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 61.1938% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 77.0021% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
