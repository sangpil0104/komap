# F_full_1.75 — HRNet_taeho_no_gaussian inference

image mIoU: **77.715549%** (valid20, empty union=0)

Control: `F_full2`
Change: `{'scale': 1.75}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "F_full_1.75",
  "contrast": "F_full2",
  "change": {
    "scale": 1.75
  },
  "config": {
    "scale": 1.75,
    "tta": "flip4",
    "precision": "fp32"
  }
}
```

Panels: original top-left, errors in ground-truth colors top-right (correct pixels white), prediction bottom-left, ground truth bottom-right.
The header reports image mIoU and 100%−image mIoU; the latter is not pixel error rate.

| Image | image mIoU | Comparison |
|---|---:|---|
| A117_step4_sample2 | 86.8437% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 68.1576% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 73.7401% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 78.8158% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 71.8329% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 86.5336% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 76.9004% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 77.8983% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 81.8141% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 81.4592% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 82.2992% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 89.3048% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 80.8854% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 80.2689% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 81.9568% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 74.2393% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 40.4576% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 85.9870% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 73.7555% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 81.1607% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
