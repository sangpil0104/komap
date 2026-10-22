# EM_context_tta_0.25 — HRNet_taeho_no_gaussian inference

image mIoU: **79.042676%** (valid20, empty union=0)

Control: `E_tta_0.4`
Change: `{'context_weight': 0.25}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "EM_context_tta_0.25",
  "contrast": "E_tta_0.4",
  "change": {
    "context_weight": 0.25
  },
  "members": [
    "EC0_0.5",
    "E_tta_0.4"
  ],
  "weights": [
    0.25,
    0.75
  ]
}
```

Panels: original top-left, errors in ground-truth colors top-right (correct pixels white), prediction bottom-left, ground truth bottom-right.
The header reports image mIoU and 100%−image mIoU; the latter is not pixel error rate.

| Image | image mIoU | Comparison |
|---|---:|---|
| A117_step4_sample2 | 87.8554% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 69.9459% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 76.0324% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 80.0578% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 72.5655% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 86.6773% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 78.2944% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 81.2446% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 82.7666% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 82.6452% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 82.5579% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 89.9341% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 82.1502% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 82.2201% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 83.6432% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 75.5724% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 41.9753% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 86.8652% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 75.9965% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 81.8534% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
