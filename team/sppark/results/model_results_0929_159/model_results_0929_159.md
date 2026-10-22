# EG1_0.25 — HRNet_taeho_no_gaussian inference

image mIoU: **79.018837%** (valid20, empty union=0)

Control: `B_gamma_0.85`
Change: `{'photometric_ensemble_weight': 0.25, 'member': 'B_gamma_0.95'}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "EG1_0.25",
  "contrast": "B_gamma_0.85",
  "change": {
    "photometric_ensemble_weight": 0.25,
    "member": "B_gamma_0.95"
  },
  "members": [
    "B_gamma_0.85",
    "B_gamma_0.95"
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
| A117_step4_sample2 | 87.7603% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 71.8128% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 72.6994% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 79.6355% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 72.6419% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 86.2562% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 79.1138% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 81.1454% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 83.4205% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 82.6565% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 82.9804% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 90.1640% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 81.5350% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 82.0472% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 83.1342% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 75.7450% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 43.1363% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 86.7820% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 75.3583% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 82.3522% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
