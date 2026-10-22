# EC2_0.75 — HRNet_taeho_no_gaussian inference

image mIoU: **78.973693%** (valid20, empty union=0)

Control: `B_patch_1024`
Change: `{'context_ensemble_weight': 0.75, 'added_member': 'A00_control'}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "EC2_0.75",
  "contrast": "B_patch_1024",
  "change": {
    "context_ensemble_weight": 0.75,
    "added_member": "A00_control"
  },
  "members": [
    "A00_control",
    "B_patch_1024"
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
| A117_step4_sample2 | 87.7944% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 69.6275% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 75.8961% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 80.0144% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 72.5445% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 86.7404% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 78.3673% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 80.6678% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 82.7473% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 82.6629% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 82.3514% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 89.9488% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 82.0421% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 82.1037% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 83.5996% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 75.6405% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 41.9518% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 86.9601% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 75.8608% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 81.9526% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
