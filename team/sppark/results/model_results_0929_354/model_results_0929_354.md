# EH2_geometric — HRNet_taeho_no_gaussian inference

image mIoU: **79.331815%** (valid20, empty union=0)

Control: `EH2_arithmetic`
Change: `{'ensemble_rule': 'geometric', 'members': ['Y_gamma0.9_halo16', 'M01_gamma', 'Y_gamma0.9_halo32'], 'weights': [1, 1, 1]}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "EH2_geometric",
  "contrast": "EH2_arithmetic",
  "change": {
    "ensemble_rule": "geometric",
    "members": [
      "Y_gamma0.9_halo16",
      "M01_gamma",
      "Y_gamma0.9_halo32"
    ],
    "weights": [
      1,
      1,
      1
    ]
  },
  "aggregation": {
    "kind": "geometric"
  },
  "members": [
    "Y_gamma0.9_halo16",
    "M01_gamma",
    "Y_gamma0.9_halo32"
  ],
  "weights": [
    1,
    1,
    1
  ]
}
```

Panels: original top-left, errors in ground-truth colors top-right (correct pixels white), prediction bottom-left, ground truth bottom-right.
The header reports image mIoU and 100%−image mIoU; the latter is not pixel error rate.

| Image | image mIoU | Comparison |
|---|---:|---|
| A117_step4_sample2 | 87.8840% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 72.2775% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 75.8840% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 79.8684% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 72.4456% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 86.4774% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 78.3604% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 81.6981% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 82.6994% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 83.0872% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 83.3044% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 90.2093% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 82.1574% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 82.3105% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 83.4948% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 76.0334% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 43.4280% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 86.9636% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 75.5471% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 82.5058% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
