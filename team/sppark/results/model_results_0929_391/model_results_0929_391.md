# WG_bicubic_0.85 — HRNet_taeho_no_gaussian inference

image mIoU: **79.248449%** (valid20, empty union=0)

Control: `W01_full_factor`
Change: `{'input_transform': {'kind': 'gamma', 'gamma': 0.85}}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "WG_bicubic_0.85",
  "contrast": "W01_full_factor",
  "change": {
    "input_transform": {
      "kind": "gamma",
      "gamma": 0.85
    }
  },
  "input_transform": {
    "kind": "gamma",
    "gamma": 0.85
  },
  "engine": "full_v2",
  "config": {
    "scale": 2.0,
    "tta": "d4",
    "precision": "fp32",
    "restore_mode": "bicubic"
  }
}
```

Panels: original top-left, errors in ground-truth colors top-right (correct pixels white), prediction bottom-left, ground truth bottom-right.
The header reports image mIoU and 100%−image mIoU; the latter is not pixel error rate.

| Image | image mIoU | Comparison |
|---|---:|---|
| A117_step4_sample2 | 87.8430% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 73.0498% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 76.1586% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 79.6987% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 72.2823% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 86.3419% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 77.9493% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 81.4467% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 82.0490% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 82.4637% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 83.4193% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 90.1134% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 81.9664% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 82.3343% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 83.2053% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 76.0225% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 43.7871% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 87.3690% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 74.9903% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 82.4786% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
