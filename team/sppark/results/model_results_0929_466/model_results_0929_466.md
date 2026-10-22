# PB_gamma_contrast0.95 — HRNet_taeho_no_gaussian inference

image mIoU: **79.283344%** (valid20, empty union=0)

Control: `M01_gamma`
Change: `{'contrast': 0.95}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "PB_gamma_contrast0.95",
  "contrast": "M01_gamma",
  "change": {
    "contrast": 0.95
  },
  "config": {
    "patch": 640,
    "scale": 2.0,
    "overlap": 170,
    "input_mode": "bicubic",
    "input_antialias": false,
    "restore": "bicubic",
    "restore_antialias": false,
    "restore_logits": false,
    "window": "gaussian",
    "sigma": 0.125,
    "tta": "d4",
    "tile_merge": "prob",
    "tta_merge": "prob",
    "halo": 0,
    "precision": "fp32",
    "gamma": 0.9,
    "contrast": 0.95,
    "sharpen": 0.0
  }
}
```

Panels: original top-left, errors in ground-truth colors top-right (correct pixels white), prediction bottom-left, ground truth bottom-right.
The header reports image mIoU and 100%−image mIoU; the latter is not pixel error rate.

| Image | image mIoU | Comparison |
|---|---:|---|
| A117_step4_sample2 | 87.8495% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 72.7555% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 75.0337% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 79.8446% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 72.3028% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 86.4163% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 78.3688% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 81.8026% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 82.8166% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 82.6800% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 83.5042% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 90.1434% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 82.0703% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 82.3845% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 83.2804% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 76.0869% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 43.8329% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 87.0877% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 75.0105% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 82.3957% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
