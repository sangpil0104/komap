# G_native_160 — HRNet_taeho_no_gaussian inference

image mIoU: **78.522472%** (valid20, empty union=0)

Control: `G00_batch1_control`
Change: `{'native_context': 160}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "G_native_160",
  "contrast": "G00_batch1_control",
  "change": {
    "native_context": 160
  },
  "fixed": {
    "magnification": 2.0,
    "relative_stride": 0.3359375
  },
  "config": {
    "patch": 320,
    "scale": 2.0,
    "overlap": 106,
    "input_mode": "bicubic",
    "input_antialias": false,
    "restore": "bilinear",
    "restore_antialias": false,
    "restore_logits": false,
    "window": "gaussian",
    "sigma": 0.125,
    "tta": "flip4",
    "tile_merge": "prob",
    "tta_merge": "prob",
    "halo": 0,
    "precision": "fp32",
    "gamma": 1.0,
    "contrast": 1.0,
    "sharpen": 0.0
  }
}
```

Panels: original top-left, errors in ground-truth colors top-right (correct pixels white), prediction bottom-left, ground truth bottom-right.
The header reports image mIoU and 100%−image mIoU; the latter is not pixel error rate.

| Image | image mIoU | Comparison |
|---|---:|---|
| A117_step4_sample2 | 87.7222% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 69.4640% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 72.3408% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 79.7636% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 72.4110% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 86.3190% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 78.8563% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 81.6302% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 83.7685% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 82.6213% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 80.7032% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 89.8492% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 82.1432% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 81.9625% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 83.3587% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 74.9163% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 40.6746% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 85.3548% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 75.3057% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 81.2843% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
