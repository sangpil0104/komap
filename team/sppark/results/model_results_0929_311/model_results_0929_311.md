# I05_clahe — HRNet_taeho_no_gaussian inference

image mIoU: **78.822067%** (valid20, empty union=0)

Control: `C640_d4_bicubic`
Change: `{'input_transform': {'kind': 'clahe', 'clip': 2.0, 'grid': 8, 'strength': 0.25}}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "I05_clahe",
  "contrast": "C640_d4_bicubic",
  "change": {
    "input_transform": {
      "kind": "clahe",
      "clip": 2.0,
      "grid": 8,
      "strength": 0.25
    }
  },
  "input_transform": {
    "kind": "clahe",
    "clip": 2.0,
    "grid": 8,
    "strength": 0.25
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
| A117_step4_sample2 | 87.7907% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 69.8590% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 76.5360% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 79.9881% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 72.0813% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 86.7195% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 77.5910% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 81.1239% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 82.1435% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 82.5314% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 81.7312% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 89.5553% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 82.1777% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 82.3966% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 83.3855% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 75.2215% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 41.0927% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 86.9426% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 76.0589% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 81.5150% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
