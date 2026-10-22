# B01_input_bicubic_aa — HRNet_taeho_no_gaussian inference

image mIoU: **78.396820%** (valid20, empty union=0)

Control: `A00_control`
Change: `{'input_antialias': True}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "B01_input_bicubic_aa",
  "contrast": "A00_control",
  "change": {
    "input_antialias": true
  },
  "config": {
    "patch": 512,
    "scale": 2.0,
    "overlap": 170,
    "input_mode": "bicubic",
    "input_antialias": true,
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
| A117_step4_sample2 | 87.1307% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 68.7745% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 74.4533% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 79.4761% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 72.5204% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 86.6252% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 78.0660% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 82.1975% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 82.5047% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 82.0939% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 81.0715% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 89.3294% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 81.4185% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 81.4209% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 82.8668% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 74.7478% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 41.7114% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 85.8203% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 74.9127% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 80.7949% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
