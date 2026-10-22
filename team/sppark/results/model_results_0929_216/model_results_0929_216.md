# H_resize_384 — HRNet_taeho_no_gaussian inference

image mIoU: **76.272816%** (valid20, empty union=0)

Control: `G00_batch1_control`
Change: `{'model_input_side': 384}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "H_resize_384",
  "contrast": "G00_batch1_control",
  "change": {
    "model_input_side": 384
  },
  "fixed": {
    "native_context": 256,
    "overlap": 170
  },
  "config": {
    "patch": 384,
    "scale": 1.5,
    "overlap": 170,
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
| A117_step4_sample2 | 86.1697% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 66.7047% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 66.4865% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 78.1573% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 71.8705% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 85.1542% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 77.9991% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 76.1236% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 82.1921% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 81.7727% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 82.1354% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 89.2849% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 79.7116% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 77.4919% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 77.8683% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 72.7406% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 39.7556% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 82.5007% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 71.2075% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 80.1297% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
