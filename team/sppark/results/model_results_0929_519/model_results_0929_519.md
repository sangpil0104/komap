# NC_native256_norm0.1 — HRNet_taeho_no_gaussian inference

image mIoU: **79.285529%** (valid20, empty union=0)

Control: `NC_control_0.1`
Change: `{'native_context': 256}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "NC_native256_norm0.1",
  "contrast": "NC_control_0.1",
  "change": {
    "native_context": 256
  },
  "fixed": {
    "magnification": 2.0,
    "relative_stride": 0.46875
  },
  "config": {
    "patch": 512,
    "scale": 2.0,
    "overlap": 136,
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
    "halo": 16,
    "precision": "fp32",
    "gamma": 0.9,
    "contrast": 1.0,
    "sharpen": 0.0
  },
  "forward_transform": {
    "normalization_blend": {
      "scope": "decoder",
      "amount": 0.1
    }
  }
}
```

Panels: original top-left, errors in ground-truth colors top-right (correct pixels white), prediction bottom-left, ground truth bottom-right.
The header reports image mIoU and 100%−image mIoU; the latter is not pixel error rate.

| Image | image mIoU | Comparison |
|---|---:|---|
| A117_step4_sample2 | 87.3065% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 75.5167% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 72.6840% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 78.8777% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 72.5112% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 86.5829% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 78.9152% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 81.2488% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 82.9730% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 83.0663% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 82.8510% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 90.1479% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 81.8263% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 82.2559% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 83.2611% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 75.2189% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 45.2830% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 86.4004% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 76.4546% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 82.3290% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
