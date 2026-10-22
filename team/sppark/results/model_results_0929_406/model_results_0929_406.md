# ZN_amount0.5 — HRNet_taeho_no_gaussian inference

image mIoU: **78.300363%** (valid20, empty union=0)

Control: `Z_norm_decoder_0.1`
Change: `{'normalization_blend_amount': 0.5}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "ZN_amount0.5",
  "contrast": "Z_norm_decoder_0.1",
  "change": {
    "normalization_blend_amount": 0.5
  },
  "forward_transform": {
    "normalization_blend": {
      "scope": "decoder",
      "amount": 0.5
    }
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
| A117_step4_sample2 | 86.3769% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 77.0051% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 73.1706% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 77.9514% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 68.0888% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 86.2685% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 78.1182% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 78.4508% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 81.4868% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 82.3038% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 82.6746% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 89.8666% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 81.4832% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 81.9509% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 82.9037% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 73.5785% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 45.7529% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 84.5466% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 74.3179% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 79.7117% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
