# BM08_bilateral_messages — HRNet_taeho_no_gaussian inference

image mIoU: **78.618145%** (valid20, empty union=0)

Control: `A00_control`
Change: `{'kind': 'bilateral_messages', 'strength': 2.0, 'color_sigma': 0.2, 'radius': 3, 'space_sigma': 1.5, 'iterations': 3}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "BM08_bilateral_messages",
  "contrast": "A00_control",
  "change": {
    "kind": "bilateral_messages",
    "strength": 2.0,
    "color_sigma": 0.2,
    "radius": 3,
    "space_sigma": 1.5,
    "iterations": 3
  },
  "cache_source": "experiments/hrnet_no_gaussian_inference_v1/outputs/20260929T115651_434343Z/arms/A00_control/probs"
}
```

Panels: original top-left, errors in ground-truth colors top-right (correct pixels white), prediction bottom-left, ground truth bottom-right.
The header reports image mIoU and 100%−image mIoU; the latter is not pixel error rate.

| Image | image mIoU | Comparison |
|---|---:|---|
| A117_step4_sample2 | 87.6266% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 68.9907% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 74.2606% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 79.7292% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 72.6233% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 86.5537% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 78.4410% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 80.5692% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 82.8828% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 82.5089% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 82.1776% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 89.7843% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 81.7281% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 81.5216% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 83.0586% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 74.8675% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 41.8460% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 86.1725% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 75.3286% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 81.6921% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
