# P32_guided — HRNet_taeho_no_gaussian inference

image mIoU: **77.172746%** (valid20, empty union=0)

Control: `A00_control`
Change: `{'kind': 'guided', 'radius': 2, 'eps': 0.01}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "P32_guided",
  "contrast": "A00_control",
  "change": {
    "kind": "guided",
    "radius": 2,
    "eps": 0.01
  },
  "cache_source": "experiments/hrnet_no_gaussian_inference_v1/outputs/20260929T115651_434343Z/arms/A00_control/probs"
}
```

Panels: original top-left, errors in ground-truth colors top-right (correct pixels white), prediction bottom-left, ground truth bottom-right.
The header reports image mIoU and 100%−image mIoU; the latter is not pixel error rate.

| Image | image mIoU | Comparison |
|---|---:|---|
| A117_step4_sample2 | 86.2013% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 67.4681% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 71.1793% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 78.0949% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 71.4554% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 85.5362% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 77.4866% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 80.2607% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 81.6242% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 81.2840% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 81.3241% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 88.8408% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 80.1719% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 79.5086% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 80.6205% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 72.8270% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 40.9680% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 84.5021% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 73.4741% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 80.6272% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
