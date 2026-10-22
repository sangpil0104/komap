# P30_guided — HRNet_taeho_no_gaussian inference

image mIoU: **77.665103%** (valid20, empty union=0)

Control: `A00_control`
Change: `{'kind': 'guided', 'radius': 2, 'eps': 0.0004}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "P30_guided",
  "contrast": "A00_control",
  "change": {
    "kind": "guided",
    "radius": 2,
    "eps": 0.0004
  },
  "cache_source": "experiments/hrnet_no_gaussian_inference_v1/outputs/20260929T115651_434343Z/arms/A00_control/probs"
}
```

Panels: original top-left, errors in ground-truth colors top-right (correct pixels white), prediction bottom-left, ground truth bottom-right.
The header reports image mIoU and 100%−image mIoU; the latter is not pixel error rate.

| Image | image mIoU | Comparison |
|---|---:|---|
| A117_step4_sample2 | 86.8702% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 67.8320% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 72.8115% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 78.5630% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 71.2844% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 85.9001% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 77.8683% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 80.4455% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 82.0778% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 81.5288% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 81.8811% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 88.9637% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 80.4991% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 80.0017% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 81.6941% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 73.0951% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 41.1410% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 85.5456% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 74.4007% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 80.8984% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
