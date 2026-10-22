# P23_smooth — HRNet_taeho_no_gaussian inference

image mIoU: **76.626697%** (valid20, empty union=0)

Control: `A00_control`
Change: `{'kind': 'smooth', 'sigma': 1.0}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "P23_smooth",
  "contrast": "A00_control",
  "change": {
    "kind": "smooth",
    "sigma": 1.0
  },
  "cache_source": "experiments/hrnet_no_gaussian_inference_v1/outputs/20260929T115651_434343Z/arms/A00_control/probs"
}
```

Panels: original top-left, errors in ground-truth colors top-right (correct pixels white), prediction bottom-left, ground truth bottom-right.
The header reports image mIoU and 100%−image mIoU; the latter is not pixel error rate.

| Image | image mIoU | Comparison |
|---|---:|---|
| A117_step4_sample2 | 86.7428% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 67.1758% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 69.0534% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 78.6564% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 71.8969% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 84.4420% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 78.1158% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 80.4157% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 82.4368% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 81.8256% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 81.8876% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 89.5277% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 79.8236% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 76.8214% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 77.4090% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 70.6952% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 39.8070% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 84.0207% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 70.7826% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 80.9981% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
