# Q03_class_weight — HRNet_taeho_no_gaussian inference

image mIoU: **78.242339%** (valid20, empty union=0)

Control: `A00_control`
Change: `{'kind': 'class_weight', 'class': 1, 'weight': 3.0}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "Q03_class_weight",
  "contrast": "A00_control",
  "change": {
    "kind": "class_weight",
    "class": 1,
    "weight": 3.0
  },
  "cache_source": "experiments/hrnet_no_gaussian_inference_v1/outputs/20260929T115651_434343Z/arms/A00_control/probs"
}
```

Panels: original top-left, errors in ground-truth colors top-right (correct pixels white), prediction bottom-left, ground truth bottom-right.
The header reports image mIoU and 100%−image mIoU; the latter is not pixel error rate.

| Image | image mIoU | Comparison |
|---|---:|---|
| A117_step4_sample2 | 87.2182% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 72.4801% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 72.2150% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 77.8278% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 67.8611% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 85.6238% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 78.0207% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 80.3144% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 83.6557% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 81.5930% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 82.2420% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 89.9149% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 80.8676% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 80.9099% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 81.5768% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 76.2007% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 43.9217% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 87.1926% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 72.6718% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 82.5390% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
