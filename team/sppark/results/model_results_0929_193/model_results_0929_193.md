# LP02_class_weight — HRNet_taeho_no_gaussian inference

image mIoU: **79.115324%** (valid20, empty union=0)

Control: `C640_d4_bicubic`
Change: `{'kind': 'class_weight', 'class': 3, 'weight': 0.8}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "LP02_class_weight",
  "contrast": "C640_d4_bicubic",
  "change": {
    "kind": "class_weight",
    "class": 3,
    "weight": 0.8
  },
  "cache_source": "/Data3/sppark/komap/experiments/hrnet_no_gaussian_inference_v1/outputs/20260929T124550_643473Z/arms/C640_d4_bicubic/probs"
}
```

Panels: original top-left, errors in ground-truth colors top-right (correct pixels white), prediction bottom-left, ground truth bottom-right.
The header reports image mIoU and 100%−image mIoU; the latter is not pixel error rate.

| Image | image mIoU | Comparison |
|---|---:|---|
| A117_step4_sample2 | 87.8453% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 70.8701% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 76.6155% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 79.9046% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 72.0727% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 86.6756% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 78.0751% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 81.4623% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 82.7111% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 82.5708% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 82.8153% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 89.9842% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 82.1196% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 82.2903% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 83.6774% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 75.9466% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 42.7444% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 86.9325% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 75.5458% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 81.4473% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
