# P20_smooth — HRNet_taeho_no_gaussian inference

image mIoU: **78.907415%** (valid20, empty union=0)

Control: `A00_control`
Change: `{'kind': 'smooth', 'sigma': 0.35}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "P20_smooth",
  "contrast": "A00_control",
  "change": {
    "kind": "smooth",
    "sigma": 0.35
  },
  "cache_source": "experiments/hrnet_no_gaussian_inference_v1/outputs/20260929T115651_434343Z/arms/A00_control/probs"
}
```

Panels: original top-left, errors in ground-truth colors top-right (correct pixels white), prediction bottom-left, ground truth bottom-right.
The header reports image mIoU and 100%−image mIoU; the latter is not pixel error rate.

| Image | image mIoU | Comparison |
|---|---:|---|
| A117_step4_sample2 | 87.7818% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 69.4456% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 75.1883% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 79.9967% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 72.5305% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 86.8253% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 78.5406% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 80.6513% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 82.9838% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 82.6793% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 82.2654% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 89.9295% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 82.0394% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 82.0567% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 83.5374% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 75.5399% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 41.7767% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 86.7013% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 75.8446% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 81.8343% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
