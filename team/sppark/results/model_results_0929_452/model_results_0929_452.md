# ZW_WG_bicubic_0.9_0.1 — HRNet_taeho_no_gaussian inference

image mIoU: **79.509320%** (valid20, empty union=0)

Control: `WG_bicubic_0.9`
Change: `{'normalization_blend': {'scope': 'decoder', 'amount': 0.1}}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "ZW_WG_bicubic_0.9_0.1",
  "contrast": "WG_bicubic_0.9",
  "change": {
    "normalization_blend": {
      "scope": "decoder",
      "amount": 0.1
    }
  },
  "forward_transform": {
    "normalization_blend": {
      "scope": "decoder",
      "amount": 0.1
    }
  },
  "engine": "full_v2",
  "config": {
    "scale": 2.0,
    "tta": "d4",
    "precision": "fp32",
    "restore_mode": "bicubic"
  },
  "input_transform": {
    "kind": "gamma",
    "gamma": 0.9
  }
}
```

Panels: original top-left, errors in ground-truth colors top-right (correct pixels white), prediction bottom-left, ground truth bottom-right.
The header reports image mIoU and 100%−image mIoU; the latter is not pixel error rate.

| Image | image mIoU | Comparison |
|---|---:|---|
| A117_step4_sample2 | 87.7599% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 75.4437% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 75.6628% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 79.4487% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 72.5531% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 86.4973% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 77.9188% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 80.8631% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 81.7736% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 82.5429% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 83.7897% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 90.1246% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 81.8244% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 82.5076% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 83.4191% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 76.2512% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 45.7153% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 87.0010% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 76.4780% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 82.6116% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
