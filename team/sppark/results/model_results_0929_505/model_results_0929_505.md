# NF06_norm_refine — HRNet_taeho_no_gaussian inference

image mIoU: **79.507538%** (valid20, empty union=0)

Control: `ZN_combine_Y_gamma0.9_halo16_0.05`
Change: `{'normalization_blend': {'scope': 'decoder', 'amount': 0.05, 'subset': 'except_last'}}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "NF06_norm_refine",
  "contrast": "ZN_combine_Y_gamma0.9_halo16_0.05",
  "change": {
    "normalization_blend": {
      "scope": "decoder",
      "amount": 0.05,
      "subset": "except_last"
    }
  },
  "forward_transform": {
    "normalization_blend": {
      "scope": "decoder",
      "amount": 0.05,
      "subset": "except_last"
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
    "halo": 16,
    "precision": "fp32",
    "gamma": 0.9,
    "contrast": 1.0,
    "sharpen": 0.0
  }
}
```

Panels: original top-left, errors in ground-truth colors top-right (correct pixels white), prediction bottom-left, ground truth bottom-right.
The header reports image mIoU and 100%−image mIoU; the latter is not pixel error rate.

| Image | image mIoU | Comparison |
|---|---:|---|
| A117_step4_sample2 | 87.8337% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 73.6757% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 75.5893% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 79.5414% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 72.6474% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 86.7629% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 78.4713% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 81.4529% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 82.6889% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 83.4518% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 83.5016% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 90.1714% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 82.1203% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 82.3676% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 83.4621% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 76.2936% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 44.4593% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 86.8650% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 76.0996% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 82.6950% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
