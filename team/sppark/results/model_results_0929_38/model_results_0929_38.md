# B_scale_1.5 — HRNet_taeho_no_gaussian inference

image mIoU: **76.468124%** (valid20, empty union=0)

Control: `A00_control`
Change: `{'scale': 1.5}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "B_scale_1.5",
  "contrast": "A00_control",
  "change": {
    "scale": 1.5
  },
  "config": {
    "patch": 512,
    "scale": 1.5,
    "overlap": 170,
    "input_mode": "bicubic",
    "input_antialias": false,
    "restore": "bilinear",
    "restore_antialias": false,
    "restore_logits": false,
    "window": "gaussian",
    "sigma": 0.125,
    "tta": "flip4",
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
| A117_step4_sample2 | 86.2807% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 67.1509% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 68.2822% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 78.1259% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 71.7949% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 85.2315% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 77.6333% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 76.4043% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 81.6855% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 81.4869% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 82.5053% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 89.3427% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 79.7843% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 77.5527% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 78.2381% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 73.0307% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 40.0552% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 83.2696% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 71.2554% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 80.2524% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
