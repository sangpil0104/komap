# K03_adaptive_scale — HRNet_taeho_no_gaussian inference

image mIoU: **77.932586%** (valid20, empty union=0)

Control: `A00_control`
Change: `{'scale_policy': {'kind': 'minimum_width', 'width': 2700, 'max_scale': 4.0}}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "K03_adaptive_scale",
  "contrast": "A00_control",
  "change": {
    "scale_policy": {
      "kind": "minimum_width",
      "width": 2700,
      "max_scale": 4.0
    }
  },
  "adaptive_scale": {
    "kind": "minimum_width",
    "width": 2700,
    "max_scale": 4.0
  },
  "config": {
    "patch": 512,
    "scale": 2.0,
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
  },
  "overlap_rule": "Preserve native overlap fraction; no labels or image identity used"
}
```

Panels: original top-left, errors in ground-truth colors top-right (correct pixels white), prediction bottom-left, ground truth bottom-right.
The header reports image mIoU and 100%−image mIoU; the latter is not pixel error rate.

| Image | image mIoU | Comparison |
|---|---:|---|
| A117_step4_sample2 | 87.3581% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 73.2768% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 68.6480% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 79.6882% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 71.9230% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 85.5687% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 78.5385% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 82.0778% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 82.9797% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 82.2625% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 80.0259% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 89.1491% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 82.2802% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 82.9009% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 83.3271% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 70.8727% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 42.2189% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 80.5131% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 74.1681% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 80.8744% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
