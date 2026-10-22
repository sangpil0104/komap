# REPRO8_ZN_combine_Y_gamma0.9_halo16_0.05 — HRNet_taeho_no_gaussian inference

image mIoU: **79.540466%** (valid20, empty union=0)

Control: `ZN_combine_Y_gamma0.9_halo16_0.05`
Change: `{'fresh_raw_image_reproduction': True}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "REPRO8_ZN_combine_Y_gamma0.9_halo16_0.05",
  "contrast": "ZN_combine_Y_gamma0.9_halo16_0.05",
  "change": {
    "fresh_raw_image_reproduction": true
  },
  "recipe": {
    "checkpoint": "/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt",
    "target": "ZN_combine_Y_gamma0.9_halo16_0.05",
    "nodes": {
      "ZN_combine_Y_gamma0.9_halo16_0.05": {
        "arm": {
          "id": "ZN_combine_Y_gamma0.9_halo16_0.05",
          "contrast": "Y_gamma0.9_halo16",
          "change": {
            "normalization_blend": {
              "scope": "decoder",
              "amount": 0.05
            }
          },
          "forward_transform": {
            "normalization_blend": {
              "scope": "decoder",
              "amount": 0.05
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
        },
        "batch": 4
      }
    }
  }
}
```

Panels: original top-left, errors in ground-truth colors top-right (correct pixels white), prediction bottom-left, ground truth bottom-right.
The header reports image mIoU and 100%−image mIoU; the latter is not pixel error rate.

| Image | image mIoU | Comparison |
|---|---:|---|
| A117_step4_sample2 | 87.7755% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 74.3744% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 75.4400% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 79.4216% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 72.7125% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 86.7430% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 78.4839% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 81.3797% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 82.6794% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 83.3948% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 83.5162% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 90.1625% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 82.0327% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 82.3986% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 83.4687% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 76.3272% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 44.8147% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 86.8186% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 76.2311% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 82.6341% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
