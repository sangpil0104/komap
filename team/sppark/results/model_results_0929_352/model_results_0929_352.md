# REPRO8_GP05_sharpen — HRNet_taeho_no_gaussian inference

image mIoU: **79.273716%** (valid20, empty union=0)

Control: `GP05_sharpen`
Change: `{'fresh_raw_image_reproduction': True}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "REPRO8_GP05_sharpen",
  "contrast": "GP05_sharpen",
  "change": {
    "fresh_raw_image_reproduction": true
  },
  "recipe": {
    "checkpoint": "/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt",
    "target": "GP05_sharpen",
    "nodes": {
      "M01_gamma": {
        "arm": {
          "id": "M01_gamma",
          "contrast": "C640_d4_bicubic",
          "change": {
            "gamma": 0.9
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
            "halo": 0,
            "precision": "fp32",
            "gamma": 0.9,
            "contrast": 1.0,
            "sharpen": 0.0
          }
        },
        "batch": 4
      },
      "GP05_sharpen": {
        "arm": {
          "id": "GP05_sharpen",
          "contrast": "M01_gamma",
          "change": {
            "kind": "sharpen",
            "amount": 0.25,
            "sigma": 1.0
          },
          "cache_source": "experiments/hrnet_no_gaussian_inference_v1/outputs/20260929T131503_450078Z/arms/M01_gamma/probs"
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
| A117_step4_sample2 | 87.8589% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 72.2745% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 75.5943% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 79.8401% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 72.3155% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 86.3916% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 78.3583% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 81.6282% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 82.6576% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 82.7207% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 83.4270% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 90.1120% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 82.1277% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 82.4029% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 83.4709% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 75.9946% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 43.2938% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 87.0251% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 75.5911% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 82.3895% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
