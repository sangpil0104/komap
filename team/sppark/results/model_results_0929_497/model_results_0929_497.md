# REPRO8_EN_norm0.1_weight0.5 — HRNet_taeho_no_gaussian inference

image mIoU: **79.511476%** (valid20, empty union=0)

Control: `EN_norm0.1_weight0.5`
Change: `{'fresh_raw_image_reproduction': True}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "REPRO8_EN_norm0.1_weight0.5",
  "contrast": "EN_norm0.1_weight0.5",
  "change": {
    "fresh_raw_image_reproduction": true
  },
  "recipe": {
    "checkpoint": "/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt",
    "target": "EN_norm0.1_weight0.5",
    "nodes": {
      "Y_gamma0.85_halo16": {
        "arm": {
          "id": "Y_gamma0.85_halo16",
          "contrast": "M00_gamma",
          "change": {
            "halo": 16
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
            "gamma": 0.85,
            "contrast": 1.0,
            "sharpen": 0.0
          }
        },
        "batch": 4
      },
      "Z_norm_decoder_0.1": {
        "arm": {
          "id": "Z_norm_decoder_0.1",
          "contrast": "C640_d4_bicubic",
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
            "gamma": 1.0,
            "contrast": 1.0,
            "sharpen": 0.0
          }
        },
        "batch": 4
      },
      "EN_norm0.1_weight0.5": {
        "arm": {
          "id": "EN_norm0.1_weight0.5",
          "contrast": "Y_gamma0.85_halo16",
          "change": {
            "normalization_branch_weight": 0.5
          },
          "members": [
            "Y_gamma0.85_halo16",
            "Z_norm_decoder_0.1"
          ],
          "weights": [
            0.5,
            0.5
          ]
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
| A117_step4_sample2 | 87.8254% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 73.8173% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 75.8073% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 79.5693% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 72.6772% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 86.5421% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 78.4892% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 81.3308% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 82.6260% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 83.0064% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 83.4548% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 90.2037% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 82.1095% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 82.4909% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 83.5695% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 76.3749% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 44.4563% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 86.9386% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 76.2075% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 82.7328% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
