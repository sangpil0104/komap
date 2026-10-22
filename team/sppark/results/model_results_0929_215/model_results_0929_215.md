# H_resize_256 — HRNet_taeho_no_gaussian inference

image mIoU: **69.994579%** (valid20, empty union=0)

Control: `G00_batch1_control`
Change: `{'model_input_side': 256}`

Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.
Checkpoint: `/Data3/sppark/komap/experiments/HRNet_taeho_multigpu/outputs/runs/HRNet_taeho_no_gaussian_0928_ddp6_20260928T131839_853555Z/last.pt`

```json
{
  "id": "H_resize_256",
  "contrast": "G00_batch1_control",
  "change": {
    "model_input_side": 256
  },
  "fixed": {
    "native_context": 256,
    "overlap": 170
  },
  "config": {
    "patch": 256,
    "scale": 1.0,
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
| A117_step4_sample2 | 80.2214% | [panel](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 64.6527% | [panel](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 56.0194% | [panel](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 71.6213% | [panel](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 66.3008% | [panel](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 73.7499% | [panel](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 78.0394% | [panel](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 70.3475% | [panel](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 80.7366% | [panel](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 79.0678% | [panel](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 80.1419% | [panel](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 86.5894% | [panel](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 72.6532% | [panel](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 67.2129% | [panel](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 62.6258% | [panel](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 61.9266% | [panel](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 37.2693% | [panel](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 76.1530% | [panel](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 57.7587% | [panel](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 76.8042% | [panel](valid_mask_comparison/A7_step1_sample4.png) |
