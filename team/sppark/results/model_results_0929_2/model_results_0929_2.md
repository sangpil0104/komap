# lhs_Model326

- Validation image mIoU: **78.031768%** (20 images)
- 이미지마다 4개 클래스 IoU를 평균한 뒤 이미지 20장을 평균합니다. union=0이면 IoU=0입니다.
- 오류 패널: 정답 픽셀은 흰색, 오답 픽셀은 해당 위치의 정답 클래스 색입니다.
- 오른쪽 위 숫자는 image mIoU 및 100% − image mIoU입니다. 후자는 픽셀 오류율과 다릅니다.

| Image | (H, W) | image mIoU | Comparison |
|---|---|---:|---|
| A117_step4_sample2_image.png | (875, 1230) | 85.1772% | [mask](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4_image.png | (480, 675) | 75.8260% | [mask](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4_image.png | (480, 675) | 74.0982% | [mask](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4_image.png | (875, 1230) | 79.0505% | [mask](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3_image.png | (909, 1278) | 72.1793% | [mask](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3_image.png | (480, 675) | 86.4873% | [mask](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1_image.png | (1024, 1440) | 75.1146% | [mask](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2_image.png | (909, 1278) | 76.0075% | [mask](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3_image.png | (1024, 1440) | 81.0335% | [mask](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3_image.png | (909, 1278) | 82.9025% | [mask](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4_image.png | (875, 1230) | 83.2428% | [mask](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3_image.png | (909, 1278) | 89.6334% | [mask](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1_image.png | (780, 1096) | 81.4587% | [mask](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1_image.png | (480, 675) | 79.7990% | [mask](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4_image.png | (480, 675) | 81.7333% | [mask](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1_image.png | (480, 675) | 73.8677% | [mask](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1_image.png | (480, 675) | 42.8956% | [mask](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4_image.png | (480, 675) | 85.1159% | [mask](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4_image.png | (480, 675) | 73.3827% | [mask](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4_image.png | (909, 1278) | 81.6296% | [mask](valid_mask_comparison/A7_step1_sample4.png) |

## 실행 정보

```json
{
  "run": "/Data3/sppark/komap/experiments/lhs/outputs/runs/Model326_20260929_075053_133567",
  "probabilities": "/Data3/sppark/komap/experiments/lhs/outputs/runs/Model326_20260929_075053_133567/evaluations/20260929_084255_966637/probabilities.npz",
  "config": {
    "arch": "unet",
    "encoder": "efficientnet-b4",
    "norm": "image_global",
    "crop": 1536,
    "scale": 3.0,
    "target_width": 0,
    "sample": "area",
    "bs": 2,
    "iters": 16000,
    "lr": 0.0003,
    "wd": 0.01,
    "dice_w": 1.0,
    "cldice_w": 0.0,
    "cldice_classes": "1,3",
    "cldice_iters": 10,
    "cldice_pool": 1,
    "grad_ckpt": false,
    "ema": 0.0,
    "accum": 1,
    "eval_every": 1000,
    "no_valid_eval": true,
    "amp_init_scale": 1024,
    "workers": 4,
    "seed": 0,
    "input": "zcdf3",
    "amp_dtype": "float16",
    "feat_stats": null,
    "deterministic_workers": false
  },
  "members": 4,
  "tta": "D4",
  "selection": "fixed final step 16000 per fold; no validation checkpoint selection"
}
```
