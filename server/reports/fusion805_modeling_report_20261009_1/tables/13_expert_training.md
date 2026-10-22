**표 13. E 학습 설정**

| 항목 | B4 | HRNet |
| --- | --- | --- |
| 구조 | EfficientNet-B4 + U-Net | HRNet-W48 + U-Net++ + boundary head |
| 구성 | 4 folds × seed0 | 4 folds × seeds1·2 |
| 초기화 | ImageNet encoder | ImageNet encoder |
| 학습량 | 16,000 iterations/fold | 70 epochs/fold |
| batch / accumulation | 2 / 1 | 4 / 2 |
| crop | 1536 | 512 |
| learning rate | 3e-4 | 3e-4 |
| weight decay | 1e-2 | 1e-4 |
| optimizer | AdamW | AdamW |
| 손실 | CE + Dice | 0.5 CE + 0.5 Dice + boundary loss |
| checkpoint 선택 | last.pt | last.pt |
| valid20 학습 중 평가 | no_valid_eval | val_every=0 |
| worker/RNG | 고정 worker seed | rng_mode=legacy, seeds1·2 |

HRNet boundary loss는 BCE+Dice이며 가중치 1.0으로 더한다. CE 클래스 가중치는 E에서 사용하지 않는다. [S05, S12]
