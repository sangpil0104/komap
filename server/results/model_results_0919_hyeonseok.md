# KoMaP 모델 결과 기록

## 번호 체계

기존 기록의 `7.2`, `8.1`, `97`, `109`처럼 불연속적이거나 중복되는 번호를 정리했다.
아래의 **새 모델 번호**는 실제 실행한 모델·개선·앙상블 단위를 기준으로 1부터 연속 부여했다.
기존 번호는 재현을 위해 `구 번호`로 함께 보존했다.

- 팀원이 제공한 `EffNet-B4 U-Net: 77%`는 점수 기준이 확인되지 않아 모델 번호에서 제외했다.
- 단순 test mask 생성·추론 비교도 실행 결과 보존을 위해 모델 번호를 부여했다.
- `test.zip`에는 정답 mask가 없어 test 점수는 계산하지 못하고 예측 mask만 생성했다.

## 전체 연속 번호 인덱스

| 새 모델 | 구 번호 | 구성 | Validation mIoU |
|---:|---|---|---:|
| 1 | 1 | ResNet34 U-Net | 73.42% |
| 2 | 2 | ConvNeXt-Tiny U-Net | 71.44% |
| 3 | 3 | SegFormer-B0 | 65.56% |
| 4 | 4 | DINOv2 decoder | 42.38% |
| 5 | 5 | SAM2 decoder | 44.15% |
| 6 | 6 | ResNet34 U-Net++ | 74.44% |
| 7 | 7.2 | DeepLabV3+ | 62.30% |
| 8 | 7.3 | HRNet-W18 | 73.23% |
| 9 | 7.4 | Mamba-UNet-style | 73.59% |
| 10 | 8.1 | EfficientNet-B4 U-Net | 73.02% |
| 11 | 8.2 | EfficientNet-B4 U-Net++ | 73.45% |
| 12 | 8.3 | EfficientNet-B5 U-Net++ | 68.03% |
| 13 | 8.4 | EfficientNetV2-S multi-scale decoder | 73.10% |
| 14 | 8.5 | HRNet-W32 context head | 73.32% |
| 15 | 8.6 | HRNet-W48 context head | 68.56% |
| 16 | 8.7 | SegFormer-B2 | 67.70% |
| 17 | 8.8 | SegFormer-B3 | 67.15% |
| 18 | 8.9 | SegNeXt-style Tiny | 67.77% |
| 19 | 8.10 | SegNeXt-style Small | 64.83% |
| 20 | 6-TTA | 모델 6 + TTA | 74.74% |
| 21 | 6.1 | differential fine-tuning | 75.28% |
| 22 | 6.2 | weighted CE/Dice + safe augmentation | **75.38%** |
| 23 | 91 | MicroNet ResNet34 + U-Net++ | 72.15% |
| 24 | 92 | MicroNet EfficientNet-B4 + U-Net++ | 69.76% |
| 25 | 93 | 모델 6 staged fine-tuning | 75.22% |
| 26 | 94 | 모델 6 focal + Dice | 74.99% |
| 27 | 95 | 모델 6 boundary + Lovász + Dice | 75.21% |
| 28 | 96 | 모델 6 4-fold ensemble | 73.14% |
| 29 | 97 | native-resolution patch baseline | **77.85%** |
| 30 | 97-calibration | native patch + Al logit calibration | 78.54% |
| 31 | 98 | hard-negative fine-tuning | 79.17% |
| 32 | 99 | boundary emphasis fine-tuning | 79.16% |
| 33 | 100 | 모델 98/99 2-way ensemble | 79.36% |
| 34 | 101 | 640px patch fine-tuning | 79.20% |
| 35 | 102 | 모델 98/99/101 3-way ensemble | 79.42% |
| 36 | 103 | D4 + Gaussian + class-conditional inference | 79.45% |
| 37 | 104 | group-clean/full validation calibration | 79.46% |
| 38 | 105 | Al binary refiner | 79.46%* |
| 39 | 106 | soft boundary + local contrast | 79.26% |
| 40 | 107 | OOF hard-negative fine-tuning | 75.84% |
| 41 | 108 | 모델 31/32/34 advanced test inference | 추론 전용 |
| 42 | 109 | ResNet34 U-Net++ + residual ASPP | **79.48%** |
| 43 | 110 | high-resolution side branch | 79.34% |
| 44 | 111 | gated encoder 1×1 adapters | 79.38% |
| 45 | 112 | encoder BN running-stat freeze | 79.26% |
| 46 | 113 | encoder BN freeze + decoder GroupNorm | 78.80% |
| 47 | 114 | ConvNeXt-Tiny native patch | 73.70% |
| 48 | 115 | HRNet-W18 native patch | 74.88% |
| 49 | 116 | EfficientNet-B4 native patch | 75.27% |
| 50 | 117 | 기존 ensemble + ASPP ensemble | **79.48%** |

\* Al refiner는 alpha=0이 최적이라 실제로는 refiner를 사용하지 않은 baseline과 같은 결과다.

## 모델 1~19: 기본 백본 탐색

모델 1~6은 대회 초안의 기본 모델이며, 모델 7~19는 DeepLab·HRNet·Mamba 및 추가 백본 후보다.
모든 수치는 초기 square-resize 기반 train/valid 비교다.

- 모델 1: ResNet34 U-Net, 73.42%
- 모델 2: ConvNeXt-Tiny U-Net, 71.44%
- 모델 3: SegFormer-B0, 65.56%
- 모델 4: DINOv2 decoder, 42.38%
- 모델 5: SAM2 decoder, 44.15%
- 모델 6: ResNet34 U-Net++, 74.44%
- 모델 7~9: DeepLabV3+ 62.30%, HRNet-W18 73.23%, Mamba-UNet-style 73.59%
- 모델 10~19: EfficientNet·HRNet·SegFormer·SegNeXt 추가 후보

추가 후보 중 최고는 모델 11 EfficientNet-B4 U-Net++의 73.45%였고, 이 조건의 최고는
모델 6 ResNet34 U-Net++의 74.44%였다. 모델 13은 `tf_efficientnetv2_s` pretrained
encoder이며, 모델 18~19는 P100 호환 portable MSCA-style 변형이다.

## 모델 20~22: 모델 6 기반 개선

| 새 모델 | 변경 내용 | 일반 mIoU | TTA mIoU |
|---:|---|---:|---:|
| 20 | 모델 6 + TTA | 74.44% | 74.74% |
| 21 | differential fine-tuning + 기존 증강 | 74.94% | 75.28% |
| 22 | weighted CE + Dice + 약한 밝기·대비·noise 증강 | 75.11% | **75.38%** |

- 모델 22 checkpoint: `/workspace/komap_model1/outputs/model62_ft_weighted_safeaug/best.pt`
- 모델 22 test mask: `/workspace/komap_model1/outputs/model62_ft_weighted_safeaug/test_masks/`

## 모델 23~28: 논문 기반 추가 실험

| 새 모델 | 구성 | 일반 mIoU | TTA mIoU |
|---:|---|---:|---:|
| 23 | MicroNet ResNet34 + U-Net++ | 71.24% | 72.15% |
| 24 | MicroNet EfficientNet-B4 + U-Net++ | 68.45% | 69.76% |
| 25 | 모델 6 staged fine-tuning | 74.88% | 75.22% |
| 26 | 모델 6 focal + Dice | 74.73% | 74.99% |
| 27 | 모델 6 boundary + Lovász + Dice | 74.86% | 75.21% |
| 28 | 모델 6 4-fold ensemble | 72.65% | 73.14% |

MicroNet은 도메인 차이 때문에 ImageNet ResNet34보다 낮았다. focal·boundary/Lovász도
추가 개선을 만들지 못했다.

원격 결과 경로:

- 모델 23: `/workspace/komap_model1/outputs/model91_micronet_resnet34/`
- 모델 24: `/workspace/komap_model1/outputs/model92_micronet_effb4/`
- 모델 25: `/workspace/komap_model1/outputs/model93_staged_model6/`
- 모델 26: `/workspace/komap_model1/outputs/model94_focal_dice_model6/`
- 모델 27: `/workspace/komap_model1/outputs/model95_boundary_lovasz_model6/`
- 모델 28: `/workspace/komap_model1/outputs/model96_4fold_ensemble/`

## 모델 29~35: native-resolution patch 및 Al 개선

모델 29부터는 원본 가로세로비를 유지한 512px patch, overlap sliding-window, Primary Si/Al
중심 crop sampling을 사용했다. 모델 29의 클래스별 TTA IoU는 Primary Si 75.32%, Al3Ni
87.05%, Eutectic Si 92.90%, Al 56.13%였다.

| 새 모델 | 변경 내용 | Global mIoU | Image-wise 평균 |
|---:|---|---:|---:|
| 29 | native patch baseline | 77.85% | 74.46% |
| 30 | 모델 29 + Al bias -0.60 | 78.54% | 75.09% |
| 31 | hard-negative 560 center fine-tuning | **79.17%** | 75.11% |
| 32 | boundary emphasis fine-tuning | 79.16% | **75.31%** |
| 33 | 모델 31/32 2-way ensemble | 79.36% | 75.51% |
| 34 | 모델 31 기반 640px patch fine-tuning | 79.20% | 75.75% |
| 35 | 모델 31/32/34 3-way ensemble | **79.42%** | 75.68% |

주요 클래스 IoU:

- 모델 31: Primary 77.91%, Al3Ni 87.74%, Eutectic 93.70%, Al 57.33%
- 모델 32: Primary 77.46%, Al3Ni 87.63%, Eutectic 93.56%, Al 57.98%
- 모델 35: Primary 77.71%, Al3Ni 87.80%, Eutectic 93.81%, Al 58.34%

원격 경로:

- 모델 29: `/workspace/komap_model1/outputs/model97_native_patch/`
- 모델 31: `/workspace/komap_model1/outputs/model98_hard_negative/`
- 모델 32: `/workspace/komap_model1/outputs/model99_boundary_native/`
- 모델 33: `/workspace/komap_model1/outputs/model100_ensemble/`
- 모델 34: `/workspace/komap_model1/outputs/model101_highres640/`
- 모델 35: `/workspace/komap_model1/outputs/model102_three_ensemble/`

모델 29 test mask는 `model97_native_patch/test_masks/`, 모델 31·32 test mask는 각각
`test_masks_al_bias_m060/`, 모델 35 test mask는 `test_masks_best/`에 있다.

## 모델 36~41: 추론·calibration 추가 실험

Claude Code 리뷰를 참고해 pseudo-labeling은 제외하고 검증했다.

| 새 모델 | 실험 | 결과 |
|---:|---|---:|
| 36 | 8-way D4 + Gaussian + class-conditional ensemble | 79.45% |
| 37 | group-clean/full validation bias calibration | 79.46% |
| 38 | Al binary refiner | 79.46%* |
| 39 | soft boundary + local contrast | 79.26% |
| 40 | A-ID OOF hard-negative fine-tuning | 75.84% |
| 41 | 모델 31/32/34 advanced test inference | test mask 생성 |

모델 36은 기존 3-way ensemble에 D4 TTA와 Gaussian tile blending을 적용했다. 모델 37의
보수적 group-clean bias는 Primary 0.0, Eutectic 0.0, Al -0.20이며, 전체 validation에서
다시 고른 bias는 Primary +0.10, Eutectic +0.20, Al -0.20이었다. 단일 validation
calibration 과적합 가능성이 있어 보수적 bias도 함께 보관했다.

- 모델 36 결과: `/workspace/komap_model1/outputs/model103_advanced_inference/results.json`
- 모델 37 결과: `/workspace/komap_model1/outputs/model104_group_clean_calibration/results.json`
- 모델 38 결과: `/workspace/komap_model1/outputs/model105_al_refiner/results.json`
- 모델 39 결과: `/workspace/komap_model1/outputs/model106_soft_boundary_clahe/`
- 모델 40 결과: `/workspace/komap_model1/outputs/model107_oof_hard_negative/`
- 모델 41 test mask: `/workspace/komap_model1/outputs/model108_advanced_d4_gaussian/`

## 모델 42~49: encoder 구조화 및 공정 백본 재탐색

모델 42~46은 모델 31 checkpoint를 이어받았고, 모델 47~49는 ImageNet pretrained weight에서
새로 시작했다. 모두 native patch + D4 TTA + Gaussian blending 조건이다.

| 새 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 42 | residual ASPP bottleneck | **79.48%** | 75.83% | 보관 |
| 43 | high-resolution side branch | 79.34% | 75.73% | 미채택 |
| 44 | gated encoder adapters | 79.38% | 75.66% | 미채택 |
| 45 | encoder BN freeze | 79.26% | 75.65% | 미채택 |
| 46 | BN freeze + decoder GroupNorm | 78.80% | 75.59% | 미채택 |
| 47 | ConvNeXt-Tiny native patch | 73.70% | 69.45% | 미채택 |
| 48 | HRNet-W18 native patch | 74.88% | 70.65% | 미채택 |
| 49 | EfficientNet-B4 native patch | 75.27% | 71.18% | 미채택 |

결과적으로 ASPP만 소폭 유효했고, high-resolution·gated adapter·BN/GN은 개선되지 않았다.
과거 square-resize 조건의 백본 점수는 공정 비교가 아니며, native patch로 재실행해도
ResNet34 계열을 넘지 못했다.

원격 경로:

- 모델 42: `/workspace/komap_model1/outputs/model109_aspp_native/`
- 모델 43: `/workspace/komap_model1/outputs/model110_highres_structure/`
- 모델 44: `/workspace/komap_model1/outputs/model111_gated_encoder/`
- 모델 45: `/workspace/komap_model1/outputs/model112_bnfreeze/`
- 모델 46: `/workspace/komap_model1/outputs/model113_bngn/`
- 모델 47: `/workspace/komap_model1/outputs/model114_convnext_native/`
- 모델 48: `/workspace/komap_model1/outputs/model115_hrnet_native/`
- 모델 49: `/workspace/komap_model1/outputs/model116_effb4_native/`

## 모델 50: 최종 ASPP 앙상블

모델 31·32·34 앙상블에 모델 42를 추가했다.

| 조합 | ASPP weight | bias (Primary, Eutectic, Al) | Global mIoU | Image-wise 평균 |
|---|---:|---|---:|---:|
| 모델 50 validation best | 0.30 | (0.0, +0.1, -0.3) | **79.48%** | 75.73% |
| 모델 50 robust | 0.20 | (0.0, +0.1, -0.3) | 79.48% | 75.72% |

validation 최고와 robust 조합의 차이가 작아 실제 제출에는 robust 조합을 우선 권장한다.

- 전체 결과: `/workspace/komap_model1/outputs/model117_ensemble_aspp/results.json`
- validation 최고 test mask: `/workspace/komap_model1/outputs/model117_ensemble_aspp/test_masks/`
- robust weight=0.20 test mask: `/workspace/komap_model1/outputs/model117_ensemble_aspp/test_masks_robust_w020/`

## 팀원 제공 참고 결과

| 구성 | 점수 | 기준 |
|---|---:|---|
| EfficientNet-B4 U-Net | **77.00%** | 미확인 |

원문 기록: `EffNet-B4 U-Net :77%`
