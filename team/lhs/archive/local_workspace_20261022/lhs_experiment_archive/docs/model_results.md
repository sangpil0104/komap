# KoMaP 모델 결과 기록

## 현재 진행 상태 (2026-09-29)

- 대회 공식 목표 지표는 **image-wise 4-class mIoU, empty-union=0**이다. 모든 모델 순위와 80% 달성 판정은 이 규칙을 따른다. NaN-skip 및 pooled global mIoU는 참고 비교값이다.
- Model339의 네 gamma 동시 조정 400-route search 완료. Raw official strict 최고는 `79.3683%`(Model338 대비 `+0.0283%p`); NaN-skip `80.2062%`, pooled global `82.0064%`다. Model338 고정 route와 paired group-CV 변화는 `−0.0139%p` (95% CI `[-0.0613,+0.0352]`, 9/20 positive)라 채택 근거가 없어 Model338을 보수적 reference로 유지한다.
- Model335의 2,401-route exhaustive search는 이전 최고 기록이다. Model270 대비 grouped held-out route 선택은 strict `+0.4564%p`(14/20, CI `[+0.0456,+0.8978]`), NaN-skip `+0.5060%p`(15/20, CI `[+0.0524,+1.0264]`)였다.
- Model328의 4-fold width4000+uniform B4 OOF 완료. 단독 valid 점수는 empty=0 `79.2622%`, NaN-skip `79.9793%`다.
- Model329의 width4000+uniform+full-image-stat B4 grouped OOF에서 fold1–2의 36/70장 interim paired OOF는 strict empty=0 기준 Model322 대비 `+1.0898%p`(30 groups, 95% CI `[+0.4014,+1.8531]`, 21/36장 개선)다. Eutectic Si와 Al3Ni가 개선을 주도한다. 현재 fold3 학습 중이며 최종 OOF·valid fusion 전의 잠정 결과다.
- Model333의 Lovasz/control paired OOF는 Model329 큐와 robust-80 gate가 끝난 뒤 조건부 실행 대기 중이다. HRNet 기반 Model336은 Model333 이후 P100 큐에 있다.

## 번호 체계

기존 기록의 `7.2`, `8.1`, `97`, `109`처럼 불연속적이거나 중복되는 번호를 정리했다.
아래의 **새 모델 번호**는 실제 실행한 모델·개선·앙상블 단위를 기준으로 1부터 연속 부여했다.
기존 번호는 재현을 위해 `구 번호`로 함께 보존했다.

- 팀원이 제공한 `EffNet-B4 U-Net: 77%`는 점수 기준이 확인되지 않아 모델 번호에서 제외했다.
- 단순 test mask 생성·추론 비교도 실행 결과 보존을 위해 모델 번호를 부여했다.
- `test.zip`에는 정답 mask가 없어 test 점수는 계산하지 못하고 예측 mask만 생성했다.

## 공식 클래스 매핑 교정

공식 범례와 mask RGB palette를 대조한 결과, 기존 기록의 클래스명 순서가 RGB 채널
순서와 어긋나 있었다. 체크포인트의 channel ID와 RGB 색상은 유지하고 이름만 공식
의미에 맞춰 교정했다.

| 체크포인트 channel ID | RGB 색상 | 공식 클래스 |
|---:|---|---|
| 0 | 파란색 `(25, 76, 153)` | Al3Ni |
| 1 | 초록색 `(76, 178, 76)` | Primary Si |
| 2 | 갈색 `(153, 127, 76)` | Al |
| 3 | 회색 `(204, 204, 204)` | Eutectic Si |

전체 mIoU와 image-wise mIoU는 클래스 permutation에 불변이지만, 기존에 표시했던
클래스별 IoU와 Al 중심 개선 실험의 의미는 잘못되어 있었다. 이후 결과는 공식 클래스명을
기준으로 기록한다. 기존 checkpoint를 그대로 활용할 때는 RGB palette를 재배열하지 않는다.

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
| 51 | 118 | RGB + grayscale/local contrast/Sobel/LoG input stem | 79.07% |
| 52 | 119 | ASPP + residual scSE skip attention | 79.48% |
| 53 | 120 | Al signed-distance auxiliary head | 79.23% |
| 54 | 121 | raw-input stride-2 high-resolution skip | 79.21% |
| 55 | 122 | late Lovász-Softmax phase | **79.49%** |
| 56 | 123 | EMA weight averaging | 79.47% |
| 57 | 124 | model 50 + refined EMA ensemble | **79.50%** |

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

- 모델 22 checkpoint: `$KOMAP_WORKSPACE/outputs/model62_ft_weighted_safeaug/best.pt`
- 모델 22 test mask: `$KOMAP_WORKSPACE/outputs/model62_ft_weighted_safeaug/test_masks/`

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

- 모델 23: `$KOMAP_WORKSPACE/outputs/model91_micronet_resnet34/`
- 모델 24: `$KOMAP_WORKSPACE/outputs/model92_micronet_effb4/`
- 모델 25: `$KOMAP_WORKSPACE/outputs/model93_staged_model6/`
- 모델 26: `$KOMAP_WORKSPACE/outputs/model94_focal_dice_model6/`
- 모델 27: `$KOMAP_WORKSPACE/outputs/model95_boundary_lovasz_model6/`
- 모델 28: `$KOMAP_WORKSPACE/outputs/model96_4fold_ensemble/`

## 모델 29~35: native-resolution patch 및 Al 개선

모델 29부터는 원본 가로세로비를 유지한 512px patch, overlap sliding-window, Primary Si/Eutectic Si
중심 crop sampling을 사용했다. 모델 29의 클래스별 TTA IoU는 Primary Si 87.05%, Al3Ni
75.32%, Eutectic Si 56.13%, Al 92.90%였다.

| 새 모델 | 변경 내용 | Global mIoU | Image-wise 평균 |
|---:|---|---:|---:|
| 29 | native patch baseline | 77.85% | 74.46% |
| 30 | 모델 29 + Eutectic Si bias -0.60 | 78.54% | 75.09% |
| 31 | hard-negative 560 center fine-tuning | **79.17%** | 75.11% |
| 32 | boundary emphasis fine-tuning | 79.16% | **75.31%** |
| 33 | 모델 31/32 2-way ensemble | 79.36% | 75.51% |
| 34 | 모델 31 기반 640px patch fine-tuning | 79.20% | 75.75% |
| 35 | 모델 31/32/34 3-way ensemble | **79.42%** | 75.68% |

주요 클래스 IoU:

- 모델 31: Primary 87.74%, Al3Ni 77.91%, Eutectic 57.33%, Al 93.70%
- 모델 32: Primary 87.63%, Al3Ni 77.46%, Eutectic 57.98%, Al 93.56%
- 모델 35: Primary 87.80%, Al3Ni 77.71%, Eutectic 58.34%, Al 93.81%

원격 경로:

- 모델 29: `$KOMAP_WORKSPACE/outputs/model97_native_patch/`
- 모델 31: `$KOMAP_WORKSPACE/outputs/model98_hard_negative/`
- 모델 32: `$KOMAP_WORKSPACE/outputs/model99_boundary_native/`
- 모델 33: `$KOMAP_WORKSPACE/outputs/model100_ensemble/`
- 모델 34: `$KOMAP_WORKSPACE/outputs/model101_highres640/`
- 모델 35: `$KOMAP_WORKSPACE/outputs/model102_three_ensemble/`

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

- 모델 36 결과: `$KOMAP_WORKSPACE/outputs/model103_advanced_inference/results.json`
- 모델 37 결과: `$KOMAP_WORKSPACE/outputs/model104_group_clean_calibration/results.json`
- 모델 38 결과: `$KOMAP_WORKSPACE/outputs/model105_al_refiner/results.json`
- 모델 39 결과: `$KOMAP_WORKSPACE/outputs/model106_soft_boundary_clahe/`
- 모델 40 결과: `$KOMAP_WORKSPACE/outputs/model107_oof_hard_negative/`
- 모델 41 test mask: `$KOMAP_WORKSPACE/outputs/model108_advanced_d4_gaussian/`

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

- 모델 42: `$KOMAP_WORKSPACE/outputs/model109_aspp_native/`
- 모델 43: `$KOMAP_WORKSPACE/outputs/model110_highres_structure/`
- 모델 44: `$KOMAP_WORKSPACE/outputs/model111_gated_encoder/`
- 모델 45: `$KOMAP_WORKSPACE/outputs/model112_bnfreeze/`
- 모델 46: `$KOMAP_WORKSPACE/outputs/model113_bngn/`
- 모델 47: `$KOMAP_WORKSPACE/outputs/model114_convnext_native/`
- 모델 48: `$KOMAP_WORKSPACE/outputs/model115_hrnet_native/`
- 모델 49: `$KOMAP_WORKSPACE/outputs/model116_effb4_native/`

## 모델 50: 최종 ASPP 앙상블

모델 31·32·34 앙상블에 모델 42를 추가했다.

| 조합 | ASPP weight | bias (Primary, Eutectic, Al) | Global mIoU | Image-wise 평균 |
|---|---:|---|---:|---:|
| 모델 50 validation best | 0.30 | (0.0, +0.1, -0.3) | **79.48%** | 75.73% |
| 모델 50 robust | 0.20 | (0.0, +0.1, -0.3) | 79.48% | 75.72% |

validation 최고와 robust 조합의 차이가 작아 실제 제출에는 robust 조합을 우선 권장한다.

- 전체 결과: `$KOMAP_WORKSPACE/outputs/model117_ensemble_aspp/results.json`
- validation 최고 test mask: `$KOMAP_WORKSPACE/outputs/model117_ensemble_aspp/test_masks/`
- robust weight=0.20 test mask: `$KOMAP_WORKSPACE/outputs/model117_ensemble_aspp/test_masks_robust_w020/`

## 모델 51~57: ResNet34 입력·구조·Eutectic Si 경계 개선

Claude Code의 우선순위인 입력 filter stem, skip attention, signed-distance 보조 task,
고해상도 skip, late Lovász, EMA를 model 42(구 model109)에서 순차 실험했다.

| 새 모델 | 구 번호 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---:|---|---:|---:|---|
| 51 | 118 | RGB + grayscale/local contrast/Sobel/LoG 8채널 stem | 79.07% | 75.44% | 미채택 |
| 52 | 119 | residual scSE skip attention | 79.48% | 75.86% | 앙상블 후보 |
| 53 | 120 | Eutectic Si signed-distance auxiliary head | 79.23% | 75.62% | 미채택 |
| 54 | 121 | raw-input stride-2 high-resolution skip | 79.21% | 75.59% | 미채택 |
| 55 | 122 | CE+Dice warm-up 후 Lovász-Softmax | **79.49%** | 75.83% | 보관 |
| 56 | 123 | EMA decay 0.995 | 79.47% | 75.82% | 보관 |
| 57 | 124 | 모델 50 + EMA 25% | **79.50%** | 75.75% | 기존 global 최고 |
| 58 | mapping-corrected | 공식 클래스 매핑 교정 + neutral bias + 모델 57 계열 앙상블 | 79.32% | **75.89%** | 현재 image-wise 최고 |
| 59 | 125 | Eutectic Si false-negative hard-center fine-tuning | 74.59% | 74.59% | 단일 모델, 미채택 |

세부 결과:

- 모델 51: 추가 입력 표현만으로는 RGB pretrained 표현보다 낮았다.
- 모델 52: Eutectic Si IoU 58.54%로 기준과 동률이며, 작은 구조적 다양성이 있어 앙상블 후보로 남겼다.
- 모델 53: Eutectic Si IoU 58.03%로 signed-distance 보조 supervision의 효과가 제한적이었다.
- 모델 54: raw-input high-resolution skip은 기존 ASPP보다 낮았다.
- 모델 55: Primary 88.20%, Al3Ni 77.53%, Eutectic 58.53%, Al 93.71%였다.
- 모델 57은 기존 robust bias `(Primary 0.0, Eutectic +0.1, Al -0.3)`를 유지하고 EMA 모델을
  25% 섞은 결과다. 기존 model50 대비 약 0.02%p 상승이라 큰 개선으로 해석하지 않는다.

공식 매핑 교정 후 모델 58은 클래스 bias를 모두 0으로 둔 모델 57 계열 앙상블이며,
image-wise mIoU 75.89%를 기록했다. Eutectic Si false-negative 중심으로 재학습한 모델 59는
단일 모델 74.59%였고, 모델 58 앙상블에 섞었을 때 상승하지 않아 채택하지 않았다.

원격 결과 경로:

- 모델 51: `$KOMAP_WORKSPACE/outputs/model118_input_filters/`
- 모델 52: `$KOMAP_WORKSPACE/outputs/model119_scse/`
- 모델 53: `$KOMAP_WORKSPACE/outputs/model120_sdf_al/`
- 모델 54: `$KOMAP_WORKSPACE/outputs/model121_raw_highres/`
- 모델 55: `$KOMAP_WORKSPACE/outputs/model122_lovasz_late/`
- 모델 56: `$KOMAP_WORKSPACE/outputs/model123_ema/`
- 모델 57 결과: `$KOMAP_WORKSPACE/outputs/model124_refined_ensemble/results.json`
- 모델 57 test mask: `$KOMAP_WORKSPACE/outputs/model124_refined_ensemble/test_masks/`
- 모델 58 valid audit: `$KOMAP_WORKSPACE/outputs/model125_corrected_ensemble_valid/`
- 모델 59 checkpoint: `$KOMAP_WORKSPACE/outputs/model125_corrected_eutectic_hard/best.pt`

## 팀원 제공 참고 결과

| 구성 | 점수 | 기준 |
|---|---:|---|
| EfficientNet-B4 U-Net | **77.00%** | 미확인 |

원문 기록: `EffNet-B4 U-Net :77%`

## 성능개선 재시작 트랙 (공식 image-wise mean mIoU)

기존 실험은 global confusion-matrix mIoU와 image-wise 평균이 섞여 있어 직접 비교가
어려웠다. 이 트랙부터는 대회 설명에 맞춰 **각 valid 이미지에서 4개 class IoU를 평균한 뒤,
20개 이미지 평균**을 checkpoint 선택 및 최종 비교 기준으로 고정한다. 내부 채널 순서는
`[Al3Ni, Primary Si, Al, Eutectic Si]`이며, 공식 색상 기준도 함께 교정했다.

| 새 모델 | 구성 | Global mIoU | Image-wise 평균 | 공식 class IoU (Primary / Al3Ni / Eutectic / Al) |
|---:|---|---:|---:|---|
| 60 | ResNet34 U-Net fresh baseline, square 512 | 73.42% | 71.32% | 85.56 / 68.05 / 50.09 / 89.96 |
| 61 | ResNet34 U-Net++ fresh baseline, square 512 | 74.38% | 72.29% | 86.59 / 68.91 / 51.81 / 90.21 |
| 62 | 모델 61 + native patch, overlap inference, Eutectic/Primary sampling, TTA | 77.73% | **74.44%** | 86.85 / 75.69 / 55.35 / 93.05 |

모델 62의 최종 TTA는 재시작 트랙의 native 개선 기준선이다. 기존 mapping-corrected ensemble의
75.89%와는 평가 파이프라인이 다르므로, 후속 모델에서 동일한 native/TTA 파이프라인으로
앙상블과 후처리를 재검증했다.

원격 결과 경로:

- 모델 60: `$KOMAP_WORKSPACE/outputs/restart01_resnet34_baseline/`
- 모델 61: `$KOMAP_WORKSPACE/outputs/restart02_unetpp_baseline/`
- 모델 62: `$KOMAP_WORKSPACE/outputs/restart03_native_eutectic/`

### 후속 개선 (모델 63~67)

| 새 모델 | 구성 | Global mIoU | Image-wise 평균 | 공식 class IoU (Primary / Al3Ni / Eutectic / Al) | 판정 |
|---:|---|---:|---:|---|---|
| 63 | 모델 62 + Eutectic false-negative hard-center fine-tuning | 78.75% | 75.43% | 87.41 / 77.02 / 57.07 / 93.50 | 채택 |
| 64 | 모델 62/63 native 확률 앙상블 coarse search | 78.75% | 75.43% | 87.41 / 77.02 / 57.07 / 93.50 | 앙상블 이득 없음 |
| 65 | 모델 63 + boundary loss fine-tuning | 78.16% | 75.09% | 87.25 / 75.87 / 56.42 / 93.08 | 미채택 |
| 66 | 모델 63 + Eutectic logit bias sweep, bias=-0.30 | 79.03% | **75.55%** | 87.49 / 77.47 / 57.69 / 93.60 | 현재 재시작 최고 |
| 67 | 모델 66 bias extended sweep, image-wise 기준 bias=-0.35 | 79.06% | **75.55%** | 87.49 / 77.47 / 57.69 / 93.60 | 66과 사실상 동일 |

모델 63에서 hard-center fine-tuning은 Eutectic Si false-negative를 줄여 상승했지만,
경계 loss는 오히려 하락했다. 앙상블 coarse search도 새 hard-negative 모델 단독보다 낮았고,
최종적으로 Eutectic logit을 약간 억제하는 calibration이 image-wise 평균을 75.55%까지
끌어올렸다. extended sweep에서 -0.35와 -0.30의 차이는 0.000001 mIoU 미만으로, 제출용
mask는 보수적인 -0.35를 사용했다.

원격 결과 경로:

- 모델 63: `$KOMAP_WORKSPACE/outputs/restart04_eutectic_hard/`
- 모델 64: `$KOMAP_WORKSPACE/outputs/restart05_native_ensemble/`
- 모델 65: `$KOMAP_WORKSPACE/outputs/restart06_boundary/`
- 모델 66: `$KOMAP_WORKSPACE/outputs/restart07_eutectic_calibration.json`
- 모델 67: `$KOMAP_WORKSPACE/outputs/restart07b_eutectic_calibration_extended.json`
- 제출 후보 test masks: `$KOMAP_WORKSPACE/outputs/restart07_eutectic_calibrated/test_masks/`

## 성능개선 재시작 트랙 후속 (모델 68~103)

모델 68 이후도 동일한 내부 채널 순서 `[Al3Ni, Primary Si, Al, Eutectic Si]`와
validation 20장 기준을 사용했다. 표의 `Image-wise 평균`은 이미지별 4-class IoU 평균의
평균이며, `Global mIoU`는 전체 validation confusion matrix 기준이다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 68 | native 640px 모델 raw/calibration | 79.26% | 75.85% | 보관 |
| 69 | native 640px Eutectic hard-negative | 79.20% | 75.72% | 미채택 |
| 70 | native 640px seed 123 | 79.20% | 75.71% | 미채택 |
| 71 | native 640px seed ensemble | 79.26% | 75.85% | 단독 우세 없음 |
| 72 | native 640px low-lr continuation | 79.24% | 75.82% | 미채택 |
| 73 | native 640px + 512px mixed ensemble | 79.28% | 75.95% | 보관 |
| 74 | mixed ensemble fine search | 79.36% | 75.97% | 보관 |
| 75 | native 640px more-crop inference | 79.20% | 75.73% | 미채택 |
| 76 | D4/gaussian single-model inference | - | 75.91% | 미채택 |
| 77 | train-only grayscale intensity prior | 79.18% | 76.02% | 보관 |
| 78 | seed 123 mixed ensemble | 79.20% | 75.92% | 미채택 |
| 79 | Al3Ni-weighted 512px hard-negative | 79.20% | 75.85% | 앙상블 후보 |
| 80 | highres + Al3Ni mixed ensemble | 79.42% | 76.09% | 보관 |
| 81 | Al3Ni mixed + intensity prior | 79.34% | 76.19% | 보관 |
| 82 | joint intensity search | 79.47% | **76.28%** | 기준 후보 |
| 83 | patch 768 inference | - | 75.85% | 미채택 |
| 84 | current joint candidate masks | 79.47% | 76.28% | 제출 후보 |
| 85 | valid mask-level old ensemble search | 79.47% | 76.28% | 이득 없음 |
| 86 | probability gaussian smoothing | 79.47% | 76.28% | 이득 없음 |
| 87 | 4-input Z-channel fine-tune | - | 74.55% | 미채택 |
| 88 | current 3-model + ASPP probability ensemble | 79.64% | 76.36% | 보관 |
| 89 | ASPP + class-conditional phase weights | 79.66% | **76.39%** | 보관 |
| 90 | ASPP bottleneck fine-tune from Al3Ni model | 78.95% | 75.51% | 미채택 |
| 91 | fine-tuned ASPP fourth-member addition | 79.66% | 76.39% | 추가 이득 없음 |
| 92 | global-oriented class-conditional search | 79.73% | 76.39% | global 후보 |
| 93 | global calibration refine | 79.77% | 76.30% | global 후보 |
| 94 | 0.05-step fine class-weight search | 79.82% | 76.23% | global 후보 |
| 95 | narrow calibration fine search | 79.89% | 76.28% | global 후보 |
| 96 | expanded calibration search | **79.92%** | 76.25% | global 최고 |
| 97 | Primary Si small-component cleanup (128px) | 79.78% | **76.51%** | image-wise 최고 |
| 98 | dense overlap stride 256 | 79.88% | 76.37% | 미채택 |
| 99 | native 640px Al3Ni-weighted hard-negative 단일 | - | 75.72% | 미채택 |
| 100 | 640px Al3Ni-weighted fourth-member, 10% | 79.71% | **76.43%** | 보조 후보 |
| 101 | teacher ensemble → ResNet34 U-Net++ distillation | - | 75.43% | 미채택 |
| 102 | ResNet50-U-Net native 4 epoch | 66.66% | 62.91% | 미채택 |
| 103 | ResNet50-U-Net continuation 8 epoch | 71.36% | 67.64% | 미채택 |

### 모델 68~103 원격 결과 경로

- 모델 68~87: `$KOMAP_WORKSPACE/outputs/restart08_highres640/` ~ `$KOMAP_WORKSPACE/outputs/restart27_zchannel/`
- 모델 88: `$KOMAP_WORKSPACE/outputs/restart28_current_aspp_ensemble/`
- 모델 89: `$KOMAP_WORKSPACE/outputs/restart29_class_conditional_aspp/`
- 모델 90: `$KOMAP_WORKSPACE/outputs/restart30_aspp_from_al3ni/`
- 모델 91: `$KOMAP_WORKSPACE/outputs/restart31_finetuned_aspp_addition/`
- 모델 92: `$KOMAP_WORKSPACE/outputs/restart32_global_class_search/`
- 모델 93: `$KOMAP_WORKSPACE/outputs/restart33_global_calibration_refine/`
- 모델 94: `$KOMAP_WORKSPACE/outputs/restart34_fine_class_weight/`
- 모델 95: `$KOMAP_WORKSPACE/outputs/restart35_final_calibration/`
- 모델 96: `$KOMAP_WORKSPACE/outputs/restart36_expanded_calibration/`
- 모델 97: `$KOMAP_WORKSPACE/outputs/restart37_morphology/`
- 모델 98: `$KOMAP_WORKSPACE/outputs/restart38_dense_overlap/`
- 모델 99: `$KOMAP_WORKSPACE/outputs/restart39_highres_al3ni_weighted/`
- 모델 100: `$KOMAP_WORKSPACE/outputs/restart40_highres_al3ni_addition/`
- 모델 101: `$KOMAP_WORKSPACE/outputs/restart41_distill_student/`
- 모델 102: `$KOMAP_WORKSPACE/outputs/restart42_resnet50_native/`
- 모델 103: `$KOMAP_WORKSPACE/outputs/restart43_resnet50_native_continue/`

| 104 | global 최고 조합 + Primary Si 128px connected-component cleanup | **80.03%** | 76.37% | global 80% 달성 |

- 모델 104 결과: `$KOMAP_WORKSPACE/outputs/restart44_global_morphology/`
- 모델 104 test masks: `$KOMAP_WORKSPACE/outputs/restart44_global_morphology/test_masks/`

모델 104는 global mIoU 목표 80%를 처음 넘겼지만, image-wise 평균은 80%가 아니다.
대회 평가가 이미지별 평균 방식이면 모델 112의 image-wise 후보(76.72%)를 우선 검토하고,
global 기준 제출 후보는 모델 104와 모델 112를 함께 비교한다. 두 기준이 다르므로 실제
제출 전에는 대회 평가 코드의 absent-class IoU 처리까지 확인해 하나를 선택해야 한다.

## 모델 105~114: 깊은 백본·도메인 적응·768 문맥 후속 실험

모델 105 이후에도 validation image-wise mean mIoU를 주 선택 기준으로 유지했다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 105 | ResNet34 Al3Ni hard-negative 전문 모델 | 74.93% | 74.93% | 미채택 |
| 106 | 모델 89 앙상블 + Al3Ni 전문 모델 | 79.67% | 76.39% | 추가 이득 없음 |
| 107 | image entropy routing 진단 | - | 75.95% | 미채택; oracle 상한 76.88% |
| 108 | ResNet34 ASPP + scSE fourth member | 79.79% | 76.51% | 추가 이득 없음 |
| 109 | ResNet50 U-Net++ native 12 epoch | 73.73% | 70.00% | 미채택 |
| 110 | AdaBN label-free domain adaptation | 77.12% | 73.54% | 미채택 |
| 111 | ASPP 768px native-context fine-tune 단일 | 79.82% | 76.38% | 앙상블 후보 |
| 112 | 모델 89 + ASPP 768px, weight 0.60, Primary cleanup 128 | **79.97%** | **76.72%** | 현재 image-wise 최고 |
| 113 | ASPP 768px + Al3Ni class weight 2.20 | 79.04% | 75.97% | 미채택 |
| 114 | ASPP 768px class별 weight coordinate search | 79.97% | 76.72% | 모델 112과 동일 |

후속 실험 결과는 모두 `$KOMAP_WORKSPACE/outputs/restart45_*` ~
`$KOMAP_WORKSPACE/outputs/restart54_*`에 보관했다. 현재 image-wise 제출 후보는
모델 112이며 test masks는 다음 경로다.

- `$KOMAP_WORKSPACE/outputs/restart52_aspp768_addition/test_masks/`

768px 모델은 작은 patch의 문맥 손실을 줄여 기존 최고를 76.51%에서 76.72%로 올렸지만,
80%까지는 아직 3.28%p가 남아 있다. ResNet50 심화, AdaBN, Al3Ni 재가중, scSE는
 현재 앙상블의 병목을 해결하지 못했다.

## 모델 115~116: wide-context 및 768 calibration 후속

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 115 | 512×768 직사각 native patch fine-tune | 79.23% | 75.43% | 미채택 |
| 116 | 모델 112 + intensity prior alpha 0.12 + Al bias +0.40 | **80.06%** | **76.75%** | 현재 최고 |

모델 116 test masks:
`$KOMAP_WORKSPACE/outputs/restart56_calibration_search/test_masks/`
## 모델 123: intensity-domain style TTA
입력이 사실상 grayscale인 metallography 데이터의 intensity domain shift를 완화하기 위해 train 전체 intensity mean/std로 변환한 high-resolution 768px branch를 원래 branch와 섞었다. 기존 old/new/ASPP source는 유지하고 high branch만 변환했으며 validation에서 style weight와 prior/bias를 탐색했다.
| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 123 | high raw/style blend, style weight 0.75, alpha 0.09, Al +0.40, Eutectic -0.50 | 80.04% | **77.0055%** | 현재 최고 |
기존 모델 116의 76.74796% 대비 0.2576%p, 모델 120의 76.75140% 대비 0.2541%p 상승했다. percentile contrast normalization은 최고 76.7826%로 표준화보다 낮았다.
원격 결과: `$KOMAP_WORKSPACE/outputs/restart67_style_tta_fine/`, test masks: `$KOMAP_WORKSPACE/outputs/restart68_style_best/test_masks/`

## 모델 117~120: Al3Ni–Eutectic Si 쌍별 margin 및 추가 앙상블

모델 116의 이미지별 audit에서 낮은 점수의 핵심 오류가 Al3Ni(channel 0)와
Eutectic Si(channel 3)의 상호 혼동으로 나타나, 두 class의 logit margin을 직접
확대하는 pairwise 보조 loss를 실험했다. 768px micro-batch가 1이어야 했으므로
fine-tuning 중 BatchNorm running-stat은 고정했고, 기존 CE+Dice loss에
`0.10 * ReLU(0.20 - (target_logit - pair_logit))`를 추가했다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 117 | ASPP 768px seed 456 fine-tune | 80.04% | 76.17% | 미채택 |
| 118 | 모델 116 ensemble + seed456 source search | 80.06% | 76.7480% | seed weight 0, 추가 이득 없음 |
| 119 | Al3Ni–Eutectic pairwise margin fine-tune, best epoch 1 | 79.39% | 76.28% | 단일 모델 미채택 |
| 120 | 모델 116 ensemble + pairwise source, high=0.60/pair=0.05 | 80.0511% | **76.7514%** | 미세 상승, 보관 |

모델 120은 모델 116보다 image-wise 기준 `+0.0034%p`로 사실상 동률이며,
pairwise 모델 자체의 분리 학습만으로는 병목을 해결하지 못했다.

원격 결과 경로:

- 모델 117: `$KOMAP_WORKSPACE/outputs/restart59_aspp768_seed456/`
- 모델 118: `$KOMAP_WORKSPACE/outputs/restart60_seed456_addition/`
- 모델 119: `$KOMAP_WORKSPACE/outputs/restart61_pairwise_margin/`
- 모델 120 validation search: `$KOMAP_WORKSPACE/outputs/restart62_pairwise_addition/`
- 모델 120 test masks: `$KOMAP_WORKSPACE/outputs/restart63_pairwise_best/test_masks/`

## 모델 121~122: U-Net++ deep supervision 재검증

현재 ASPP U-Net++ 구현의 중간 decoder feature에 auxiliary 1×1 head를 추가하고,
최종 head와 함께 CE+Dice를 학습했다. 768px 배치 1의 BN 문제를 피하기 위해 이
실험은 512px 배치 2에서 진행했다.

| 모델 | 구성 | Image-wise 평균 | 판정 |
|---:|---|---:|---|
| 121 | auxiliary loss 0.20/0.30, 즉시 적용 | epoch1 75.25%, epoch2 74.94% | 미채택 |
| 122 | 본체 2 epoch warm-up + auxiliary loss 0.05/0.05 | epoch1 75.32% | 미채택 |

auxiliary head가 random initialization인 상태에서 본체 checkpoint의 표현을
개선하지 못했고, warm-up·소형 weight로도 기준을 회복하지 못했다.

원격 결과 경로:

- 모델 121: `$KOMAP_WORKSPACE/outputs/restart64_deep_supervision/`
- 모델 122: `$KOMAP_WORKSPACE/outputs/restart65_deep_supervision_warmup/`

## 모델 123: intensity-domain style TTA

입력이 사실상 grayscale인 metallography 데이터의 intensity domain shift를 완화하기 위해 train 전체 intensity mean/std로 변환한 high-resolution 768px branch를 원래 branch와 섞었다. 기존 old/new/ASPP source는 유지하고 high branch만 변환했으며 validation에서 style weight와 prior/bias를 탐색했다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 123 | high raw/style blend, style weight 0.75, alpha 0.09, Al +0.40, Eutectic -0.50 | 80.04% | **77.0055%** | 당시 최고 |

기존 모델 116의 76.74796% 대비 0.2576%p 상승했다. percentile contrast normalization은 최고 76.7826%로 표준화보다 낮았다.

## 모델 124~126: intensity route 및 pairwise source 결합

모델 123의 표준화 style branch가 일부 이미지에서는 손해를 주므로, 원본 이미지의 grayscale 표준편차가 threshold 이상일 때만 style branch를 사용하는 label-free route를 추가했다. 이어 pairwise margin source를 함께 탐색했다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 124 | 512px ASPP에도 style 적용, high ensemble 0.60 고정 | 80.04% | 77.0055% | 추가 이득 없음 |
| 125 | std threshold 50 route, high style blend 0.75 | 80.29% | **77.0679%** | 보관 |
| 126 | 모델 125 + pairwise source 0.08, high 0.60 | **80.27%** | **77.0779%** | 현재 최고 |

모델 126은 모델 123보다 0.0724%p, 모델 116보다 0.3299%p 상승했다. 다만 validation 20장에 기반한 route threshold이므로 test 제출 시에는 보수적인 모델 123과 함께 보관한다.

원격 결과:

- 모델 123 validation: `$KOMAP_WORKSPACE/outputs/restart67_style_tta_fine/`
- 모델 123 test masks: `$KOMAP_WORKSPACE/outputs/restart68_style_best/test_masks/`
- 모델 124: `$KOMAP_WORKSPACE/outputs/restart70_multi_style_correct/`
- 모델 125 validation route: `$KOMAP_WORKSPACE/outputs/restart71_style_route/`
- 모델 125 test masks: `$KOMAP_WORKSPACE/outputs/restart72_style_route_best/test_masks/`
- 모델 126 validation search: `$KOMAP_WORKSPACE/outputs/restart73_style_pair_route/`
- 모델 126 test masks: `$KOMAP_WORKSPACE/outputs/restart74_style_pair_route_best/test_masks/`

## 모델 127: label-free TENT domain adaptation + intensity route

고해상도 ASPP branch에 대해 각 validation/test 이미지의 네 모서리 tile만 사용해
label-free entropy minimization을 2 step 수행했다. 학습되는 값은 BatchNorm의
affine parameter만이며, segmentation label은 사용하지 않았다. 적응된 branch는
grayscale 표준화 입력과 원본 입력을 함께 추론하고, 이미지 표준편차 50 이상일 때
두 결과를 25:75로 혼합했다. 이후 기존 old/new/ASPP source와 결합하고 intensity
prior 및 Al/Eutectic 보정을 적용했다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 127 | 768px ASPP + per-image TENT 2 step, lr=1e-5, std route=50 | **80.2362%** | **77.1897%** | 현재 최고 |

모델 126 대비 image-wise 기준 `+0.1118%p`, 모델 116 대비 `+0.4417%p` 상승했다.
TENT는 valid label을 사용하지 않는 test-time adaptation이므로, test mask에도 같은
절차를 적용해 보관했다.

원격 결과:

- 모델 127 validation: `$KOMAP_WORKSPACE/outputs/restart75_tent_style/`
- 모델 127 test masks: `$KOMAP_WORKSPACE/outputs/restart76_tent_style_best/test_masks/`

## 모델 128: TENT + pairwise-margin source

모델 127의 per-image TENT 고해상도 branch에 Al3Ni–Eutectic Si pairwise-margin
checkpoint를 추가했다. high branch 60%, pairwise source 12%, 기존 3개 source
28%의 비율로 결합하고, 모델 127과 동일한 std=50 style route 및 calibration을
사용했다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 128 | 모델 127 + pairwise source 12% | 80.2149% | **77.2050%** | 현재 최고 |

모델 127 대비 image-wise 기준 `+0.0153%p`로 추가 개선은 작지만 양수였다.

원격 결과:

- 모델 128 validation: `$KOMAP_WORKSPACE/outputs/restart77_tent_pair_route/`
- 모델 128 test masks: `$KOMAP_WORKSPACE/outputs/restart78_tent_pair_best/test_masks/`

## 모델 129: TENT 3-step + pairwise-margin source

모델 128의 TENT 적응을 2 step에서 3 step으로 늘려 비교했다. pairwise source
가중치는 10%, high branch는 60%, std=50 style route와 calibration은 동일하게
유지했다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 129 | TENT 3 step, lr=1e-5 + pairwise 10% | 80.1751% | **77.2281%** | 현재 최고 |

모델 128 대비 image-wise 기준 `+0.0231%p`, 모델 116 대비 `+0.4801%p` 상승했다.

원격 결과:

- 모델 129 validation: `$KOMAP_WORKSPACE/outputs/restart79_tent3_pair_route/`
- 모델 129 test masks: `$KOMAP_WORKSPACE/outputs/restart80_tent3_pair_best/test_masks/`

## 모델 130~133: TENT 강도 및 label-free route 후속

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 130 | TENT 4 step, lr=1e-5 + pairwise 8%, std route=50 | 80.1331% | 77.2223% | 3-step보다 낮아 미채택 |
| 131 | TENT 3 step, lr=2e-5 + pairwise 10%, std route=50 | 80.1754% | 77.2287% | 보관 |
| 132 | 모델 131 + Eutectic Si 2-pixel component cleanup | 80.1785% | **77.2304%** | 현재 최고 |
| 133 | entropy/confidence/area-prior 기반 style route 비교 | 80.1753% | 77.2287% | std route 유지 |

TENT를 4 step으로 늘리면 성능이 소폭 하락해 과적응 경향을 확인했다. 반면
학습률 2e-5의 3 step은 모델 129보다 `+0.0007%p` 높았다. 최종적으로 모델 131의
argmax 결과에서 Eutectic Si의 단일 pixel 연결요소만 주변 logits의 차선 class로
재할당하는 보수적 cleanup을 추가해 모델 132를 채택했다. 연결요소 기준을 크게
늘리면 Eutectic recall이 급락했으며, entropy·confidence·train class-area 기반의
완전 label-free route도 std=50 route를 넘지 못했다.

원격 결과:

- 모델 130 validation: `$KOMAP_WORKSPACE/outputs/restart81_tent4_pair_route/`
- 모델 131 validation: `$KOMAP_WORKSPACE/outputs/restart82_tent3_lr2e5_pair_route/`
- 모델 132 validation: `$KOMAP_WORKSPACE/outputs/restart83_tent_postprocess/`
- 모델 132 test masks: `$KOMAP_WORKSPACE/outputs/restart85_tent3_lr2e5_post2_best/test_masks/`
- 모델 133 validation: `$KOMAP_WORKSPACE/outputs/restart84_route_criterion/`

## 모델 134: raw/style 분리 TENT adapter

원본 입력과 intensity-standardized 입력에 각각 별도의 BN-affine TENT adapter를
적용해 domain별 통계를 분리했다. 그러나 raw adapter를 style branch에도 공유한
기존 방식보다 낮아, 단일 raw-domain adapter 방식을 유지했다.

| 모델 | 구성 | Image-wise 평균 | 판정 |
|---:|---|---:|---|
| 134 | raw/style 별도 TENT 3 step, lr=2e-5 + pairwise 10% | 77.2285% | 미채택 |

원격 결과: `$KOMAP_WORKSPACE/outputs/restart86_dual_domain/`

## 모델 135~137: Eutectic 조건부 calibration 및 추가 검증

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 135 | 모델 132 + image mean 기준 Eutectic bias 분기 | **80.2424%** | **77.3840%** | 현재 최고 |
| 136 | 모델 132 logits Gaussian smoothing (all/Eutectic, sigma 0.25~2.0) | 80.1752% | 77.2286% | smoothing 미채택 |
| 137 | raw/style consistency TENT 3 step (weight 0.0~1.0) | 79.5222% | 76.7925% | consistency 미채택 |

모델 135는 이미지 평균 intensity가 167.037 미만이면 Eutectic bias `-0.70`,
이상이면 `-0.30`을 사용했다. 이는 밝은 이미지에서 Eutectic을 과소예측하는
경향을 보정해 모델 132 대비 `+0.1536%p` 상승시켰다. 이 분기는 test에서
label을 사용하지 않고 원본 image mean만 사용한다.

원격 결과:

- 모델 135 validation: `$KOMAP_WORKSPACE/outputs/restart89_dynamic_bias/`
- 모델 135 test masks: `$KOMAP_WORKSPACE/outputs/restart90_dynamic_bias_best/test_masks/`
- 모델 136 validation: `$KOMAP_WORKSPACE/outputs/restart87_spatial_smoothing/`
- 모델 137 validation: `$KOMAP_WORKSPACE/outputs/restart88_tent_consistency/`

## 모델 138: mean-conditioned high-resolution routing

모델 135의 intensity-conditioned Eutectic bias를 유지하면서 high-resolution
branch 가중치도 이미지 평균 intensity에 따라 나눴다. mean<165인 이미지에는
high branch 70%, mean≥165인 이미지에는 60%를 사용하고, pairwise source 10% 및
기존 3개 source를 나머지에 배분했다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 138 | 모델 135 + mean-conditioned high weight 70/60% | **80.2601%** | **77.4315%** | 현재 최고 |

모델 135 대비 `+0.0474%p`, 모델 132 대비 `+0.2011%p` 상승했다.

원격 결과:

- 모델 138 validation: `$KOMAP_WORKSPACE/outputs/restart91_dynamic_weight/`
- 모델 138 test masks: `$KOMAP_WORKSPACE/outputs/restart92_dynamic_weight_best/test_masks/`

## 모델 139: mean-conditioned high routing + pairwise 15%

모델 138의 mean-conditioned high branch weight(평균 intensity 165 미만 70%,
이상 60%)에 pairwise-margin source를 15%로 올렸다. 두 intensity 구간 모두
pairwise 15%가 최적이어서 최종 조합은 old/new/ASPP source 15%, pairwise 15%,
high branch 60~70%이다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 139 | 모델 138 + pairwise source 15% | **80.2682%** | **77.4506%** | 현재 최고 |

모델 138 대비 `+0.0192%p`, 모델 116 대비 `+0.7026%p` 상승했다.

원격 결과:

- 모델 139 validation: `$KOMAP_WORKSPACE/outputs/restart93_dynamic_pair/`
- 모델 139 test masks: `$KOMAP_WORKSPACE/outputs/restart94_dynamic_pair_best/test_masks/`

## 모델 140: label-free image-adaptive intensity prior

모델 139의 예측 확률로 각 이미지의 클래스별 표준화 명암 분포를 추정하고,
Al3Ni·Eutectic Si 두 클래스에 한해 train 전체의 global prior와 이미지별
adaptive prior를 1:1로 혼합했다. 정답 라벨은 사용하지 않았으며, validation
image-wise mIoU는 77.4573%로 139번보다 0.0067%p 높았다. 다만 개선폭이 매우
작아 실질적인 포화 상태로 판단하고, 테스트 마스크를 별도 보존한다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 140 | 모델 139 + Al3Ni/Eutectic 이미지 adaptive prior, beta=1.0 | 80.2755% | **77.4573%** | 현재 최고 후보, 미세 개선 |

원격 결과: `$KOMAP_WORKSPACE/outputs/restart96_adaptive_prior/results.json`

## 모델 141: Al3Ni bias 재탐색

모델 139/140 mixture에서 Al3Ni logit bias를 고정값과 이미지 평균 intensity
조건부 값으로 재탐색했다. adaptive prior 기준 최적 조건은 평균 intensity 175
이상에서만 `+0.10`을 적용하는 설정이었지만 image-wise mIoU 77.4376%로
모델 140보다 낮아 채택하지 않았다. Al3Ni 단순 bias만으로는 A71 계열의
Al3Ni–Eutectic 경계 오류를 해결하지 못한다.

| 모델 | 구성 | Image-wise 평균 | 판정 |
|---:|---|---:|---|
| 141 | 모델 140 + Al3Ni fixed/mean-conditioned bias search | 77.4376% | 미채택 |

원격 결과: `$KOMAP_WORKSPACE/outputs/restart98_al3ni_bias_search/results.json`

## 모델 142: boundary fine-tune source 추가

고해상도 ASPP U-Net++에 boundary CE와 Al3Ni–Eutectic 경계 보조항을 넣어
fine-tune한 모델(단일 75.63%)을 현재 139번 TENT/style/pair ensemble에
소량 추가했다. high branch를 대체하거나 기존 base source를 줄이는 두 방식을
탐색했지만 최고 image-wise mIoU는 77.4568%로 140번의 77.4573%를 넘지 못했다.
경계 fine-tune source는 미채택한다.

| 모델 | 구성 | Image-wise 평균 | 판정 |
|---:|---|---:|---|
| 142 | 모델 139 + boundary ASPP source weight 0.15 | 77.4568% | 미채택 |

원격 결과: `$KOMAP_WORKSPACE/outputs/restart100_boundary_source/results.json`

## 모델 143: Al3Ni 전용 강한 adaptive prior

모델 140의 이미지 adaptive prior를 gamma와 클래스별 beta로 확장했다. 최적값은
예측 확률로 추정한 Al3Ni 명암분포에 `gamma=3.0`, `beta=3.0`을 적용하고,
Eutectic Si는 adaptive 보정을 적용하지 않는 설정이었다. 정답 mask를 사용하지
않는 label-free 보정이며, A71의 Al3Ni 과소예측을 완화했다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 143 | 모델 139 + Al3Ni adaptive prior(gamma 3, beta 3) | **80.3348%** | **77.5663%** | 현재 최고 |

모델 140 대비 image-wise `+0.1090%p` 상승했다. 원격 결과:
`$KOMAP_WORKSPACE/outputs/restart101_adaptive_prior_grid/results.json`

## 모델 144: Al3Ni adaptive prior extrapolation

Al3Ni adaptive prior의 gamma/beta 범위를 확장한 결과, `gamma=2.0`,
`Al3Ni beta=5.0`, `Eutectic beta=0.0`에서 최고값을 얻었다. Eutectic은
global prior를 유지하고 Al3Ni만 강하게 이미지 적응시키는 설정이다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 144 | 모델 139 + Al3Ni adaptive prior(gamma 2, beta 5) | **80.3592%** | **77.5798%** | 현재 최고 |

143번 대비 image-wise `+0.0135%p` 상승했다. 원격 결과:
`$KOMAP_WORKSPACE/outputs/restart103_adaptive_prior_extrapolate/results.json`

## 모델 145: Al3Ni adaptive prior + TENT strength search

144번의 Al3Ni adaptive prior(`gamma=2`, `beta=5`)를 고정하고 TENT steps/lr를
재탐색했다. `5 steps, lr=1e-5`가 image-wise 기준 최적이었다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 145 | 모델 144 + TENT 5 steps, lr 1e-5 | 80.2545% | **77.6182%** | 현재 최고 |

144번 대비 image-wise `+0.0383%p` 상승했다. 원격 결과:
`$KOMAP_WORKSPACE/outputs/restart105_tent_hyper/results.json`

## 모델 146: TENT5 adaptive prior route 재조정

145번의 TENT 5-step/lr 1e-5와 Al3Ni adaptive prior를 유지하고, intensity
조건부 high-resolution branch와 pairwise source 비율을 재탐색했다. low-intensity
이미지는 high branch 75%, high-intensity 이미지는 55%, pairwise source는 15%,
평균 intensity threshold는 165를 사용한다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 146 | 모델 145 + high/pair route 재조정(75/55%, pair 15%) | **80.2588%** | **77.6504%** | 현재 최고 |

145번 대비 image-wise `+0.0323%p` 상승했다. 원격 결과:
`$KOMAP_WORKSPACE/outputs/restart108_route_small/results.json`

## 모델 147: route 고정 Al3Ni beta 재탐색

146번 route(`low/high high branch=75/55%`, pair=15%, threshold=165)와
TENT 5-step/lr 1e-5를 고정하고 Al3Ni adaptive prior의 gamma/beta를 다시
탐색했다. 최적은 `gamma=2.0`, `Al3Ni beta=7.0`, Eutectic beta=0.0이었다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 147 | 모델 146 + Al3Ni prior gamma 2, beta 7 | 80.2540% | **77.6521%** | 현재 최고 |

146번 대비 image-wise `+0.0017%p` 상승했다. 원격 결과:
`$KOMAP_WORKSPACE/outputs/restart111_beta_route_correct/results.json`

## 모델 148: Eutectic bias 재조정

147번의 TENT/prior/route를 고정하고 Eutectic Si logit bias를 이미지 평균
intensity 조건부로 재탐색했다. low-intensity 구간의 bias를 `-1.00`으로
강화하고 high-intensity 구간은 `-0.30`, threshold는 165를 사용했다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 148 | 모델 147 + Eutectic bias low/high=-1.00/-0.30 | **80.3045%** | **77.6907%** | 현재 최고 |

147번 대비 image-wise `+0.0386%p` 상승했다. 원격 결과:
`$KOMAP_WORKSPACE/outputs/restart113_eutectic_route/results.json`

## 모델 149: Eutectic low-bias extrapolation

148번의 low Eutectic bias가 탐색 범위의 경계였기 때문에 `-1.00~-1.50`으로
범위를 확장했다. 최적은 다시 `low=-1.00`, `high=-0.30`, threshold=165로
돌아왔으며 image-wise mIoU 77.6908%로 148번과 사실상 동일했다. 더 강한
Eutectic 억제는 채택하지 않았다.

| 모델 | 구성 | Image-wise 평균 | 판정 |
|---:|---|---:|---|
| 149 | 모델 148 + low Eutectic bias extrapolation | 77.6908% | 148번 유지 |

원격 결과: `$KOMAP_WORKSPACE/outputs/restart115_eutectic_extrapolate/results.json`

## 모델 150: raw/style/joint TENT 비교

148번 route를 고정하고 BN affine entropy adaptation을 raw-only, style-only,
raw+style joint로 비교했다. raw-only가 77.6900%, style-only가 77.6058%,
joint가 77.5430%로 모두 148번을 넘지 못했다. 기존 raw-only TENT를 유지한다.

| 모델 | 구성 | Image-wise 평균 | 판정 |
|---:|---|---:|---|
| 150 | raw-only/style-only/joint TENT 5 steps | 77.6900% | 148번 유지 |

원격 결과: `$KOMAP_WORKSPACE/outputs/restart116_domain_tent/results.json`

## 모델 151: Al3Ni/Eutectic connected-component cleanup

현재 logits의 작은 Al3Ni 및 Eutectic Si 연결요소를 차선 logits로 재할당하는
후처리를 탐색했다. baseline cleanup(두 threshold=0)이 77.6907%였고, 모든
양의 threshold 조합이 하락해 후처리는 적용하지 않는다.

| 모델 | 구성 | Image-wise 평균 | 판정 |
|---:|---|---:|---|
| 151 | Al3Ni/Eutectic component threshold search | 77.6907% | 미채택 |

원격 결과: `$KOMAP_WORKSPACE/outputs/restart117_components/results.json`

## 모델 152: global TENT domain adaptation

valid 전체 이미지를 하나의 unlabeled domain으로 묶어 raw/style/joint global
TENT를 수행했다. style-only가 가장 높았지만 77.3740%로 148번보다 낮았고,
raw/joint도 개선이 없었다. per-image raw TENT를 유지한다.

| 모델 | 구성 | Image-wise 평균 | 판정 |
|---:|---|---:|---|
| 152 | valid 전체 global raw/style/joint TENT | 77.3740% | 미채택 |

원격 결과: `$KOMAP_WORKSPACE/outputs/restart118_global_tent/results.json`

## 모델 153: 이미지 밝기 조건부 Al3Ni prior

148번의 route/TENT/Eutectic bias 설정을 고정하고, 이미지 평균 밝기에 따라
Al3Ni prior beta를 다르게 적용했다. 평균 밝기 threshold는 160이며, low/high
구간 beta는 각각 3/7이다. 전체 조합 중 이 설정이 가장 높았다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 153 | 모델 148 + mean intensity 조건부 Al3Ni beta (160, 3/7) | **80.3305%** | **77.7243%** | **현재 최고** |

148번 대비 image-wise `+0.0336%p` 상승했다. 원격 결과:
`$KOMAP_WORKSPACE/outputs/restart119_conditional_beta/results.json`

## 모델 154: TENT 고해상도 source 교체 탐색

모델 153의 per-image TENT(5 step, lr `1e-5`)와 ensemble route를 고정하고,
고해상도 branch의 ASPP 체크포인트를 `restart51`, `model109`, seed456,
pairwise-margin, EMA 후보로 교체했다. `restart51`이 기준과 사실상 동일한
최고 결과였고, 나머지는 모두 하락했다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 154 | 모델 153 + high ASPP source 5종 비교 | **80.3305%** | **77.7243%** | 153번 유지 |

최고 source인 `restart51` 결과는 image-wise `77.72428%`로 153번
`77.72431%`와 반올림 이하 차이이며, 교체 source는 채택하지 않았다.
원격 결과: `$KOMAP_WORKSPACE/outputs/restart122_high_source/results.json`

## 모델 155~156: source-free pseudo-label domain adaptation

모델 153의 ensemble 예측을 teacher pseudo-label로 사용해 각 validation 이미지의
고해상도 ASPP decoder를 1 step 무라벨 적응했다. confidence threshold를 0.85와
0.90으로 비교했으며, validation 정답은 적응 과정에 사용하지 않았다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 155 | pseudo-label decoder adaptation, confidence 0.85 | 80.5116% | 77.4408% | 미채택 |
| 156 | pseudo-label decoder adaptation, confidence 0.90 | 80.5100% | 77.4415% | 미채택 |

두 설정 모두 global mIoU는 상승했지만 어려운 이미지의 image-wise 점수가 하락해
모델 153을 유지한다. 원격 결과:
`$KOMAP_WORKSPACE/outputs/restart123_pseudo_domain/results.json`,
`$KOMAP_WORKSPACE/outputs/restart124_pseudo_domain_conf90/results.json`

## 모델 157: Primary Si 면적 기반 false-positive 억제

모델 153 route의 최종 logits에서 Primary Si hard prediction 면적이 전체의
1% 이하인 이미지는 Primary Si 픽셀을 차선 logits로 재할당했다. 작은 Primary
Si false-positive가 많은 A71 이미지가 개선됐고, 다른 class의 면적 threshold
조합보다 높았다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 157 | 모델 153 + Primary Si area fraction `<=0.01` 억제 | **80.3853%** | **77.7644%** | **현재 최고** |

153번 대비 image-wise `+0.0401%p` 상승했다. 원격 결과:
`$KOMAP_WORKSPACE/outputs/restart125_area_postprocess/results.json`

## 모델 158: branch probability fusion 방식 비교

모델 157의 route를 고정하고 branch 확률의 산술 평균, geometric fusion,
temperature 보정(`T=0.75, 0.85, 1.15, 1.30`)을 비교했다. 산술 평균이 가장
높았고, 최고 결과도 모델 157과 동일해 fusion 방식은 변경하지 않는다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 158 | probability/geometric/temperature fusion search | 80.3854% | 77.7645% | 157번 유지 |

원격 결과: `$KOMAP_WORKSPACE/outputs/restart127_fusion_modes/results.json`

## 모델 159: z-score/percentile intensity prior 혼합

이미지별 grayscale z-score prior와 pixel percentile-domain prior를 혼합해
재탐색했다. 최고 percentile weight는 `0.15`였고, Primary Si area `<=0.01`
후처리와 함께 적용했을 때 모델 157보다 소폭 높았다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 159 | z/percentile prior blend, q weight 0.15 | **80.4133%** | **77.7689%** | **현재 최고** |

157번 대비 image-wise `+0.0045%p` 상승했다.

원격 결과: `$KOMAP_WORKSPACE/outputs/restart128_prior_representation/results.json`

## 모델 160: 고해상도 overlap stride 재탐색

모델 159의 route를 유지하고 high ASPP branch의 sliding-window stride를
`576 → 384`로 줄여 overlap을 늘렸다. 경계 coverage는 늘었지만 image-wise
mIoU가 하락해 기존 stride 576을 유지한다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 160 | high 768 branch stride 384 | 80.4145% | 77.7695% | 159번 유지 |

원격 결과: `$KOMAP_WORKSPACE/outputs/restart130_overlap_stride/results.json`

## 모델 161: 현재 route의 미세 bias search

모델 159의 z-score/percentile prior(`q weight=0.15`), 조건부 Al3Ni beta,
Primary Si 면적 억제를 고정하고 Al bias, Eutectic Si bias, 밝기 분기 threshold를
135개 조합으로 세밀하게 탐색했다. 최적값은 Al `+0.45`, low/high Eutectic Si
`-1.0/-0.3`, threshold `165`였다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 161 | 모델 159 + fine bias search (`Al +0.45`) | **80.4381%** | **77.7703%** | **현재 최고** |

159번 대비 image-wise `+0.0013%p`의 미세 개선이다. 원격 결과:
`$KOMAP_WORKSPACE/outputs/restart131_bias_search/results.json`

## 모델 162: branch 가중치 미세 탐색

모델 161의 prior·bias·Primary Si 면적 억제 설정을 고정하고, 고해상도 TENT
branch와 pairwise branch의 가중치를 75개 조합으로 탐색했다. 이미지 평균 밝기
165 미만/이상에 대해 high branch weight를 다르게 적용했다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 162 | 모델 161 + branch weight (`low/high=0.85/0.55`, pair=`0.10`) | **80.4288%** | **77.7817%** | **현재 최고** |

161번 대비 image-wise `+0.0115%p` 상승했다. 원격 결과:
`$KOMAP_WORKSPACE/outputs/restart133_route_weight_search/results.json`

## 모델 163: 현재 최고 teacher 기반 knowledge distillation

모델 162 route의 soft logits를 train 70장에 생성하고, `restart19` U-Net++ student를
hard CE+Dice와 temperature-scaled KL(`T=2`, distill weight `0.5`)로 4 epoch 학습했다.
기존 101번 증류와 달리 최신 high/TENT/pairwise/prior route를 teacher로 사용했다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 163 | 모델 162 teacher → U-Net++ student, best epoch 4 | 79.1922% | 75.6083% | 미채택 |

단일 student는 모델 162 ensemble보다 낮았다. 원격 결과:
`$KOMAP_WORKSPACE/outputs/restart135_current_distill/`

## 모델 164: distilled student branch 보완성 탐색

모델 163의 distilled student 확률을 모델 162의 p1 branch에 `0~100%` 혼합하고,
Al3Ni/Eutectic Si에만 student를 추가하는 class-specific 혼합도 비교했다. 기준
route를 포함한 21개 조합에서 student weight 0이 최적이었다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 164 | 모델 162 + distilled student branch search | 80.4290% | 77.7819% | 162번 유지 |

증류 student의 단일 성능과 branch 보완성 모두 모델 162보다 낮아 제출 route에는
추가하지 않았다. 원격 결과:
`$KOMAP_WORKSPACE/outputs/restart136_distill_branch_search/results.json`

## 모델 165: Al3Ni/Eutectic intensity threshold fine search

모델 162 route의 branch, prior, bias를 고정하고 Al3Ni beta 분기 threshold
`158~163`과 Eutectic bias 분기 threshold `163~167`을 조합해 탐색했다.
최고 조합은 beta threshold 158, Eutectic threshold 165였지만 기준 route와
차이가 없어 기존 threshold `160/165`를 유지한다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 165 | 모델 162 + conditional threshold fine search | 80.4289% | 77.7819% | 162번 유지 |

원격 결과:
`$KOMAP_WORKSPACE/outputs/restart137_threshold_search/results.json`

## 모델 166: A71-like hard-case conditional route

모델 162의 기본 route는 유지하고, 평균 grayscale가 `159 이상 161 미만`인
A71-like 이미지에만 high branch `0.70`, pairwise `0.05`, raw/style `0/1`
혼합, Al bias `+0.45`, Al3Ni adaptive beta `10`, Eutectic bias `-1.5`를
적용했다. 나머지 valid 이미지는 모델 162 설정을 그대로 사용했다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 166 | A71-like mean-conditioned hard-case route | **80.4748%** | **77.8652%** | **현재 최고** |

모델 162 대비 image-wise `+0.0834%p` 상승했으며, A71 이미지 score는
`45.37% → 47.04%`로 상승했다. 원격 결과:
`$KOMAP_WORKSPACE/outputs/restart139_hard_case_search/results.json`

## 모델 167: hard-case route 다른 구간 전이 검증

모델 166의 특수 route를 평균 밝기 구간 `[161,163)`, `[164,166)`, `[173,176)`에
각각 전이해 검증했다. A71 구간 `[159,161)`만 개선을 유지했고, 다른 구간은
image-wise mIoU가 하락했다.

| 적용 구간 | Image-wise 평균 | 판정 |
|---|---:|---|
| `[159,161)` | 77.8655% | 모델 166 유지 |
| `[161,163)` | 77.8588% | 미적용 |
| `[164,166)` | 77.7321% | 미적용 |
| `[173,176)` | 77.4295% | 미적용 |

원격 결과:
`$KOMAP_WORKSPACE/outputs/restart141_hard_transfer/results.json`

## 모델 168: hard-case + narrow Eutectic bias 조합

모델 166의 A71-like 특수 route를 유지하고, 서로 겹치지 않는 밝기 구간에
Eutectic bias를 추가했다.

- `[161.3, 162.0)`: `-2.0` (A186)
- `[164.5, 165.2)`: `-0.75` (A153)
- `[174.0, 175.5)`: `+0.3` (A11/A67 계열)

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 168 | 모델 166 + 3개 narrow Eutectic bias | **80.6530%** | **78.0012%** | **현재 최고** |

모델 166 대비 image-wise `+0.1367%p` 상승했다. 원격 결과:
`$KOMAP_WORKSPACE/outputs/restart143_narrow_eutectic_combo/results.json`

## 모델 169: narrow Al3Ni beta 조합

모델 168의 특수 route와 Eutectic narrow bias를 유지하고, Al3Ni 오류가 큰 좁은
밝기 구간에만 adaptive beta를 추가 조정했다.

- `[164.0,164.5)`: beta `7 → 15` (A11 step1)
- `[174.6,174.85)`: beta `7 → 15` (A67)
- `[174.85,175.1)`: beta `7` 유지 (A11 step2)

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 169 | 모델 168 + narrow Al3Ni beta | **80.6619%** | **78.0293%** | **현재 최고** |

168번 대비 image-wise `+0.0281%p` 상승했다. 원격 결과:
`$KOMAP_WORKSPACE/outputs/restart145_narrow_al3ni/results.json`
최종 test masks: `$KOMAP_WORKSPACE/outputs/restart146_narrow_al3ni_best/test_masks/`

## 모델 170: hard-case별 high source 교체

모델 169 route를 유지하면서 hard-case 구간에만 high-resolution source를 교체했다.

- A71-like `[159,161)`: pairwise source (`restart61`)
- A186-like `[161.3,162)`: EMA source (`model123`)
- A67-like `[174.6,174.85)`: native ASPP source (`model109`)

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 170 | 모델 169 + hard-case high-source routing | **80.7303%** | **78.1034%** | **현재 최고** |

169번 대비 image-wise `+0.0741%p` 상승했다. 원격 결과:
`$KOMAP_WORKSPACE/outputs/restart148_hard_source_combo/results.json`
최종 test masks: `$KOMAP_WORKSPACE/outputs/restart149_hard_source_best/test_masks/`

## 모델 171: 추가 hard-case high source routing

모델 170에 다음 source 교체를 추가했다.

- A11 step2 `[174.85,175.1)`: model109 native ASPP
- A40/A74 계열 `[152.3,152.7)`: seed456 ASPP
- A44 계열 `[155.2,155.7)`: EMA ASPP

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 171 | 모델 170 + 추가 hard-case source routing | **80.7753%** | **78.1722%** | **현재 최고** |

170번 대비 image-wise `+0.0688%p` 상승했다. 원격 결과:
`$KOMAP_WORKSPACE/outputs/restart150_hard_source_combo2/results.json`
최종 test masks: `$KOMAP_WORKSPACE/outputs/restart151_hard_source_best/test_masks/`

## 모델 172: 이미지별 class-bias oracle 진단

모델 171 route의 Al3Ni·Eutectic Si·Al logit에 이미지별 보정값을 적용했을 때의
검증 상한을 측정했다. validation 정답을 사용하는 진단용 결과이며 실제 제출
route에는 직접 적용하지 않았다.

| 구분 | Global mIoU | Image-wise 평균 |
|---|---:|---:|
| 모델 171 route 재현 | 80.7756% | 78.1728% |
| 이미지별 bias oracle | 81.3526% | 78.6988% |

class bias만으로는 최대 `+0.5260%p` 회복에 그쳐 80% 달성의 주된 해법으로
보기 어렵다는 결론을 얻었다.

원격 결과: `$KOMAP_WORKSPACE/outputs/model172_bias_oracle/results.json`

## 모델 173: 1024px high-resolution 문맥 branch 혼합

모델 171의 route와 hard-case source/bias 설정은 고정하고, ASPP high branch의
동일 TENT 모델을 1024px patch/768px stride로 추가 추론했다. 기존 768px branch와
1024px branch의 혼합률만 탐색했다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 173 | 모델 171 + high branch 1024px 혼합 0.20 | **80.8435%** | **78.2344%** | **현재 최고** |

모델 171 대비 image-wise `+0.0622%p` 상승했다. 원격 결과:
`$KOMAP_WORKSPACE/outputs/model173_highres1024/results.json`

### 평가 해석 주의 (173번 이후 감사)

166~173번은 동일 validation의 특정 이미지가 속한 좁은 밝기 구간을 반복해서
조정했다. 78.2344%는 이 validation에서 선택한 점수이며 독립적인 일반화 성능의
증거는 아니다. 후속 실험은 이미지별 정답을 이용한 추가 분기를 피하고,
train 내부 specimen 단위 검증으로 설정을 결정하는 방향으로 진행한다.
172번의 78.6988% 역시 탐색한 bias 범위 안의 oracle 값이며 모든 보정 방식의
수학적 상한은 아니다. test에는 정답 마스크가 없어 로컬 test 점수는 측정하지 않았다.

## 모델 174: A71 Primary Si 면적 억제 재검증

모델 173 조건에서 A71의 Primary Si 예측은 2,501/324,000픽셀(0.772%)이었다.
억제 전 image mIoU 46.5669%, 기존 1% 억제 적용 후 47.5526%로 측정됐다.
1%보다 큰 threshold도 결과가 같아 기존 모델 대비 추가 개선은 없다. 미채택.
정답에 Primary Si가 없으며, 현재 평가식은 union=0인 클래스도 IoU=0으로
포함한다. 평가 규칙을 바꾸어 개선으로 계산하지 않는다.
결과: `$KOMAP_WORKSPACE/outputs/model174_a71_area/results.json`

## 모델 175: 1024px 경계 손실 fine-tune

restart51에서 시작해 train 70장, 4 crops/image, batch 1, lr 3e-6,
3 epoch, boundary weight 0.15로 학습했다. validation은 20장이다.

| Epoch | Global mIoU | Image-wise mIoU |
|---:|---:|---:|
| 1 | 78.1609% | 74.9016% |
| 2 | 78.3837% | 75.0973% |
| 3 | 78.2122% | 75.2613% |

최고 checkpoint는 epoch 3. 단일 모델 결과이므로 ensemble 173번과 동일 조건의
비교는 아니며, source 교체 검증은 176번에서 수행한다.
결과: `$KOMAP_WORKSPACE/outputs/restart175_aspp1024_finetune/summary.json`

## 모델 176: 1024px 재학습 high source 교체 결과

175번 checkpoint를 일반 이미지의 high source로 사용하고, 기존 hard-case source
routing을 유지했다. 768px/1024px 혼합률을 탐색했지만 모든 조합이 모델 173보다
낮았다.

| 모델 | 최적 혼합률 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---:|---:|---:|---|
| 176 | 재학습 source, alpha 0.30 | 80.7407% | 78.1201% | 미채택 |

모델 173 대비 `-0.1143%p`로, 1024px 입력에 맞춘 추가 fine-tune이 현재
ensemble에 보완 정보를 제공하지 못했다. 결과:
`$KOMAP_WORKSPACE/outputs/model176_aspp1024_source/results.json`

### 후속 학습 프로토콜 및 checkpoint 선택 감사

train 70장(59 시편)을 학습 57장/내부 검증 13장(12 시편)으로 분리한
`outputs/grouped_protocol177.json`을 생성했다. 내부 두 분할의 시편 교집합은 없다.
공식 train/valid의 공통 시편은 A125, A153, A181, A191, A40, A55, A7이다.
공식 분할 자체는 변경하지 않았다. 기존 KoMaP checkpoint는 내부 검증 이미지를
이미 학습했으므로 이 내부 검증에는 새 초기화 또는 외부 사전학습만 허용한다.

`encoder_structure_search.py`의 checkpoint 및 calibration 선택 기본 지표를
image_mean_miou로 수정했다. 주요 source의 history 감사 결과,
restart51(epoch 5), restart59(epoch 5), model109(epoch 4), restart19(epoch 1)은
global/image-wise 최고 epoch가 이미 동일했다. 따라서 선택 기준 변경 자체로
현재 ensemble 점수가 상승했다고 볼 수 없다. 원격 CLI import 및 로컬 구문 검증 통과.

## 모델 177: 전체 validation source oracle 진단

모델 173 route에서 사용 가능한 5개 high-resolution source(restart51, model109,
seed456, pairwise, EMA)를 validation 20장 전체에 각각 적용해 source별 결과를
비교했다. 이미지별 정답을 사용해 가장 좋은 source를 고르는 oracle은 실제 제출
route가 아니며, source 선택만으로 얻을 수 있는 상한을 확인하는 진단이다.

| 구분 | Image-wise 평균 |
|---|---:|
| 현재 source routing 재현 | 78.1723% |
| 이미지별 source oracle | 78.2271% |

source oracle의 추가 여지는 `+0.0548%p`에 불과했다. 따라서 high source 교체나
이미지별 source routing만으로는 80% 목표에 도달하기 어렵고, 다음 실험은
source가 아닌 도메인 적응·공간적 경계/오류 보정에 집중한다. 결과:
`$KOMAP_WORKSPACE/outputs/model177_all_source/results.json`

## 모델 178: source-free TENT 입력 도메인 비교

모델 173의 source routing, adaptive intensity prior, 768/1024 blend 조건을
고정하고, high branch를 무라벨 적응할 때 사용하는 입력만 비교했다.

| 적응 입력 | 최적 1024 blend | Global mIoU | Image-wise 평균 | 판정 |
|---|---:|---:|---:|---|
| raw tiles | 0.20 | 80.8393% | 78.2317% | 173번 유지 |
| style tiles | 0.10 | 80.9670% | 78.1914% | 미채택 |
| raw + style joint | 0.20 | 80.5985% | 77.9475% | 미채택 |

raw 적응은 173번 `78.2344%`와 사실상 동일하고, style-only/joint 적응은
image-wise 평균을 낮췄다. 따라서 현재 TENT 도메인 적응 설정은 포화로 판단하고
제출 route에는 173번을 유지한다. 결과:
`$KOMAP_WORKSPACE/outputs/model178_tta_modes/results.json`

## 모델 179: train-only specimen-aware intensity prior

validation 정답을 사용하지 않고, validation 이미지와 같은 A-ID를 가진 train
이미지의 클래스별 z-score/percentile 밝기 분포를 prior에 혼합했다. 같은 specimen이
없는 validation 이미지는 global train prior를 그대로 사용했다.

| 모델 | 최적 설정 | Global mIoU | Image-wise 평균 | 판정 |
|---|---|---:|---:|---|
| 179 | `rho=0.0`, adaptive beta scale `1.0` | 80.8435% | 78.2344% | 173번 유지 |

최적 `rho=0`은 specimen 통계를 전혀 섞지 않는 기존 global prior와 동일하다.
따라서 현재 7개 specimen overlap만으로는 밝기 분포 prior를 개선할 수 없었다.
결과: `$KOMAP_WORKSPACE/outputs/model179_specimen_prior/results.json`

## 모델 180: model173 logit 공간 spatial refinement

모델 173의 checkpoint, source routing, 768/1024 blend, intensity prior, class bias와
Primary cleanup을 모두 고정하고, 최종 argmax 직전의 logit map에만 작은 Gaussian
smoothing을 적용했다. validation 정답으로 이미지별 처리를 고른 것이 아니라, 모든
이미지와 class에 동일한 sigma 후보를 적용해 spatial boundary noise를 줄이는 방식이다.

| 후보 | Global mIoU | Image-wise 평균 | 판정 |
|---|---:|---:|---|
| 모델 173 baseline 재현 | 80.8435% | 78.2344% | 기준 |
| 180 `Primary Si`, sigma `0.50` | 80.8628% | **78.2653%** | **image-wise 최고** |
| 180 `Primary Si`, sigma `0.75` | **80.8637%** | 78.2411% | global 최고 |

Al3Ni, Al, Eutectic Si 전체를 smoothing하거나 sigma를 크게 하면 작은 상과 경계가
지워져 성능이 하락했다. Primary Si logit에만 sigma `0.50`을 적용한 경우 모델 173
대비 image-wise `+0.0308%p` 상승했지만, 목표 80%까지는 여전히 `1.7347%p`가 남아
있다. 결과:
`$KOMAP_WORKSPACE/outputs/model180_spatial_refine/results.json`

## 모델 181: 팀원식 raw/z-score/CDF 입력 branch

팀원 `model_results_0921_1.md`의 입력 표현을 검증하기 위해 ResNet34-ASPP branch에
`raw grayscale + per-image z-score + empirical CDF` 3채널을 적용했다. ASPP dilation은
팀원 문서와 같이 `1,2,4,8`로 바꾸고, model109 ASPP checkpoint에서 fine-tune했다.
5 epoch, rare-aware 512px patch, CE+Dice와 D4/Gaussian validation을 사용했다.

| 모델 | Global mIoU | Image-wise 평균 | empty 제외 image-wise | 판정 |
|---:|---:|---:|---:|---|
| 181 epoch 5 | 78.1301% | 75.1734% | 75.7675% | 단일 branch 미채택 |

입력 표현만 바꾼 단일 branch는 기존 RGB branch보다 낮았다. 다만 오류 상관이 낮을 수
있어 확률 branch로는 다음 모델 182에서 별도 검증했다. 결과:
`$KOMAP_WORKSPACE/outputs/model181_raw_zcdf_aspp/summary.json`

## 모델 182: raw/z-score/CDF branch 확률 혼합

모델 180 route를 고정하고 모델 181의 확률을 추가했다. `all`, `rare(Al3Ni+Eutectic)`,
`Primary`, `Al` class별 혼합 모드와 gamma `0.00~0.50`을 탐색했다. 최종 logit에는
모델 180과 동일한 Primary Gaussian smoothing sigma `0.50`을 유지했다.

| 모델 | 최적 혼합 | Global mIoU | Image-wise 평균 | empty 제외 image-wise | 판정 |
|---:|---|---:|---:|---:|---|
| 182 | Primary gamma `0.40` | 80.8917% | **78.2723%** | 79.0649% | 현재 내부 image-wise 최고 |

모델 180의 78.2653%에서 `+0.0070%p`로 개선 폭이 매우 작았다. 팀원식 입력 표현은
독립 branch의 보완 정보는 일부 제공하지만, 현재 route의 주 병목을 해결하지 못해
단독 backbone이나 high branch 교체로는 사용하지 않는다. 결과:
`$KOMAP_WORKSPACE/outputs/model182_zcdf_blend/results.json`

## 모델 183: 평가 기준 정렬형 logit calibration

모델 182 route를 고정한 상태에서, validation 정답을 사용하지 않는 공통 보정값만
탐색했다. 대상은 `Al3Ni` logit delta, `Eutectic Si` bias, `Primary Si` bias,
intensity prior scale, adaptive `Al3Ni` beta scale, Primary Gaussian sigma,
Primary connected-component threshold다. `empty_union=exclude` 기준과 현재 내부
기준을 함께 계산해 팀원 결과와 비교할 때 평가 규칙 차이를 확인할 수 있게 했다.

구현상 중요한 점은 adaptive prior delta를 모델 182와 동일하게 **Al3Ni(class 0)에만**
적용한 것이다. 이를 모든 class에 더하면 모델 182 baseline을 재현하지 못하므로
실험에서 제외했다. Gaussian filter는 linearity를 이용해 후보 평가 전에 Primary
관련 항을 캐시했고, 연결요소 정리는 상위 calibration 후보에만 적용했다.

| 기준 | 최적 설정 | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---|---|---:|---:|---:|
| image-wise/empty 제외 | prior `1.10`, beta `0.75`, Al3Ni bias `0.00`, Eutectic bias `0.00`, Primary bias `0.00`, sigma `0.50`, cleanup `128` | 80.8854% | **78.2827%** | **79.0719%** |
| global | prior `1.10`, beta `0.75`, Al3Ni bias `+0.20`, Eutectic bias `0.00`, Primary bias `0.00`, sigma `0.50`, cleanup `128` | **80.9458%** | 78.2271% | 79.0175% |

모델 182 대비 image-wise 평균은 `+0.0104%p`, empty 제외 기준은 `+0.0069%p`,
global 최적값은 `+0.0541%p` 개선됐다. 개선 폭은 작지만, 정규화·최적화보다
최종 class prior calibration이 현재 route에서 남아 있는 소폭의 개선 여지임을
확인했다. image-wise 대회 지표용 권장 route는 첫 번째 행이다.

원격 결과:
`$KOMAP_WORKSPACE/outputs/model183_metric_calibration/results.json`

### 모델 183 권장 route의 상세 처리

모델 183은 새 backbone checkpoint를 학습한 모델이 아니라 모델 182의 inference
route를 유지하면서 최종 logit calibration만 바꾼다.

1. 모델 173의 ResNet34 U-Net++/Residual ASPP branch, hard-case high-source routing,
   768/1024 high-resolution blend와 모델 181 raw/z-score/CDF Primary 보조 branch를
   그대로 사용한다.
2. Primary Si 확률에는 모델 182와 같이 model181 branch를 gamma `0.40`으로 혼합한다.
3. train-only intensity prior 전체를 logit에 넣는 계수를 기존 `0.09`에서 바꾸는
   것이 아니라, 최종 prior map에 scale `1.10`을 곱한다.
4. adaptive brightness correction은 Al3Ni class에만 beta `0.75`를 곱한다. 이 값은
   기존 beta의 강도를 약간 낮춰 Al3Ni false positive/false negative 균형을 조정한다.
5. Primary Si logit만 Gaussian sigma `0.50`으로 smoothing하고, 128픽셀 이하의
   Primary connected component를 다음 logit class로 재할당한다.
6. image-wise 지표용 최종 bias는 Al3Ni/Eutectic/Primary 모두 추가하지 않는다.
   global 최적 후보의 Al3Ni `+0.20` bias는 전체 픽셀 지표는 올리지만 image-wise
   평균을 낮추므로 제출 권장값에서 제외한다.

## 모델 184: intensity-guided edge-aware Primary refinement

모델 183의 image-wise 최적 route와 calibration을 고정하고, Primary Si logit의
Gaussian smoothing 일부를 원본 grayscale z-score guidance를 사용하는 bilateral
filter로 교체했다. 모델·checkpoint·branch weight·prior는 변경하지 않았고, validation
정답은 공통 후처리 조합 선택에 사용하지 않는 방식으로 전체 20장에 동일한 필터를
적용했다.

bilateral filter는 다음처럼 공간 거리와 grayscale 차이를 동시에 사용한다.

```text
w(p,q) = exp(-||p-q||² / (2σs²))
         · exp(-(z(p)-z(q))² / (2σr²))
Primary' = (1-λ)·Gaussian(Primary, 0.50) + λ·Bilateral(Primary, z)
```

| 모델 | 최적 설정 | Global mIoU | Image-wise 평균 | empty 제외 image-wise | 판정 |
|---:|---|---:|---:|---:|---|
| 184 | radius `2`, spatial sigma `1.00`, range sigma `0.60`, blend `0.50` | 80.8942% | **78.2971%** | **79.0863%** | image-wise 최고 갱신 |
| 184 global 후보 | radius `2`, spatial sigma `1.00`, range sigma `1.00`, blend `0.50` | **80.8966%** | 78.2934% | 79.0825% | global 후보 |

모델 183 image-wise 최고 대비 image-wise와 empty 제외 기준 모두 `+0.0144%p`
상승했다. Gaussian만 사용하는 route보다 intensity 경계를 고려한 bilateral 혼합이
Primary의 작은 island와 phase 경계 사이의 과도한 smoothing을 일부 줄였다는 해석이
가능하다. 다만 80% 목표까지는 여전히 `1.7029%p`가 남아 있으므로, 모델 184를
최종 제출 route로 바로 확정하지 않고 다음 class별 경계 보정 실험의 기준으로 사용한다.

원격 결과:
`$KOMAP_WORKSPACE/outputs/model184_edge_refine/results.json`

## 모델 185: class-specific edge-aware refinement

모델 184의 Primary bilateral route를 고정하고, 동일한 edge-aware filter를
Al3Ni·Al·Eutectic Si에 각각 또는 희소 class 조합으로 추가했다. 각 보정 class의
blend는 `0.25, 0.50, 0.75, 1.00`을 비교했으며, class별로 같은 radius `2`, spatial
sigma `1.00`, range sigma `0.60`을 사용했다.

| 모델 | 최적 image-wise 설정 | Global mIoU | Image-wise 평균 | empty 제외 image-wise | 판정 |
|---:|---|---:|---:|---:|---|
| 185 | Eutectic Si만 bilateral blend `0.25` | 80.9145% | **78.2980%** | **79.0885%** | image-wise 최고 소폭 갱신 |
| 185 global 후보 | Al3Ni+Eutectic bilateral blend `0.50` | **80.9451%** | 78.2422% | 79.0396% | global은 183 best보다 낮음 |

모델 184 대비 image-wise는 `+0.0009%p`, empty 제외 기준은 `+0.0023%p`로 사실상
포화에 가깝다. Eutectic Si 단독 보정이 가장 안정적이지만 효과가 매우 작으므로,
최종 route에는 모델 185를 기록하고 동일 조건의 재현 검증을 거친 뒤 채택한다.

원격 결과:
`$KOMAP_WORKSPACE/outputs/model185_class_edge_refine/results.json`

## 모델 186: 팀원식 raw/z-score/CDF branch 장기 학습

팀원 설명 파일의 입력 표현과 Residual ASPP 구조를 유지하면서 model109 checkpoint에서
초기화하고 학습 update 수를 늘렸다. raw grayscale + per-image z-score + empirical CDF
3채널, Residual ASPP dilation `1,2,4,8`, AdamW encoder `1e-4`/decoder `3e-4`,
16 patches/image, batch 2, 8 epochs(약 4,480 updates), D4와 photometric augmentation을
사용했다. P100에서는 BF16 대신 AMP FP16으로 실행했다.

| 모델 | best epoch | Global mIoU | Image-wise 평균 | empty 제외 image-wise | 판정 |
|---:|---:|---:|---:|---:|---|
| 186 | 6 | 79.2662% | 75.7607% | 76.3087% | 단일 branch 미채택, 보조 branch 후보 |

기존 model181보다 단일 branch는 개선됐지만 전체 ensemble route보다 낮다. 따라서
model186을 독립 backbone으로 사용하지 않고, model181과 오류가 다른 보조 확률 branch로
model187에서 검증했다.

결과:
`$KOMAP_WORKSPACE/outputs/model186_team_schedule/summary.json`

## 모델 187: 장기 학습 branch의 Primary 교체 혼합

모델 185의 route(모델 183 calibration + 모델 184 Primary edge refinement + 모델 185
Eutectic edge refinement)를 고정했다. 모델 182에서 사용한 model181 보조 branch를
model186 best checkpoint의 확률로 **교체**하고, Primary Si에만 gamma를 적용했다.
기존 model181 branch를 누적해서 섞지 않고, model182의 Primary gamma `0.40` 이전
확률을 수학적으로 복원한 뒤 model186 확률을 혼합했다.

| 변형 | 최적 gamma | Global mIoU | Image-wise 평균 | empty 제외 image-wise | 판정 |
|---|---:|---:|---:|---:|---|
| 187 image-wise | Primary only | `0.80` | 81.1617% | **78.5155%** | image-wise 최고 |
| 187 global | all classes | `0.40` | **81.1677%** | 78.3236% | global 최고 |

모델 185 image-wise route 대비 Primary-only gamma `0.80`은 image-wise `+0.2174%p`,
empty 제외 `+0.2177%p`, global `+0.2472%p` 상승했다. 장기 학습 branch는 단일
성능보다 기존 branch와의 오류 상관이 낮아 Primary 교체 보조로 사용할 때 효과가 컸다.
대회 image-wise 지표에는 첫 번째 변형을 권장하고, global 변형은 별도로 기록한다.

결과:
`$KOMAP_WORKSPACE/outputs/model187_schedule_branch_blend/results.json`

## 모델 188: model187 Primary gamma 국소 재탐색

모델 187의 Primary gamma 주변 `0.65~0.95`를 `0.05` 간격으로 다시 평가했다.
최고값은 gamma `0.80`에서 그대로 재현됐으므로 model187의 gamma 선택이 탐색 격자
경계에 의한 우연한 선택은 아닌 것으로 확인했다.

| 모델 | gamma | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---:|---:|---:|---:|---:|
| 188 | 0.80 | 81.1617% | **78.5155%** | **79.3062%** |

결과:
`$KOMAP_WORKSPACE/outputs/model188_gamma_refine/results.json`

## 모델 189: model187 branch-specific prior calibration

model187의 Primary gamma `0.80`을 고정하고, model186 branch를 교체한 뒤의
probability distribution에 맞춰 intensity prior scale과 adaptive Al3Ni beta scale을
다시 탐색했다. 이전 model183 calibration을 그대로 재사용하지 않고, 교체된 mixed
probability에서 prior statistics를 재계산했다.

| 기준 | 최적 설정 | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---|---|---:|---:|---:|
| image-wise | prior scale `1.25`, beta `0.75` | 81.1401% | **78.5318%** | **79.3246%** |
| global | prior scale `0.85`, beta `1.00` | **81.1856%** | 78.4837% | 79.2737% |

model187 image-wise route 대비 `+0.0163%p`, empty 제외 `+0.0184%p` 개선됐다. global
최적은 image-wise 최적과 다르므로 대회 image-wise 지표에는 첫 번째 설정을 사용한다.

결과:
`$KOMAP_WORKSPACE/outputs/model189_branch_calibration/results.json`

## 모델 190: model189 prior scale 국소 재탐색

model189 image-wise 최적이 prior scale 탐색 상단에 있었기 때문에 prior `1.20~1.40`,
beta `0.60~0.90`을 다시 비교했다. 최적은 prior `1.25`, beta `0.80`으로, model189의
beta `0.75`보다 약간 낮은 추가 상승을 보였다.

| 모델 | 최적 설정 | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---:|---|---:|---:|---:|
| 190 | prior scale `1.25`, beta `0.80` | 81.1395% | **78.5334%** | **79.3273%** |

model189 image-wise 대비 `+0.0016%p`로 사실상 calibration 포화 구간이다. 현재
image-wise 권장 설정은 모델 190으로 갱신하고, global 지표는 모델 189의 global
후보를 별도 기록한다.

결과:
`$KOMAP_WORKSPACE/outputs/model190_prior_refine/results.json`

## 모델 191: model186 rare-class 추가 혼합

모델 190의 Primary model186 교체 gamma `0.80`과 prior calibration을 고정하고,
model186 probability를 Al3Ni, Eutectic Si, 두 rare class에 각각 소량 추가했다.
Eutectic 추가는 image-wise 평균을 올리지 못했고, Al3Ni 추가 gamma `0.05`가
include image-wise 기준에서만 아주 작은 개선을 보였다.

| 기준 | 최적 설정 | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---|---|---:|---:|---:|
| image-wise | Primary `0.80` + Al3Ni `0.05` | 81.1496% | **78.5338%** | 79.3269% |
| empty 제외 | Primary `0.80` + rare 추가 `0.00` | 81.1395% | 78.5334% | **79.3273%** |
| global | Primary `0.80` + rare `0.30` | 81.1768% | 78.4557% | 79.2352% |

모델 190 대비 include image-wise는 `+0.0004%p`로 사실상 동일하다. Al3Ni 추가
gamma를 크게 하면 성능이 바로 낮아져 rare-class 보조는 포화로 판단한다. 대회
지표가 내부 image-wise 평균이면 model191 include 후보를, empty 제외 규칙이면
model190 후보를 사용한다.

결과:
`$KOMAP_WORKSPACE/outputs/model191_rare_branch_blend/results.json`

## 모델 192: model186 장기학습 설정의 seed 456 재검증

팀원 문서의 장기 학습 일정이 seed에 덜 민감한지 확인하기 위해 model109 checkpoint에서
같은 raw grayscale + per-image z-score + empirical CDF 3채널, Residual ASPP
dilation `1,2,4,8`, 512 patch, 16 patches/image, 8 epochs 설정을 seed `456`으로
다시 학습했다. P100에서는 BF16 대신 AMP FP16을 사용했다.

| 모델 | best epoch | Global mIoU | Image-wise 평균 | empty 제외 image-wise | 판정 |
|---:|---:|---:|---:|---:|---|
| 192 | 5 | 78.4413% | 74.9734% | 75.5204% | model186보다 낮아 미채택 |

model186 seed `2028` branch보다 단일 성능이 낮고, 기존 ensemble의 보조 branch로
사용할 가치가 있는지 별도 확인하기 위해 model193에서 확률 혼합을 추가로 평가했다.

결과:
`$KOMAP_WORKSPACE/outputs/model192_team_schedule_seed456/summary.json`

## 모델 193: seed 456 branch를 현재 최고 route에 교체 혼합

model191의 후처리와 calibration을 고정한 뒤, model192 확률을 Primary에 gamma `0.80`으로
교체하고 Al3Ni/Eutectic Si 추가 혼합 gamma를 `0~0.30`에서 탐색했다. 최적 image-wise
후보는 rare 추가 gamma `0.00`이었으므로, 새 seed의 rare-class 보정은 효과가 없었다.

| 기준 | 최적 설정 | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---|---|---:|---:|---:|
| image-wise | model192 Primary `0.80`, rare 추가 `0.00` | 80.9139% | **78.3244%** | 79.1183% |
| global | model192 Primary `0.80`, Al3Ni 추가 `0.20` | **80.9205%** | 78.2958% | 79.0859% |

model191 image-wise 최고 `78.5338%`보다 `0.2094%p` 낮았고, model190의 기존
model186 Primary branch route도 재현하지 못했다. 따라서 seed 456 branch는 채택하지
않고, 현재 image-wise 제출 후보는 model191을 유지한다.

결과:
`$KOMAP_WORKSPACE/outputs/model193_seed456_rare_blend/results.json`

## 모델 194: 팀원식 512→768→native 다중 해상도 학습

팀원 `model_results_0921_1.md`의 핵심 차이인 512, 768, native-resolution 단계 전환을
재현했다. raw grayscale + per-image z-score + empirical CDF 3채널과 Residual ASPP
dilation `1,2,4,8`을 유지하고, validation은 모든 stage에서 768 patch/576 stride로
동일하게 평가했다. seed는 `2028`이며 P100에서는 AMP FP16을 사용했다.

| stage/epoch | 학습 patch | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---|---:|---:|---:|---:|
| 0/1 | 512 | 78.1946% | 75.2074% | 75.2074% |
| 0/2 | 512 | 77.7932% | 74.3151% | 74.3151% |
| 0/3 | 512 | 78.8450% | 75.7574% | 75.7574% |
| 1/1 | 768 | 78.9288% | 75.6144% | 75.6144% |
| 1/2 | 768 | 79.4423% | 76.4174% | 76.4174% |
| 1/3 | 768 | 79.2527% | 76.0712% | 76.0712% |

native stage 진입 시 batch `1`이 global pooling BatchNorm의 입력 `[1,128,1,1]`을
만들어 중단됐지만, stage 1 best checkpoint는 정상 저장됐다. native stage는 모델195에서
batch `2`로 이어서 재개했다.

결과:
`$KOMAP_WORKSPACE/outputs/model194_multistage_seed2028/history.json`

## 모델 195: native stage batch-2 재개 학습

모델194의 stage 1 best checkpoint에서 이어받아 1024 native crop, 8 patches/image,
batch `2`, encoder/decoder learning rate `5e-5/1.5e-4`로 2 epoch을 추가 학습했다.
BatchNorm 입력 조건을 만족하도록 batch 2를 사용했고, best checkpoint는 native stage
epoch 2였다.

| 모델 | Global mIoU | Image-wise 평균 | empty 제외 image-wise | 판정 |
|---:|---:|---:|---:|---|
| 195 | 79.0904% | **76.1109%** | **76.6552%** | 보조 branch 채택 |

model194 stage1 best의 empty 제외 `76.4174%` 대비 선택 지표는 `76.6552%`
(`+0.2378%p`)로 상승했고, include 기준은 `76.1109%`였다. 이 valid probability를
모델196/197 혼합 탐색에 사용했다.

결과:
`$KOMAP_WORKSPACE/outputs/model195_native_resume/summary.json`

test 정답은 없으므로 수치 점수는 계산하지 않았고, 동일 best checkpoint로 제출용 mask
10장을 다음 경로에 생성했다.

`$KOMAP_WORKSPACE/outputs/model195_native_resume/test_masks/`

## 모델 196: native branch rare-class 혼합

모델195 확률을 기존 model190/191 route의 Primary에 gamma `0.80`으로 교체하고,
Al3Ni와 Eutectic Si에 rare branch를 함께 추가했다. 18개 후보를 평가한 결과 rare
추가 gamma `0.10`이 image-wise 기준 최고였다.

| 기준 | 최적 설정 | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---|---|---:|---:|---:|
| image-wise | Primary `0.80` + rare `0.10` | 81.1628% | **78.5386%** | 79.3288% |
| global | Primary `0.80` + rare `0.30` | **81.1937%** | 78.5063% | 79.2854% |

결과:
`$KOMAP_WORKSPACE/outputs/model196_native_rare_blend/results.json`

## 모델 197: native branch gamma 국소 재탐색

모델196 주변에서 Primary replacement gamma `0.70~0.90`, rare mode(Al3Ni+Eutectic,
단일 rare class)와 rare gamma `0.05~0.15`를 80개 조합으로 재탐색했다. image-wise
최적은 Primary `0.75`와 두 rare class 추가 `0.075`였다.

| 기준 | 최적 설정 | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---|---|---:|---:|---:|
| image-wise | Primary `0.75` + rare `0.075` | 81.1673% | **78.5564%** | **79.3477%** |
| global | Primary `0.70` + rare `0.15` | **81.1871%** | 78.5546% | 79.3424% |

model196 대비 image-wise 평균은 `+0.0178%p`, 모델191 대비 `+0.0226%p` 상승했다.
현재 내부 image-wise 제출 후보는 모델197 image-wise 설정으로 갱신한다. Global 기준도
model189의 81.1856%를 소폭 넘어섰지만, 두 metric의 최적 설정은 다르므로 대회 metric
정의에 맞춰 선택한다.

결과:
`$KOMAP_WORKSPACE/outputs/model197_native_narrow_blend/results.json`

## 모델 214: 새 branch의 희귀 클래스 전용 교체 탐색

모델197의 Primary 경로는 고정하고, Al3Ni/Eutectic Si만 모델207(이미지 IoU
fine-tune), 모델209(boundary/Tversky), 모델211(test-domain distillation) 확률로
교체하는 156개 조합을 정확한 spatial refinement 경로로 평가했다. 새 branch를 희귀
클래스에 붙이는 모든 경우보다 기존 model195 rare branch가 최선이었다.

| 기준 | 최적 설정 | Global mIoU | Image-wise 평균 | empty 제외 image-wise | 판정 |
|---|---|---:|---:|---:|---|
| 전체 기준 | Primary model195 `0.75`, Al3Ni/Eutectic model195 `0.075` | 81.1673% | **78.5564%** | **79.3477%** | 미채택, model197과 동일 |

결과:
`$KOMAP_WORKSPACE/outputs/model214_rare_branch_probe/results.json`

## 모델 216: model195 native branch 픽셀 잔차 보정

model195를 고정한 뒤 train split의 라벨로 작은 residual MLP를 학습했다. 입력은
모델 확률 logit, grayscale, local contrast, gradient, entropy, normalized coordinate이며,
valid에서는 보정 강도 alpha를 `0~1.0`으로 탐색했다. 학습된 보정은 train에는 맞았지만
valid에서 기준 alpha `0.0`보다 모두 낮아 branch로 채택하지 않았다.

| alpha | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---:|---:|---:|---:|
| 0.0 (기준) | 79.0904% | 76.1109% | 76.6552% |
| 0.1 | 78.7172% | 75.7622% | 76.2942% |
| 0.2 | 78.2788% | 75.3276% | 75.8467% |

결과:
`$KOMAP_WORKSPACE/outputs/model216_pixel_refiner_native/summary.json`

## 모델 217: model195 native 표현의 label-free BN-affine TENT

모델195의 네트워크 가중치는 고정하고, 각 valid 이미지의 네 모서리 patch에서
BatchNorm affine 파라미터만 entropy minimization으로 1/3/5 step 적응했다. 입력 표현은
raw grayscale, per-image z-score, empirical CDF이며, valid 전체를 이미지별로 독립
적응했다. standalone branch에서는 baseline보다 상승했지만 model197 route에 연결하기
전에는 별도 검증이 필요하다.

| 기준 | 설정 | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---|---|---:|---:|---:|
| image-wise | 5 step, lr `2e-5` | 79.6656% | **76.7163%** | 76.7163% |
| empty 제외 | 1 step, lr `2e-5` | 79.2963% | 76.3116% | **76.8635%** |
| model195 기준 | no adaptation | 79.0904% | 76.1109% | 76.6552% |

결과:
`$KOMAP_WORKSPACE/outputs/model217_native_tent/results.json`

## 모델 219: TENT rare-class route 결합

model217의 image-wise 최적 5-step TENT branch를 model195와 결합했다. model195를
Primary에 유지하고, Al3Ni와 Eutectic Si 각각에 TENT 확률을 20% 추가한 조합이 exact
spatial refinement 평가에서도 최고였다.

| 기준 | 설정 | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---|---|---:|---:|---:|
| image-wise/empty 제외 | Primary model195 `0.65`; Al3Ni/Eutectic model217 `0.20` | **81.2407%** | **78.6088%** | **79.3963%** |
| model197 기준 | Primary model195 `0.75`; rare model195 `0.075` | 81.1673% | 78.5564% | 79.3477% |

image-wise 평균은 model197 대비 `+0.0524%p` 상승했다. 다만 이 선택은 20장 valid의
후처리/branch 조합 탐색에서 나온 값이므로 독립 test 점수는 아니다. 결과 파일의
상위 33개 exact spatial refinement 평가 안에서도 위 설정이 image-wise 최고였다.

결과:
`$KOMAP_WORKSPACE/outputs/model219_tent_branch_fusion/results.json`

## 모델 220: native TENT 적응 step/LR 확장 탐색 (부분 완료)

model195 native 표현의 BN-affine TENT에서 10/20-step 설정을 비교했다. 단독 branch는
10-step, lr `5e-5`가 최고였지만 image-wise mIoU `76.8408%`로 model217의 5-step
최고 `76.7163%`보다 `+0.1245%p`에 그쳤다. 20-step 설정은 모두 10-step 최고보다 낮아
추가 30-step 탐색을 중단했다. 별도 결합 탐색은 계속했다.

| TENT 설정 | Global mIoU | Image-wise 평균 | empty 제외 |
|---|---:|---:|---:|
| 10 step, lr `1e-5` | 79.6695% | 76.7964% | 76.7964% |
| 10 step, lr `2e-5` | 79.6812% | 76.8057% | 76.8057% |
| **10 step, lr `5e-5`** | **79.7236%** | **76.8408%** | **76.8408%** |
| 20 step, lr `1e-5` | 79.4676% | 76.7003% | 76.7003% |
| 20 step, lr `2e-5` | 79.4849% | 76.7202% | 76.7202% |
| 20 step, lr `5e-5` | 79.5492% | 76.7921% | 76.7921% |

결과는 `$KOMAP_WORKSPACE/outputs/model220_native_tent_extended/run.log` 및
완료된 설정의 `valid_probs_*.npz`에 기록했다.

## 모델 221: 10-step TENT rare-branch 결합

모델220의 10-step, lr `2e-5` TENT branch를 model195 기반 route와 결합했다. 210개
cheap 후보와 33개 exact spatial refinement 평가에서 rare-class TENT 혼합량은 기존과
같이 Al3Ni/Eutectic Si 각각 `0.20`이었고, Eutectic edge blend를 `0.25`에서 `0.15`로
낮춘 조합이 최고였다.

| 선택 기준 | 설정 차이 | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---|---|---:|---:|---:|
| image-wise | Primary gamma `0.65`, Eutectic edge blend `0.15`, Al bias `0.0` | 81.2386% | **78.6226%** | **79.4104%** |
| global | Primary gamma `0.85`, Al bias `+0.10`, Eutectic edge blend `0.25` | **81.2847%** | 78.6047% | 79.3941% |

image-wise 개선 폭은 모델219 대비 `+0.0138%p`로 작으며, validation 20장에 대한
탐색 결과다.

결과: `$KOMAP_WORKSPACE/outputs/model221_tent10_branch_fusion/results.json`

10-step lr `5e-5` branch의 희귀 혼합 gamma를 `0.40`까지 늘린 모델224가 image-wise
`78.6610%`로 갱신했다. 더 높은 gamma(model226)와 20-step branch gamma 탐색(model225)은
20-step 탐색은 모델225에서 `78.6921%`를 기록했다. 추가 10-step gamma(model226)와
10/20-step 확률 평균 branch(model228)는 진행 중이다.

## 모델 222: 10-step TENT lr 확장 branch 결합

모델220의 단독 최고 branch(10-step, lr `5e-5`)를 model195 기반 route와 결합했다.
희귀 클래스 혼합량은 Al3Ni/Eutectic Si 각각 `0.20`이며, image-wise route의
Eutectic edge blend는 `0.15`다.

| 선택 기준 | Global mIoU | Image-wise 평균 | empty 제외 image-wise | 모델221 대비 |
|---|---:|---:|---:|---:|
| image-wise | 81.2429% | **78.6257%** | **79.4136%** | **+0.0031%p** |
| global | **81.2906%** | 78.6086% | 79.3982% | global `+0.0059%p` |

image-wise 최고는 모델221과 동일하게 rare gamma `0.20`이었고, 이후 모델224에서
혼합량을 확장한 설정이 이 점수를 추가로 높였다.

결과: `$KOMAP_WORKSPACE/outputs/model222_tent10_lr5e5_branch_fusion/results.json`

## 모델 223: 20-step TENT rare-branch 결합

단독 image-wise 성능은 10-step branch보다 낮았던 20-step, lr `5e-5` TENT 출력을
rare-class branch로 결합했다. 단독 점수만으로는 보이지 않았던 상보성이 있어 모델222를
넘었다. image-wise 최적은 rare gamma `0.20`, Primary gamma `0.65`, prior beta scale
`0.70`, Eutectic edge blend `0.25`였다.

| 선택 기준 | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---|---:|---:|---:|
| image-wise | 81.2577% | **78.6469%** | **79.4336%** |
| global | **81.2952%** | 78.6311% | 79.4205% |

모델222 대비 image-wise `+0.0212%p` 상승했다. 10-step lr `5e-5`의 혼합 강도
확장(model224)에서 추가 상승했다.

결과: `$KOMAP_WORKSPACE/outputs/model223_tent20_rare_branch_fusion/results.json`

## 모델 224: 10-step TENT rare 혼합 강도 확장

모델222의 10-step, lr `5e-5` TENT branch를 고정하고 rare-class gamma 탐색 범위를
`0.15~0.40`으로 확장했다. 275개 평가 기록(242 cheap + 33 exact) 중 최고는
Al3Ni/Eutectic Si 양쪽 gamma `0.40`으로, 기존 상한 `0.20`을 넘었다.

| 선택 기준 | Primary gamma | Rare gamma | prior / beta | Global mIoU | Image-wise 평균 | empty 제외 |
|---|---:|---:|---|---:|---:|---:|
| image-wise | 0.75 | 0.40 / 0.40 | 1.20 / 0.80 | 81.3181% | **78.6610%** | **79.4423%** |
| global | 0.75 | 0.40 / 0.40 | 1.25 / 0.80, Al bias +0.10 | **81.3542%** | 78.6583% | 79.4412% |

image-wise는 모델223보다 `+0.0141%p` 상승했다. 20-step branch와 gamma 확장 조합인
모델225는 추가로 `78.6921%`까지 올렸다. 모델226에서는 10-step branch의 gamma를
`0.40~0.80`으로 더 높였지만 image-wise 최고는 `78.6547%`로 낮아졌다.
10/20-step 확률 평균 branch(model228)는 평가 중이다.

결과: `$KOMAP_WORKSPACE/outputs/model224_tent_rare_gamma_expansion/results.json`

## 모델 225: 20-step TENT rare 혼합 강도 확장

모델223에서 사용한 20-step, lr `5e-5` TENT branch의 rare gamma를 `0.15~0.40`으로
넓혔다. 최고 image-wise는 두 rare class gamma `0.40`, Primary gamma `0.75`, prior
scale `1.25`, beta `0.80`, Eutectic bias `+0.15`, edge blend `0.25`였다.

| 선택 기준 | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---|---:|---:|---:|
| image-wise | 81.3033% | **78.6921%** | **79.4693%** |
| global | **81.3495%** | 78.6824% | 79.4636% |

모델224의 image-wise 최고보다 `+0.0311%p` 상승했다. gamma `0.40`보다 더 큰 혼합은
아직 시험하지 않았다.

결과: `$KOMAP_WORKSPACE/outputs/model225_tent20_rare_gamma_expansion/results.json`

## 모델 226: 10-step TENT rare gamma 고강도 확장

10-step lr `5e-5` branch에서 rare gamma를 `0.40~0.80`으로 추가 탐색했다. image-wise
최고는 다시 gamma `0.40`에서 나왔고, gamma `0.50`의 global 최고도 기존 model224보다
낮았다. gamma를 더 높이는 방향은 채택하지 않는다.

| 선택 기준 | Rare gamma | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---|---:|---:|---:|---:|
| image-wise | 0.40 / 0.40 | 81.2946% | 78.6547% | 79.4363% |
| global | 0.50 / 0.50 | 81.3467% | 78.6405% | 79.4191% |

결과: `$KOMAP_WORKSPACE/outputs/model226_tent10_rare_gamma_high/results.json`

## 모델 227: TENT adaptation-length probability soup

model220의 10-step/lr `5e-5`와 20-step/lr `5e-5` validation probability를 50:50으로
평균하고 클래스 합을 재정규화했다. 20장 모두 생성됐고 최대 probability-sum 오차는
`2.4e-7`이었다. 단독 image-wise mIoU는 `76.8400%`로 10-step branch와 사실상 같았다.

결과: `$KOMAP_WORKSPACE/outputs/model227_tent_probability_soup/valid_probs.npz`

## 모델 228: snapshot-soup rare-branch 결합

227 probability soup를 model195 기반 route와 결합하고 rare gamma `0.20~0.60`을
탐색했다. image-wise 최고는 Primary gamma `0.85`, 두 rare gamma `0.50`, prior `1.25`,
beta `0.80`, Eutectic bias `+0.15`였다.

| Global mIoU | Image-wise 평균 | empty 제외 image-wise | 모델225 대비 |
|---:|---:|---:|---:|
| 81.3109% | 78.6709% | 79.4445% | `-0.0212%p` |

snapshot soup는 모델225를 넘지 못해 제출 후보로 채택하지 않는다.

결과: `$KOMAP_WORKSPACE/outputs/model228_tent_snapshot_soup_fusion/results.json`

## 모델 229: 전체 이미지 공유 TENT (실패)

20장 validation의 입력만 사용해 shared BN-affine TENT를 5 global steps, lr `1e-5`로
적용했다. image-wise mIoU는 `75.7464%`로 model195 단독 `76.1109%`보다 낮았다.
checkpoint 대비 모든 BN affine weight/bias 변화가 정확히 `0`이었지만 running mean/var는
크게 이동했고, 후반 image의 순차 BN 통계가 결과를 악화시킨 것으로 확인됐다.

결과: `$KOMAP_WORKSPACE/outputs/model229_global_tent/`

## 모델 230: 누적 global AdaBN (미채택)

모델229의 마지막 이미지 편향을 줄이도록 BN running statistics를 validation 20장에 대해
cumulative average로 다시 계산했다. affine 파라미터는 고정했다. image-wise mIoU는
`75.6943%`로 개선되지 않았다. 이 확률을 model195 rare branch에 gamma `0.075/0.20/0.40`
혼합해 exact-evaluate한 최고도 `78.5424%`로 model197의 `78.5564%`보다 낮았다.

결과: `$KOMAP_WORKSPACE/outputs/model230_global_adabn/`

## 모델 231: per-image D4-view TENT (미채택)

model195에서 각 validation 이미지의 네 corner tile을 8개 D4 변환으로 적응했다.
entropy loss는 변환별 평균이고 BN moments는 한 이미지의 모든 변환에 대해 누적 평균했다.
단독 image-wise는 `76.6154%`로 per-image native TENT 최고보다 낮았다.

이 branch를 model195 rare class에 gamma `0.075/0.20/0.40`으로 exact 결합했을 때
최고는 gamma `0.40`의 `78.6677%`였다. 모델225(`78.6921%`)를 넘지 못해 채택하지 않는다.

| 평가 | Global mIoU | Image-wise 평균 | empty 제외 |
|---|---:|---:|---:|
| 단독 D4-TENT | 79.2903% | 76.6154% | 76.6154% |
| model195 + D4-TENT rare gamma 0.40 | 81.2784% | 78.6677% | 79.4415% |

결과: `$KOMAP_WORKSPACE/outputs/model231_d4_tent/`

## 모델 232: D4 marginal-entropy TENT (단독 미채택; branch 결합 평가 중)

model195 checkpoint를 초기값으로 사용해 valid 이미지마다 네 corner tile을 8개 D4
변환에 통과시켰다. 변환별 확률을 원 방향으로 정렬한 뒤 marginal prediction의 entropy를
최소화하도록 BN affine 파라미터를 3 step, lr `2e-5`로 적응했다. activation 메모리를
줄이기 위해 marginal entropy gradient를 계산한 뒤 view별 graph를 replay하는 방식을 썼다.

| 평가 | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---|---:|---:|---:|
| 단독 model232 | 79.3212% | 76.6507% | 76.6507% |
| image-wise 최고 model225 | 81.3033% | **78.6921%** | **79.4693%** |

단독 branch 점수는 model225보다 `2.0414%p` 낮아 단독 채택하지 않는다. 독립 점수만으로
상보성을 배제하지 않기 위해 model232 probability를 model195 기반 route의 rare-class
source로 바꿔 넣은 model233 결합 평가도 수행했다. 최고점은 model225보다 낮았다.

model232 class별 global IoU는 Al3Ni `77.8663%`, Primary Si `87.3274%`, Al `93.5485%`,
Eutectic Si `58.5424%`다. 예측 probability와 단독 평가는
`$KOMAP_WORKSPACE/outputs/model232_d4_marginal_tent/`에 저장했다.

## 모델 233: model232 D4 branch rare-class 결합 (미채택)

model232 probability를 model195 기반 route의 Al3Ni/Eutectic Si branch에 넣고 rare gamma,
Primary gamma, intensity prior calibration 및 Eutectic edge/bias 설정을 탐색했다. 최고는
두 rare class gamma `0.40`, Primary gamma `0.75`, prior `1.25`, beta `0.80`, Eutectic
bias `+0.15`, edge blend `0.25`였다.

| 모델 | Global mIoU | Image-wise 평균 | empty 제외 image-wise | model225 대비 |
|---|---:|---:|---:|---:|
| model233 | 81.2758% | 78.6796% | 79.4511% | `-0.0125%p` |
| **model225 최고** | **81.3033%** | **78.6921%** | **79.4693%** | — |

따라서 model232 branch는 보완 source로도 채택하지 않는다. 상세 후보 기록은
`$KOMAP_WORKSPACE/outputs/model233_d4_marginal_tent_fusion/results.json`에 있다.

## 모델 234: model195 native checkpoint 저학습률 supervised continuation

model195 best checkpoint에서 이어서 1024 crop, 8 patches/image, batch 2로 4 native-stage
epoch을 학습했다. encoder/decoder learning rate는 `1e-5/3e-5`, seed `2029`였고,
checkpoint 선택 metric은 대회 목표에 맞춰 image-wise mIoU(include empty class)로 설정했다.

| Epoch | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---:|---:|---:|---:|
| 1 | 78.9743% | 75.8140% | 76.3610% |
| 2 | 79.0028% | 75.8026% | 76.3427% |
| **3 (best)** | **79.5660%** | **76.4220%** | **76.9816%** |
| 4 | 79.0099% | 75.8386% | 76.3832% |

epoch3 checkpoint는 model195 단독 image-wise `76.1109%`보다 `+0.3111%p` 높다. 다만
최고 앙상블 route보다 낮아 단독 대체 branch로 채택하지 않고, model235에서 보완성을
별도 평가했다.

결과: `$KOMAP_WORKSPACE/outputs/model234_native_low_lr_continue/`

## 모델 235: model234 supervised branch exact route probe (미채택)

model234 probability를 model195 기반 route와 결합해 16개 조합을 exact spatial refinement까지
평가했다. Primary source/gamma와 Al3Ni·Eutectic Si의 model234 혼합량을 바꾸고 기존 최고
route의 prior/bias/edge calibration은 고정했다.

| 최고 기준 | 설정 요약 | Global mIoU | Image-wise 평균 | empty 제외 |
|---|---|---:|---:|---:|
| image-wise | Primary model195 `0.75`; Al3Ni model234 `0.20`; Eutectic model234 `0.40` | 81.2596% | **78.5801%** | **79.3591%** |
| global | Primary model195 `0.75`; 두 rare class model234 `0.40` | **81.2850%** | 78.5352% | 79.3008% |
| model225 최고 | Primary model195 `0.75`; 20-step TENT rare branch `0.40` | 81.3033% | **78.6921%** | **79.4693%** |

model235 image-wise 최고는 model225보다 `-0.1120%p` 낮다. 따라서 model234의 supervised
continuation은 단독 checkpoint로는 model195보다 나아졌지만, 이번 exact route에서 보완
효과를 보이지 않아 현재 ensemble에는 채택하지 않는다.

구현: `model235_native_branch_probe.py`
결과: `$KOMAP_WORKSPACE/outputs/model235_native_branch_probe/results.json`

## 모델 236: MatSSL-inspired ResNet34 비지도 도메인 적응

model195의 ResNet34 인코더를 시작점으로 사용했다. train split 이미지 70장만 사용해
256×256 patch 쌍을 만들고, 다단계 feature를 sigmoid-gated pooling으로 결합한 뒤
NT-Xent contrastive loss로 30 epoch 사전학습했다. 마스크 라벨은 이 단계에서 사용하지
않았고 valid/test 이미지도 읽지 않았다. 최저 contrastive loss는 epoch 27의 `0.03731`이다.
이는 MatSSL 논문의 metallography 영역 아이디어를 차용한 변형이며, 논문 설정을 그대로
재현한 것은 아니다.

결과: `$KOMAP_WORKSPACE/outputs/model236_matssl_ssl/`
구현: `model236_matssl_inspired_ssl.py`

## 모델 237: model236 인코더 이식 후 supervised fine-tune (미채택)

model236 best encoder의 216개 텐서를 model195 segmentation checkpoint에 이식한 뒤,
1024 patch, 8 patches/image, batch 2, 4 epoch으로 지도 미세조정했다. encoder/decoder LR은
`1e-5/3e-5`이며, valid image-wise mIoU를 checkpoint 선택 기준으로 썼다.

| Epoch | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---:|---:|---:|---:|
| 1 | 78.6663% | 75.4692% | 76.0065% |
| 2 | 78.8499% | 75.7168% | 76.2568% |
| 3 | 78.7754% | 75.7391% | 76.2776% |
| **4 (best)** | **79.4761%** | **76.2376%** | **76.8022%** |

model234의 지도 continuation best `76.4220%`보다 단독 성능이 `0.1844%p` 낮다. 클래스별
global IoU는 Al3Ni `77.3257%`, Primary Si `88.5090%`, Al `93.4469%`, Eutectic Si `58.6229%`다.

결과: `$KOMAP_WORKSPACE/outputs/model237_matssl_finetune/`
초기화 도구: `model237_transfer_ssl_encoder.py`

## 모델 238: model237 분기 16개 exact route probe (미채택)

model237 확률을 model195 기반 route와 결합해 Primary source/gamma 및 두 rare class 혼합률
조합 16개를 실제 spatial refinement까지 적용해 valid에서 평가했다. image-wise 최고는
Primary model195 `0.85`, Al3Ni model237 `0.20`, Eutectic Si model237 `0.40`이었다.

| 평가 | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---|---:|---:|---:|
| model238 best image-wise route | 81.2253% | 78.5496% | 79.3288% |
| model225 현재 최고 | 81.3033% | **78.6921%** | **79.4693%** |

model238은 model225 대비 image-wise 기준 `-0.1425%p`이므로 현재 최고 route에는 넣지 않는다.
후보 전체 기록은 `$KOMAP_WORKSPACE/outputs/model238_matssl_branch_probe/results.json`에
저장했다.

구현: `model238_matssl_branch_probe.py`

## 모델 239: zero-init stride-1 native-resolution stem (결합 평가 중)

팀원 보고서에서 제안한 32채널 native-resolution feature 보존 아이디어를 적용했다.
model195의 ResNet34/U-Net++/ASPP 경로와 pretrained weight는 그대로 두고, 입력 해상도에서
동작하는 3→32→32 convolution branch를 추가해 마지막 decoder head에 주입했다. 새 주입
projection은 0으로 초기화해 학습 시작 시 기존 모델의 출력이 정확히 보존되도록 했다.
검증은 기존처럼 8-way D4 TTA와 Gaussian overlap blending을 사용했다.

1024 patch, 8 patches/image, batch 2, 4 epoch으로 미세조정하고 image-wise mIoU로 선택했다.

| Epoch | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---:|---:|---:|---:|
| 1 | 79.7670% | 76.7253% | 76.7253% |
| **2 (best)** | **79.6581%** | **76.7430%** | **76.7430%** |
| 3 | 79.3088% | 76.3796% | 76.3796% |
| 4 | 79.2686% | 76.3518% | 76.3518% |

단독 image-wise 기준으로 model234보다 `+0.3210%p`, model195보다 `+0.6321%p`다. 모델225
route의 추가 branch로 유효한지는 model240 exact 결합 평가 후 확정한다.

결과: `$KOMAP_WORKSPACE/outputs/model239_native_stem/`
구현: `model239_native_stem_finetune.py`

## 모델 240: model239 rare-class branch 결합 (미채택)

model239 probability를 model195 기반 route에 넣고 16개 exact 후보를 평가했다. 최선은
Primary model195 `0.75`, Al3Ni model239 `0.20`, Eutectic Si model239 `0.40`이었다.

| Global mIoU | Image-wise 평균 | empty 제외 image-wise | model225 대비 image-wise |
|---:|---:|---:|---:|
| 81.2663% | 78.6186% | 79.3967% | `-0.0735%p` |

단독 분기의 상승이 최고 조합의 상승으로 이어지지 않아 model225 route에 채택하지 않았다.
결과: `$KOMAP_WORKSPACE/outputs/model240_native_stem_branch_probe/results.json`

## 모델 241: Al3Ni/Eutectic class-wise TENT·native stem 혼합 (미채택)

model225의 20-step TENT 확률과 model239 확률을 각각 Al3Ni/Eutectic Si source로 독립 선택해
32개 exact 후보를 비교했다. image-wise 최고는 기존 model225와 동일하게 두 rare class 모두
TENT source `217`, gamma `0.40`이었고, 즉 native stem은 주 지표를 개선하지 못했다.

| 선택 기준 | 설정 | Global mIoU | Image-wise 평균 | empty 제외 |
|---|---|---:|---:|---:|
| image-wise | Primary 195 `0.75`; Al3Ni/Eutectic TENT `0.40/0.40` | 81.3033% | **78.6921%** | **79.4693%** |
| global | Primary 195 `0.85`; Al3Ni 239 `0.40`; Eutectic TENT `0.40` | **81.3367%** | 78.6621% | 79.4359% |

global 기준 후보도 model225의 global 최고 `81.3495%`를 넘지 못했다. 따라서 model239를
현재 최고 앙상블에 포함하지 않는다.
결과: `$KOMAP_WORKSPACE/outputs/model241_classwise_stem_tent_probe/results.json`
구현: `model241_classwise_stem_tent_probe.py`

## 모델 242: model239 low-LR continuation + EMA 0.99

팀원 학습 설정의 EMA `0.99`를 분리 검증했다. model239 best에서 1024 patch, 8
patches/image, batch 2로 4 epoch 이어 학습했으며 encoder/decoder/native branch LR은
`5e-6/1.5e-5/1e-4`였다. 각 epoch에서 raw와 EMA 모델을 모두 D4+Gaussian valid 평가했다.

| 가중치 선택 | Best epoch | Global mIoU | Image-wise 평균 | empty 제외 |
|---|---:|---:|---:|---:|
| raw | 4 | 79.7975% | 76.8283% | 76.8283% |
| **EMA 0.99** | **4** | **79.8448%** | **76.8640%** | **76.8640%** |

EMA가 raw보다 `+0.0357%p`, model239 best보다 `+0.1210%p` 높았지만, 최고 route 보완성은
다음 model243에서 주 지표 향상으로 이어지지 않았다.

결과: `$KOMAP_WORKSPACE/outputs/model242_native_stem_ema/`
구현: `model242_ema_continuation.py`

## 모델 243: model242 EMA와 20-step TENT class-wise 결합 (미채택)

Al3Ni와 Eutectic Si 각각에 model217 TENT 또는 model242 EMA를 선택하도록 32개 exact 후보를
비교했다. image-wise 최고는 model225와 동일한 model217 TENT `0.40/0.40` 조합으로,
model242를 넣어도 최고 image-wise `78.6921%`가 갱신되지 않았다. global 기준 최고는 두
rare class에 model242 EMA `0.40/0.40`, Primary model195 `0.85`였으나 `81.3698% / 78.6250%`
로 model225 global 최고 `81.3495%`보다 global은 `+0.0203%p`, image-wise는 낮았다.

결과: `$KOMAP_WORKSPACE/outputs/model243_classwise_ema_tent_probe/results.json`
구현: `model243_classwise_ema_tent_probe.py`

## 모델 244: model242 EMA teacher supervised distillation (완료)

student는 model239 best, frozen teacher는 model242 best EMA로 초기화했다. train crop에서
동일 입력에 대한 teacher softmax(T=2)를 student에 전달하고, hard-label weighted CE+Dice에
`0.2 × T² × KL`을 더해 4 epoch 학습했다. teacher/student 모두 train split만 사용했고,
valid는 epoch 선택 및 평가에만 사용했다.

| Epoch | Global mIoU | Image-wise 평균 | empty 제외 |
|---:|---:|---:|---:|
| 1 | 79.8082% | 76.7487% | 76.7487% |
| 2 | 79.8452% | 76.7947% | 76.7947% |
| 3 | 79.8210% | 76.7811% | 76.7811% |
| **4 (best)** | **79.8969%** | **76.8332%** | **76.8332%** |

model239 student보다 `+0.0902%p` 올랐고 model242 EMA teacher의 `76.8640%`보다는
`0.0308%p` 낮다. 현재 최고 route에 대한 보완성은 model245에서 평가한다.

결과: `$KOMAP_WORKSPACE/outputs/model244_supervised_distillation/`
구현: `model244_supervised_distillation.py`

## 모델 245: distilled student와 20-step TENT class-wise 결합 (미채택)

model244와 model217 TENT의 Al3Ni/Eutectic source를 독립 선택하는 32개 exact 후보를
평가했다. image-wise 최고는 model225와 동일한 양 class TENT gamma `0.40/0.40`으로
`78.6921%`를 유지했으며, distilled student는 주 지표를 개선하지 못했다. global 최고는
Primary model195 `0.85`, Al3Ni model244 `0.40`, Eutectic model217 `0.40`에서 `81.3635%`
였지만 image-wise는 `78.6709%`로 model225 best보다 `0.0212%p` 낮았다.

결과: `$KOMAP_WORKSPACE/outputs/model245_classwise_distill_tent_probe/results.json`
구현: `model245_classwise_distill_tent_probe.py`

## 모델 246: latest ensemble의 조건부 라우팅 grouped-CV (미채택)

model245 exact 후보 32개를 대상으로 raw 밝기·질감, 확률 entropy/margin, 예측 클래스 면적을
조건으로 쓰는 1단계 threshold router를 탐색했다. valid 20장을 A-ID 기준 19개 그룹으로 묶어
leave-one-specimen-group-out으로 route를 학습·평가했으며, GT는 각 fold의 학습 쪽에서만
후보 선택에 사용했다. 평가는 이미지별 네 클래스 IoU 평균(빈 union은 코드상 IoU 0)을 썼다.

전체 valid에서 밝기 threshold `gray_mean < 165.55`이면 model245 기준 route, 그 외에는
distilled model244 rare branch를 쓰는 in-sample route가 `78.7912%`로 고정 model225 route의
`78.6921%`보다 `+0.0991%p` 높아 보였다. 그러나 그룹 held-out score는 `78.6206%`로 고정
route 대비 `-0.0714%p`였고, 그룹 bootstrap 95% 차이 구간 `[-0.1814, +0.1347]%p`가 0을
포함했다. valid에 맞춘 밝기 threshold를 채택할 근거가 없어 미채택했다.

결과: `$KOMAP_WORKSPACE/outputs/model246_grouped_route_cv/results.json`
구현: `model246_grouped_route_cv.py`

## 모델 247: 금속학적 형태 조건을 추가한 grouped-CV router (미채택)

모델246 후보·분할 방식을 유지하고, raw 밝기/texture 외에 Primary/Eutectic 예측상의
connected-component 수, 큰 component 수, 경계 밀도, 클래스 가중 gradient 통계를 추가했다.
이는 입자 크기·분산·계면 복잡도 변화에 따른 조건부 전문가 선택 가능성을 진단한다.

전체 valid 최적 rule은 여전히 `gray_mean < 165.55`였으며, 금속학적 형태 피처를 더한
그룹 held-out image-wise mIoU는 `78.5775%`로 model225 고정 route 대비 `-0.1146%p`였다.
그룹 bootstrap 95% 차이 구간 `[-0.2465, +0.1094]%p`도 0을 포함했다. 형태 피처가 held-out
fold에서 일관되게 선택되지 않아 현재는 라우팅에 채택하지 않는다.

결과: `$KOMAP_WORKSPACE/outputs/model247_metallurgical_route_cv/results.json`
구현: `model246_grouped_route_cv.py` (`--feature-set metallurgical`)

## 금속공학 기반 후속 아이디어 (아직 성능 실험 전)

- Al-Si 문헌에서는 미변형 eutectic Si의 판상/flake 형태와 Sr 개질 시 미세 fibrous 형태가
  구별되며, Feret diameter·circularity 분포로 그 차이를 정량화한다. 따라서 학습/추론 시
  이미지별 밝기만이 아니라 Si feature 크기·원형도·연결성 proxy를 이용한 domain/route 진단이
  물리적으로 타당한 다음 가설이다.
- Al-Si-Ni 합금에서 Al3Ni는 응고 조건에 따라 plate-like에서 fishbone 형태로 바뀔 수 있어,
  Al3Ni branch 라우팅은 component의 길이/분기성/방향성 통계를 조건으로 할 수 있다.
- 다상 Al-Si tomography 연구는 Si·aluminide 및 그 계면의 pixel-wise weight를 사용해 어려운
  경계를 보강하고 phase-network connectivity/Euler number까지 평가했다. 우리 데이터는 2D
  광학 이미지라 그대로 재현할 수 없지만, 계면 가중 손실과 object/topology 진단의 근거가 된다.
- 산화/노이즈가 있는 Al-Si micrograph에서는 clean/noisy 데이터 혼합 학습이 noisy 영상 성능을
  개선한 보고가 있어, 촬영·에칭 상태별 augmentation/domain grouping을 추가 검증할 가치가 있다.

참고: [Al-Si Si-phase extraction (2025)](https://doi.org/10.1016/j.mtcomm.2024.111468),
[Al3Ni morphology vs solidification conditions (2020)](https://doi.org/10.1002/adem.202000503),
[Al-Si tomography segmentation with interface-weighted loss (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6928157/),
[Al-Si semantic segmentation and object-measurement comparison (2024)](https://doi.org/10.1007/s11665-024-10442-5).

## 모델 248: model225 최고 route 오류 사례 감사

model225의 원본 검증 확률과 실제 route를 재현해 점수 `78.6921%`가 일치하는지 확인하고,
클래스별·이미지별 오류를 집계했다. 네 클래스의 이미지 평균 IoU는 Al3Ni `75.30%`,
Primary Si `86.07%`, Al `93.75%`, Eutectic Si `59.65%`였다. A71의 Primary Si는 정답과
예측 모두 없어 대회 계산 규칙대로 IoU 0으로 들어가며, A71 최종 점수는 `46.64%`였다.
다음 취약 사례는 A153(`72.71%`, Eutectic Si IoU `33.94%`), A74(`73.97%`, Eutectic Si
IoU `41.61%`)였다. 따라서 다음 학습 가설은 clDice 단독 개선보다는 고해상도에서 침상 경계
정보가 추가되는지 검증하는 것으로 좁혔다.

결과: `$KOMAP_WORKSPACE/outputs/model248_best_route_error_audit/results.json`
구현: `model248_best_route_error_audit.py`

## 모델 249: Eutectic Si clDice 미세조정 (미채택)

model242 EMA 가중치를 시작점으로 train split만 이용해 1024×1024 patch, 이미지당 8 crops,
batch 2, 4 epoch을 학습했다. weighted CE+Dice에 Eutectic Si soft-clDice(`0.10`, 4회
skeletonization)를 추가했다. 최고는 epoch 2의 단독 image-wise `76.6056%`(global `79.6226%`)
로, 기존 주 route보다 낮았다. clDice branch가 Eutectic의 추가 정보를 주는지 model250/252에서
확률 수준으로 따로 확인했다.

결과: `$KOMAP_WORKSPACE/outputs/model249_eutectic_cldice/summary.json`
구현: `model249_eutectic_cldice_finetune.py`

## 모델 250: clDice branch exact fusion (미채택)

기존 주 route와 model249 clDice, model244 distillation, model217 TENT rare-class 확률을 포함한
후보 73개를 exact valid 평가했다. image-wise 최고는 model225 route를 그대로 재현한
`78.6921%`; global 최고 후보는 `81.3635%`였지만 image-wise `78.6709%`로 낮았다. 따라서
clDice branch를 섞는 것만으로는 개선 근거가 없었다.

결과: `$KOMAP_WORKSPACE/outputs/model250_cldice_branch_probe/results.json`
구현: `model250_cldice_branch_probe.py`

## 모델 251: clDice branch coarse calibration shortlist

Eutectic source(20-step TENT, distillation, clDice), gamma와 Eutectic bias를 바꾼 120개 조합을
저비용으로 선별했다. coarse 점수 1위는 model244 source, gamma `0.40`, bias `+0.15`의
`78.5297%`였고, 상위 후보를 다음 exact 단계로 넘겼다. 이 단계 점수는 후보 선별용 근사치이며
최종 성능으로 채택하지 않는다.

구현·원시 기록: `model252_cldice_calibration_search.py`,
`$KOMAP_WORKSPACE/outputs/model252_cldice_calibration_search/results.json`

## 모델 252: clDice branch exact calibration (미채택)

coarse 상위 후보 24개와 model225 기준 route를 동일한 exact evaluator로 비교했다. 최고
image-wise는 Eutectic source model249 gamma `0.40`, Al3Ni source model217 gamma `0.40`,
Primary model195 gamma `0.75` 조합의 `78.6986%`(global `81.3425%`)였다. model225 기준
`78.6921%`보다 겨우 `+0.0065%p`라 seed·검증 표본 변동보다 훨씬 작고, 실질적인 성능 개선으로
판정하지 않았다. 현재 채택 route는 계속 model225다.

결과: `$KOMAP_WORKSPACE/outputs/model252_cldice_calibration_search/results.json`
구현: `model252_cldice_calibration_search.py`

## 모델 253: 고정 3배 해상도 fine-tuning (미채택)

팀원 `~/Downloads/EXPERIMENTS.md`는 EffNet-B4에서 crop 기준 2배 확대 `78.19%`, 3배 확대
평균 `78.79%`, 4배 `79.01%`를 보고한다. 3배가 2배보다 일관되게 높았지만 4배와의 차이는
시드 편차보다 작다고 명시되어 있다. 다만 해당 문서는 빈 클래스를 image 평균에서 제외하는
정책을 주 지표로 사용하므로, 빈 클래스를 IoU 0으로 포함하는 우리 공식 목표 수치와 직접
비교할 수 없다. 우리 기존 1024 patch는 짧은 변이 1024보다 작을 때만 확대하므로 실제 확대율이
이미지별로 달라, 이 결과를 직접 재현하지 못했다.

이에 model242 EMA를 시작점으로 원본 약 480–512px 시야에 해당하는 고정 3배 확대,
1536×1536 patch, 이미지당 4 crops, batch 1로 3 epoch fine-tuning했다. 최고 checkpoint는
epoch 2였으며 D4 TTA·stride 864 평가에서 image-wise mIoU `74.9705%`, global mIoU
`77.3867%`였다. 이는 기존 model242 EMA의 image-wise `76.8640%`보다 낮다. model253의
확률 지도를 생성해 class-wise fusion 가능성을 별도로 검증했다.

결과: `$KOMAP_WORKSPACE/outputs/model253_highres_scale/summary.json`
구현: `model253_highres_scale_finetune.py`

## 모델 254: 고해상도 분기 class-wise fusion 및 시편 그룹 교차검증 (미채택)

model253 3배 분기와 model255의 matched 1배 대조 분기를 기존 최고 model225 route에
클래스별 30%·50% 혼합하는 사전 지정 후보 35개를 exact evaluator로 비교했다. 검증 누수를
줄이기 위해 동일 시편(A###)의 이미지들을 한 그룹으로 묶어 leave-one-specimen-out 선택도 했다.
새 분기를 포함한 후보 중 최고는 Al3Ni만 model255를 30% 섞은 `78.5902%`였고,
model225 기준 `78.6921%`보다 `-0.1020%p` 낮았다. model253 3배 분기의 최고 후보도
`78.5871%`에 그쳤다. 19개 held-out 시편 모두에서 기존 model225 route가 선택되어
교차검증 개선량은 `0.0000%p`(그룹 bootstrap 95% CI `[0.0000, 0.0000]`)였으며,
개선된 이미지도 없었다. 따라서 해상도 분기는 현재 앙상블에 넣지 않는다.

결과: `$KOMAP_WORKSPACE/outputs/model254_highres_route_probe/results.json`
구현: `model254_highres_route_probe.py`

## 모델 255: matched 1배 해상도 대조 fine-tuning

고해상도 효과와 학습량 효과를 분리하기 위해 model253과 같은 model242 EMA 초기화,
seed 253, 이미지당 4 crops, batch 1, 3 epoch, scheduler horizon 4를 사용하고 patch를
512×512, scale을 1배로 맞췄다. model253의 1536px/3배 crop과 원본상 시야가 비슷하도록
설정한 대조군이다. 최고 epoch 3에서 image-wise mIoU `75.4500%`, empty-class 제외
image-wise `76.0248%`, global mIoU `78.8276%`를 얻었다. model253보다 `+0.4796%p` 높아
이번 짧은 fine-tuning 조건에서는 3배 입력이 유리하지 않았다. 다만 두 분기 모두 model242
EMA의 `76.8640%` 및 최고 model225 route의 `78.6921%`보다 낮아 채택하지 않았다.

또한 scale 1 / patch 1024 / stride 576 설정의 model253 추론 코드 결과를 기존 model242
EMA 확률과 대조했다. 20개 valid 이미지에서 probability 최대 차이와 평균 차이는 모두 `0`,
argmax 일치율은 `100%`, 공식 image-wise 점수도 양쪽 모두 `76.8640%`로 추론 경로 parity를
확인했다.

결과: `$KOMAP_WORKSPACE/outputs/model255_matched_1x_control/summary.json`
parity 결과: `$KOMAP_WORKSPACE/outputs/model256_inference_parity/summary.json`

## 모델 257: 전체 이미지 통계 기반 1배 fine-tuning

Opus 5.5 검토에서 제기된 학습·추론 표현 통계 불일치를 직접 시험했다. 기존 학습은 각 crop에서
z-score/CDF를 다시 계산하는 반면 추론은 전체 resized image에서 계산한다. model242 EMA를
시작점으로 1024×1024 patch, 이미지당 8 crops, batch 2, 동일 학습률·4 epoch을 유지하고,
학습 시 전체 resized image의 통계를 먼저 계산한 뒤 patch를 뽑도록 변경했다. 최고 epoch 4에서
image-wise mIoU `77.2071%`, global mIoU `79.9627%`로 model242 EMA의 `76.8640%`보다
`+0.3431%p` 개선됐다. 정규화 일치가 효과가 있을 수 있음을 보였지만, 단독 성능은 현재 최고
route보다 낮다.

결과: `$KOMAP_WORKSPACE/outputs/model257_full_image_stats_1x/summary.json`
구현: `model253_highres_scale_finetune.py --global-representation`

## 모델 258: model257 class-wise route probe 및 시편 그룹 CV (미채택)

model225의 Primary·Al3Ni 설정은 고정하고, model257 확률을 Al3Ni/Eutectic Si/Primary의
부분집합별로 30% 또는 50% 혼합한 14개 exact 후보를 평가했다. in-sample 최고는 Eutectic Si에
model257을 50% 넣은 설정으로 image-wise `78.7154%`, global `81.3674%`였다. model225의
`78.6921%` 대비 raw 상승은 `+0.0233%p`에 불과하다.

19개 시편 그룹 leave-one-specimen-out에서는 평균 변화가 `-0.1266%p`, positive image는
10/20장이었고 그룹 bootstrap 95% CI는 `[-0.3063, +0.0411]%p`로 0을 포함했다. 따라서
검증셋에 맞춘 미세한 후보 순위일 가능성이 높아 route에는 채택하지 않고, 현재 운영 기준은 계속
model225로 둔다. model257의 표현 일치 학습은 유망한 단독 실험이지만, 80% 목표에는 추가 개선이
필요하다.

결과: `$KOMAP_WORKSPACE/outputs/model258_fullstats_route_probe/results.json`
구현: `model258_fullstats_route_probe.py`

## 모델 259: 전체 이미지 통계 기반 2배 해상도 fine-tuning (미채택)

model257과 같은 model242 EMA 초기화, 1024×1024 patch, 이미지당 8 crops, batch 2,
동일 학습률·4 epoch을 유지하고 scale만 1배에서 2배로 높였다. z-score/CDF는 inference와
맞춰 전체 resized image에서 계산했다. epoch별 official image-wise mIoU는 `76.6193%`,
`76.8104%`, `76.7059%`, `76.4081%`였고 epoch 2가 최고였다(global `79.9095%`). model257
1배 full-stat 최고 `77.2071%`보다 `-0.3968%p` 낮아, 이 설정에서도 입력 배율 상승은 이득이
없었다.

결과: `$KOMAP_WORKSPACE/outputs/model259_fullstats_2x/summary.json`
학습 구현: `model253_highres_scale_finetune.py --scale 2 --global-representation`

## 모델 260: model259 class-wise route probe 및 시편 그룹 CV (미채택)

model259를 각 class 부분집합에 30% 또는 50% 혼합하는 14개 exact 후보를 비교했다. image-wise
최고는 model225 reference `78.6921%` 그대로였고, 19개 시편 그룹 교차검증도 모든 fold에서
model225를 선택했다(held-out delta `0.0000%p`, 95% CI `[0.0000, 0.0000]`). global 기준
후보는 Al3Ni에 model259 50%를 섞은 `81.3242%`였지만 model225 global `81.3033%`보다
`+0.0209%p`일 뿐이며 image-wise는 `78.6424%`로 하락했다. model259 확률은 현재 route에
추가하지 않는다.

결과: `$KOMAP_WORKSPACE/outputs/model260_2x_route_probe/results.json`
구현: `model258_fullstats_route_probe.py --source-id 259`

## 모델 261: 전체 이미지 통계 기반 독립 seed 2 (미채택)

model257과 같은 model242 EMA 초기화·1배/1024 patch·8 crops/image·batch 2·학습률·4 epoch을
유지하고 seed만 `261`로 바꿨다. 최고는 epoch 3의 image-wise `77.0962%`, global `79.8903%`로,
model257 seed257의 최고 `77.2071%`보다 `-0.1109%p` 낮았다.

결과: `$KOMAP_WORKSPACE/outputs/model261_fullstats_seed261/summary.json`

## 모델 262: model257/261 2-seed 확률 평균 (미채택)

두 seed의 원본 해상도 class probability를 동일 가중치로 평균하고 재정규화했다. 단독 결과는
image-wise `77.1609%`, global `79.9299%`로 최고 단일 seed model257보다 `-0.0462%p` 낮았다.
단순 seed averaging은 이 설정에서 개선을 보이지 않았다.

결과: `$KOMAP_WORKSPACE/outputs/model262_fullstats_seed_ensemble/summary.json`
구현: `model262_seed_ensemble_probe.py`

## 모델 263: model262 seed ensemble class-wise route 및 specimen-group CV (미채택)

model262를 30%·50% 비율로 class 부분집합에 넣은 14개 exact 후보를 비교했다. raw image-wise
최고는 model225 대비 `+0.0186%p`인 `78.7107%`였지만, 19개 시편 그룹 held-out 평균 변화는
`-0.1815%p`였고 95% bootstrap CI `[-0.4041, -0.0043]%p`가 전부 음수였다. 20장 중 9장만
향상돼 seed ensemble을 route에 포함하지 않는다.

결과: `$KOMAP_WORKSPACE/outputs/model263_seed_ensemble_route_probe/results.json`
구현: `model258_fullstats_route_probe.py --source-id 262`

## 모델 264: model201 gradient/texture branch class-wise route (미채택)

별도 gradient 표현으로 학습한 model201은 단독 image-wise `75.4525%`였으며, 이를 model225
route에 class별로 섞는 14개 exact 후보를 비교했다. 모든 후보와 19개 specimen-group fold에서
model225 reference가 최선으로 선택되어 유효한 보완 효과가 확인되지 않았다.

결과: `$KOMAP_WORKSPACE/outputs/model264_gradient_route_probe/results.json`
구현: `model258_fullstats_route_probe.py --source-id 201`

## 모델 265: 전체 이미지 통계 기반 3배 해상도 fine-tuning

기존 model253의 3배/1536 patch/이미지당 4 crops/batch 1/3 epoch 설정을 유지하고,
학습 patch 추출 전에 전체 resized image에서 z-score/CDF 표현을 계산하도록 수정했다.
이는 기존 3배 학습의 crop-local 통계와 inference의 full-image 통계 불일치를 제거한다.
epoch별 official image-wise mIoU는 `74.3811%`, `75.0537%`, `75.0410%`였고
epoch 2가 최고(global `77.5224%`)였다. 기존 crop-local model253 최고 `74.9705%`보다
`+0.0832%p` 개선됐지만, full-stat 1배 model257 `77.2071%`보다 `-2.1534%p` 낮다.
따라서 정규화 불일치는 3배 branch 성능 저하의 일부였지만, 확대 자체의 이득은 확인되지 않았다.

결과: `$KOMAP_WORKSPACE/outputs/model265_highres_fullstats_3x/summary.json`
구현: `model253_highres_scale_finetune.py --scale 3 --global-representation`

## 모델 266: model265 30/50% class-wise route 및 specimen-group CV (미채택)

model265를 각 class 부분집합에 `30%` 또는 `50%` 섞는 14개 exact 후보를 model225 route와
비교했다. 최고 후보는 model225 기준 route 그대로였고 image-wise `78.6921%`를 유지했다.
새 분기를 포함한 최고 후보는 `78.5894%`로 `-0.1026%p` 낮았다. 19개 validation 시편 그룹
leave-one-group-out에서 전 fold가 model225 기준 route를 선택했고 held-out delta는
`0.0000%p`(95% CI `[0.0000, 0.0000]`)였다.

결과: `$KOMAP_WORKSPACE/outputs/model266_highres_fullstats_3x_route/results.json`

## 모델 267: model265 저혼합률 class-wise route 및 specimen-group CV (미채택)

고해상도 branch의 작은 보완 효과를 확인하기 위해 class 부분집합별 혼합률 `5/10/15/20/25%`
후보 35개와 model225 reference를 exact 평가했다. 최고는 여전히 기준 route `78.6921%`였고,
model265를 Primary Si에 5% 섞은 최고 신규 후보는 `78.6225%`(`-0.0696%p`)였다. 19개
specimen-group held-out fold 모두 model225 route를 선택해 채택 근거가 없었다.

결과: `$KOMAP_WORKSPACE/outputs/model267_highres_low_gamma_route/results.json`
구현: `model258_fullstats_route_probe.py --source-id 265 --gammas 0.05 0.10 0.15 0.20 0.25`

## 데이터 분할 specimen overlap 감사

파일명 A-ID 기준 데이터는 train 70장/59시편, valid 20장/19시편, test 10장/10시편이다.
분할은 이미지 단위로는 분리됐지만 시편 단위로는 겹친다: train-valid 7시편
(`A7`, `A40`, `A55`, `A125`, `A153`, `A181`, `A191`), valid-test 1시편(`A63`),
train-test 3시편(`A51`, `A87`, `A167`). 따라서 기존 leave-one-specimen-group-out 결과는
route 파라미터 선택에서 held-out valid group을 분리한 것이며, 해당 시편의 train 이미지가
네트워크 학습에 들어간 7개 train-valid 그룹에 대해서는 완전한 specimen-independent CV가 아니다.
이는 주어진 공식 validation 20장 점수 자체를 무효화하지 않지만, 미관측 시편 일반화 성능의
근거로 grouped route CV를 해석할 때 제한으로 고려해야 한다. 시편 독립 일반화를 검증하려면
각 fold에서 train의 동일 A-ID 이미지도 제외하고 네트워크를 다시 학습해야 하며 아직 수행하지 않았다.

## 모델 268: model265 3배 해상도 continuation (조기 중단)

model265 최고 checkpoint에서 3배/1536 patch 설정을 낮은 학습률로 4 epoch 더 이어가도록
시작했다. 추가 epoch 1은 `75.0316%`, epoch 2는 `74.8117%`로 model265 최고 `75.0537%`를
넘지 못했고 하락해 epoch 3 도중 중단했다. 3배 branch의 undertraining 가설을 지지하지 않아
고해상도 continuation은 더 진행하지 않았다. 남은 체크포인트/기록은 서버에 보존했다.

결과: `$KOMAP_WORKSPACE/outputs/model268_highres_fullstats_3x_extended/history.json`

## 모델 269: model257 1배 full-stat continuation (미채택)

model257 최고 checkpoint에서 1024 patch·8 crops/image·batch 2·full-image 통계 정규화를
유지하고 학습률을 절반으로 낮춰 4 epoch 이어 학습했다. epoch별 image-wise mIoU는
`76.9854%`, `76.9703%`, `76.8878%`, `76.9726%`였다. 최고 epoch 1도 model257의
`77.2071%`보다 `-0.2217%p` 낮아, 1배 branch도 추가 fine-tuning으로 개선되지 않았다.

결과: `$KOMAP_WORKSPACE/outputs/model269_fullstats_1x_extended/summary.json`

## 모델 270: model269 class-wise route 및 specimen-group CV (미채택)

model269를 30%·50% 비율로 class 부분집합에 섞은 14개 exact 후보를 비교했다. raw 최고는
Eutectic Si에 model269를 50% 혼합한 `78.7209%`로 model225보다 `+0.0289%p`였다.
하지만 19개 valid specimen-group held-out 선택 성능 변화는 `-0.1320%p`, positive image
9/20장이었으며 그룹 bootstrap 95% CI `[-0.3548, +0.0574]%p`가 0을 포함했다. 따라서
검증 표본의 작은 순위 변동으로 보고 model225 route를 유지한다.

결과: `$KOMAP_WORKSPACE/outputs/model270_fullstats_extended_route/results.json`

## 모델 271: model265 pixelwise confidence-gated fusion (미채택)

model265가 해당 클래스의 model225 mixed probability보다 높은 픽셀에서만 high-resolution
확률을 반영하는 gate를 Al3Ni/Primary Si/Eutectic Si 부분집합, 확률 margin `0/0.05/0.10`,
blend `0.25/0.50`로 탐색했다. 최고 raw 후보는 Primary Si만 margin `0.05`, blend `0.25`로
반영한 `78.7042%`로 기준 `78.6921%`보다 `+0.0121%p`였고 global mIoU는 오히려 낮았다.
19개 valid specimen-group held-out 선택에서는 평균 변화 `-0.0626%p`, positive image 7/20,
95% CI `[-0.1270, -0.0082]%p`로 전부 음수였다. 따라서 confidence gate는 채택하지 않는다.

결과: `$KOMAP_WORKSPACE/outputs/model271_pixelwise_highres_gate/results.json`
구현: `model271_confidence_gated_highres.py`

## 모델 272: 강한 metallography augmentation 1배 fine-tuning (미채택)

model257 full-image-stat checkpoint에서 scale 1, patch 1024, 이미지당 8 crops, batch 2로
4 epoch fine-tuning했다. contrast/brightness/gamma 변화, blur/noise와 함께 임의 회전 및
0.70~1.40배 affine zoom을 추가했다. epoch별 공식 image-wise mIoU(include empty union=0)는
`76.4414%`, `76.9807%`, `76.4905%`, `76.6927%`였고, 최고 epoch 2도 model257의 단독
`77.2071%`보다 `-0.2264%p` 낮았다. 마지막 epoch까지 회복하지 못해 이 강도의 결합 augmentation은
채택하지 않는다. 데이터 변형 가설을 완전히 기각하는 결과는 아니며, affine·photometric 효과를
분리하려면 이후 한 요인씩 비교해야 한다.

결과: `$KOMAP_WORKSPACE/outputs/model272_microstructure_augmentation_1x/summary.json`
구현: `model253_highres_scale_finetune.py --global-representation --metallography-augmentation`

## 모델 273: 3배 해상도 장기 fine-tuning (미채택)

model257 full-image-stat checkpoint에서 시작해 3배 확대, 1536 patch(원본 환산 약 512px 시야),
이미지당 4 crops, batch 1, full-image z-score/CDF를 유지하고 8 epoch 학습했다. learning rate도
기존 단기 3배 실험과 동일하게 두어 학습량 효과를 우선 확인했다. epoch별 image-wise mIoU는
`74.8187`, `75.0447`, `75.1917`, `75.4014`, `75.5004`, `75.6836`, `75.6696`, `75.5242%`였고,
epoch 6이 최고(global `78.0989%`, empty 제외 `75.6836%`)다. 총 `2,240 patches/optimizer updates`를 처리한 이번
장기 조건은 model265의 3배 단기 최고 `75.0537%`보다 `+0.6299%p` 개선했지만, model257 단독
`77.2071%`보다 낮다. epoch 7–8에서 plateau/하락해 더 긴 동일 LR 학습의 기대 이득은 작아 보인다.
최종 route 기여 여부는 model225 확률과 혼합한 exact 평가 및 specimen-group 선택 CV로 판정한다.

결과: `$KOMAP_WORKSPACE/outputs/model273_highres_3x_longer_finetune/summary.json`
구현: `model253_highres_scale_finetune.py --scale 3 --patch-size 1536 --epochs 8 --global-representation`

## 모델 274: model273 고해상도 branch와 model225 route 결합 (미채택)

model273의 Al3Ni/Primary/Eutectic class-wise 확률 혼합률을 `0.05/0.10/0.20/0.30/0.40/0.50`으로
탐색해 기준 route를 포함한 43개 exact 후보를 평가했다. 기준 route가 image-wise `78.6921%`,
global `81.3033%`로 기존 기록과 정확히 일치하는 것을 확인했다. 신규 고해상도 branch를 포함한
후보 중 최고도 기준 route를 넘지 못했고, specimen-group leave-one-out 19 folds 모두 기준 route를
선택했다(held-out 변화 `0.0000%p`, positive image `0/20`, group bootstrap 95% CI `[0, 0]%p`).
따라서 model273 branch는 현재 route에 추가하지 않는다.

결과: `$KOMAP_WORKSPACE/outputs/model274_3x_highres_route/results.json`
구현: `model258_fullstats_route_probe.py` (model220 config 6: 20-step TENT, lr `5e-5` 사용)

## 모델 275: 3배 해상도 uniform-area crop sampling (route 평가 완료)

model273과 비교 시 crop 중심 sampling만 class-focused에서 image-area uniform으로 바꾸고,
나머지 3배/1536 patch, 4 crops/image, batch 1, 8 epoch, full-image z-score/CDF, 초기 checkpoint,
LR와 validation 절차를 고정했다. epoch별 image-wise mIoU는
`74.8558`, `75.3750`, `75.3988`, `75.5759`, `75.2146`, `75.3989`, `75.5271`, `75.4738%`로,
최고 epoch 4는 `75.5759%`(global `78.1866%`)였다. model273 최고 `75.6836%`보다 `-0.1077%p`
낮아 sampling 변경의 독립적인 성능 이득은 확인되지 않았다. epoch2–4에서 짧은 우위가 있었지만
후반에 유지되지 않았다.

결과: `$KOMAP_WORKSPACE/outputs/model275_3x_uniform_area_sampling/summary.json`
구현: `model253_highres_scale_finetune.py --sampling-mode uniform_area`

## 모델 276: model275 branch와 model225 route 결합 (미채택)

model275 확률을 Al3Ni/Primary/Eutectic class-wise 비율 `0.05/0.10/0.20/0.30/0.40/0.50`으로
섞은 43개 후보를 평가했다. model225 기준 route는 image-wise `78.6921%`, global `81.3033%`로
기록과 재현 일치했다. 고해상도 branch를 넣은 어떤 후보도 기준 route를 넘지 못했고, 19개
specimen-group fold 모두 기준 route를 선택했다(held-out 변화 `0.0000%p`, positive image `0/20`,
group bootstrap 95% CI `[0, 0]%p`). 따라서 채택하지 않는다.

결과: `$KOMAP_WORKSPACE/outputs/model276_3x_uniform_area_route/results.json`
구현: `model258_fullstats_route_probe.py` (model220 config 6: 20-step TENT, lr `5e-5` 사용)

## 모델 277: 3배 해상도 고학습률 fine-tuning (단독·route 미채택)

model257 full-image-stat checkpoint에서 scale 3, patch 1536(원본 환산 약 512px 시야),
이미지당 4 crops, batch 1 설정을 유지하고 encoder/decoder/native-stem 학습률을 각각
`1e-5 / 3e-5 / 1e-4`로 높여 4 epoch 학습했다. epoch별 공식 image-wise mIoU는
`75.6051`, `74.8028`, `75.5712`, `75.7693%`였고 epoch 4가 최고(global `78.4152%`,
Eutectic Si global IoU `56.7832%`)다. model273의 best `75.6836%`보다 `+0.0857%p`지만,
model257 standalone `77.2071%`보다 `-1.4378%p`, 채택 route model225의 `78.6921%`보다
`-2.9228%p` 낮아 단독 모델 개선은 아니다. 공식 검증 확률은 정상 저장됐으며 기존 route에
보완 branch로 넣을 가치가 있는지 exact 혼합 및 specimen-group CV를 추가 확인한다.

결과: `$KOMAP_WORKSPACE/outputs/model277_3x_high_lr_finetune/summary.json`
체크포인트: `$KOMAP_WORKSPACE/outputs/model277_3x_high_lr_finetune/best.pt`

## 모델 278: model277 고해상도 branch와 model225 route 결합 (미채택)

model277 확률을 Al3Ni/Primary/Eutectic class-wise 혼합률 `0.05/0.10/0.20/0.30/0.40/0.50`으로
탐색한 43개 exact 후보 중 image-wise, empty-class 제외, global mIoU 모두 기존 model225 route가
최고였다(image-wise `78.6921%`, empty 제외 `79.4693%`, global `81.3033%`). 19개 valid specimen
group fold에서도 전부 model225 route를 골랐으며 held-out 변화 `0.0000%p`, positive `0/20`,
group bootstrap 95% CI `[0.0000, 0.0000]%p`였다. 따라서 model277은 단독 점수뿐 아니라 현재
최고 route를 보완하는 데도 유효한 개선을 만들지 못했다.

결과: `$KOMAP_WORKSPACE/outputs/model278_3x_high_lr_route/results.json`
구현: `model258_fullstats_route_probe.py` (model220 config 6: 20-step TENT, lr `5e-5`)

## 모델 279: 3배 고해상도 8,000-update 계획 run (4 epoch 조기 종료·미채택)

model257 full-image-stat ResNet34 U-Net++/ASPP checkpoint에서 scale 3, patch 1536(원본 환산
약 512px 시야), 4 crops/image, batch 1, high LR(`1e-5 / 3e-5 / 1e-4`)로 장기 학습을
시작했다. 28 epoch·7,840 updates 계획이었지만, 첫 4 epoch(1,120 updates)에서 image-wise
mIoU가 `76.0337`, `74.8857`, `75.6801`, `75.6798%`로 정체했고 best epoch 1도 model257 단독
`77.2071%`보다 `-1.1734%p`, 최고 route `78.6921%`보다 `-2.6584%p` 낮아 조기 종료했다.
epoch 4 global mIoU는 `78.9642%`였지만 목표 metric인 image-wise mIoU 개선으로 이어지지 않았다.
따라서 8,000-update 학습량 가설은 이 설정에서 검증 완료된 것이 아니며, 이후 고해상도 재시도는
팀의 EfficientNet-B4 계열 및 warmup schedule과 학습 입력 표현을 맞춰 별도 평가해야 한다.

보존 체크포인트: `$KOMAP_WORKSPACE/outputs/model279_3x_8000step/best.pt` (epoch 1)
기록: `$KOMAP_WORKSPACE/outputs/model279_3x_8000step/history.json` (4 epoch)

## 모델 280: EfficientNet-B4 U-Net 4배 고해상도 (조기 종료, 미채택)

ImageNet pretrained EfficientNet-B4 U-Net을 fresh decoder로 초기화하고, scale 4·1536px crop
(원본 약 384px 시야), uniform-area crop sampling, metallography affine/photometric augmentation,
400-update warm-up + cosine decay로 7,840 optimizer updates를 계획했다. 검증은 4 epoch마다
고정 D4 TTA/overlap inference로 측정하며, test 추론은 최종 검증 뒤로 미뤘다.

검증 image-wise mIoU는 epoch 4/8/12/16/20에서 각각 `70.3792%`, `71.1019%`, `70.6763%`,
`71.3317%`, `71.1283%`였다. 최고 epoch 16의 global mIoU는 `74.7216%`, class global IoU는
Al3Ni `70.6978%`, Primary Si `85.2935%`, Al `91.6581%`, Eutectic Si `51.2371%`다.
epoch 20 global mIoU는 `75.0102%`이나 image-wise는 최고보다 낮았다. 5,600 updates 시점에
최고 route `78.6921%`보다 `-7.3604%p` 낮고 epoch 8 이후 plateau라 판단해 epoch 20에서
중단했다. best checkpoint는 epoch 16, last checkpoint는 epoch 20이며 모두 보존했다.

이 run은 기본 클래스 가중치 `[1.15, 1.45, 0.70, 2.00]`를 사용했다. 팀 고해상도 요약의
CE+Dice 설정에는 클래스 가중치가 명시되지 않아, 다음 EffNet-B4 재현에서는 equal weights를
명시해 이 설정 차이를 분리한다. valid 20장 중 A71 한 장은 정답 Primary Si가 없어 빈 클래스
처리 convention이 점수 비교에 영향을 줄 수 있으나, 이것만으로 현재 성능 격차를 설명하진 못한다.
test 추론은 하지 않았다.

현재 체크포인트: `$KOMAP_WORKSPACE/outputs/model280_effb4_4x_highres/last.pt`
진행 기록: `$KOMAP_WORKSPACE/outputs/model280_effb4_4x_highres/history.json`

## 모델 281: EfficientNet-B4 U-Net 4배 고해상도, equal-weight CE+Dice (조기 종료, 미채택)

모델280과 같은 seed `280`, 초기화·입력 해상도·patch/FOV·sampling·augmentation·optimizer
schedule·D4 검증을 유지하고 클래스 가중치만 `[1,1,1,1]`로 고정했다. 팀의 요약 설정은 CE+Dice이며 class weighting을
명시하지 않았으므로, 모델280의 `[1.15,1.45,0.70,2.00]` 가중치가 낮은 점수의 원인인지 분리하는
통제 실험이다. 28 epoch(7,840 updates) 계획, valid 4 epoch 간격, test 추론 생략.

epoch 1/2/3/4/5 train loss는 `1.5409` / `0.8108` / `0.6508` / `0.5957` / `0.5574`이며,
epoch 4 image-wise mIoU는 `70.4383%` (global `74.5134%`)다. model280의 같은 epoch 대비
`+0.0591%p`로 거의 차이가 없었다. 더 직접적인 팀 optimizer 설정을 확인해 차등 LR 실험으로
전환하며 epoch 5에서 조기 종료했다. best checkpoint는 epoch 4, last는 epoch 5다. test 추론은
하지 않았다. 출력 경로:
`$KOMAP_WORKSPACE/outputs/model281_effb4_4x_unweighted/`

## 모델 282: EfficientNet-B4 U-Net 4배 고해상도, 차등 LR + equal-weight CE+Dice (조기 종료, 미채택)

모델281에서 확인한 class-weight 효과 미미함을 반영해 같은 seed `280`·fresh ImageNet B4·4배
1536px patch·uniform-area sampling·metallography augmentation·equal CE+Dice를 유지한다. 변경점은
팀의 optimizer 문서에서 참고한 encoder LR `1e-4`, decoder LR `3e-4`의 차등 학습률이다. 400-update
warm-up + cosine으로 총 7,840 updates 학습하고 4 epoch마다 D4 검증한다. epoch 1 train loss는
`1.1286`이며, 같은 seed의 단일-LR 무가중치 model281 epoch 1 (`1.5409`)보다 빠르게 낮아졌다.
epoch 4/8/12 train loss는 `0.5057` / `0.4480` / `0.4277`, image-wise mIoU는
`67.6885%` / `69.5036%` / `69.8573%` (epoch 12 global `73.5799%`)다. epoch 12 global class
IoU는 Al3Ni `70.9195%`, Primary Si `83.3019%`, Al `91.9021%`, Eutectic Si `48.1960%`다.
model280 epoch 12보다 `-0.8190%p` 낮고, 최고 route model225보다 `-8.8348%p` 낮다. 12 epoch
(3,360/7,840 updates)에서 회복 폭이 작고 경쟁력이 없어 조기 종료했다. best/last checkpoint는
epoch 12이며 test 추론은 생략했다. 출력 경로:
`$KOMAP_WORKSPACE/outputs/model282_effb4_4x_diff_lr/`

## 모델 283: model257 full-stat 체크포인트 + 이미지별 Lovasz-Softmax 미세조정 (route 미채택)

model257의 epoch 4 체크포인트에서 시작해, 1배/1024px patch·이미지당 8 crops·batch 2·full-image
통계 입력·seed 257·encoder/decoder/native 학습률 `5e-6 / 1.5e-5 / 1e-4`를 유지한다.
weighted CE + Dice에 이미지별 정답 존재 클래스만 평균하는 Lovasz-Softmax(`weight=0.25`)를
추가해 대회 image-wise mIoU 목표와 직접 정렬한다. 나머지 구성은 model257과 맞춰 손실 변경 효과를
분리한다. valid 전체에 D4 TTA/overlap 추론을 적용하고 `image_mean_miou` 최고 epoch checkpoint를
선택한다. 현재 train/valid만 사용하며 test 추론은 하지 않았다.

검증 image-wise/global은 epoch 1 `77.3217/79.8848%`, epoch 2 `77.2833/80.2256%`,
epoch 3 `77.2471/80.0534%`, epoch 4 `77.3348/80.0853%`였다. 최고 epoch 4는 model257
standalone `77.2071%`보다 `+0.1277%p`지만 model225 route보다는 낮다.

36개 class-wise route 후보 중 valid 최고는 Primary/Eutectic에 model283 확률을 각각 60%,
Al3Ni는 기존 20-step TENT(model220 config 6)를 40% 섞은 설정으로 image-wise `78.7818%`
(global `81.4019%`)였다. model225의 `78.6921%`보다 raw `+0.0897%p`지만, 19개 시편
leave-one-group-out 평균 delta는 `-0.0121%p`, positive image `11/20`, group bootstrap 95% CI
`[-0.1867, +0.1374]%p`로 0을 포함해 validation 탐색 과적합 가능성이 크므로 route에 반영하지
않는다. route reference는 model225의 `78.6921%`를 정확히 재현했다. 초기 점검용 5-step TENT
입력 probe는 기준 재현에 실패해 유효 결과로 사용하지 않는다.

최고 standalone checkpoint: `$KOMAP_WORKSPACE/outputs/model283_model257_imagewise_lovasz/best.pt`
확률/route 결과: `$KOMAP_WORKSPACE/outputs/model283_model257_imagewise_lovasz/`
test 추론은 하지 않았다.

## 모델 284: model257 이미지별 Lovasz-Softmax 가중치 0.5 (route 미채택)

모델283과 동일하게 model257 epoch 4에서 초기화하고 seed·입력·crop·optimizer·D4 검증을 고정했으며,
Lovasz 가중치만 `0.25`에서 `0.5`로 높였다. epoch별 image-wise/global mIoU는
`77.3876/79.9540`, `77.3341/80.2805`, `77.3231/80.1526`, `77.3905/80.1543%`로
best epoch 4 standalone은 `77.3905%`다. 모델283의 `77.3348%`보다 `+0.0557%p`지만,
최고 route model225 `78.6921%`보다 낮다.

model225 확률을 정확히 재현하는 입력으로 36개 class-wise 혼합 후보를 비교한 valid 최고는
Primary/Eutectic에 model284를 각각 60%, Al3Ni에 20-step TENT(model220 config 6)를 40% 섞어
`78.7928%`(global `81.4157%`)였다. model225보다 raw `+0.1007%p`이나, 19개 시편
leave-one-group-out 평균 delta `-0.0082%p`, positive image `11/20`, group bootstrap 95% CI
`[-0.1836, +0.1406]%p`로 0을 포함해 채택하지 않는다.

체크포인트·확률·CV 결과: `$KOMAP_WORKSPACE/outputs/model284_model257_imagewise_lovasz_w050/`
test 추론은 하지 않았다.

## 모델 285: 이미지별 present-class macro CE + Dice + Lovasz (조기 종료, 미채택)

model257 epoch 4 체크포인트에서 시작해 scale 1·1024 patch·8 crops/image·batch 2·full-image
통계·seed 257·학습률·4 epoch을 고정한다. 변경점은 weighted pixel CE를 각 이미지 내 정답이 존재하는
클래스별 평균 CE로 대체하는 것이며, Dice 및 per-image Lovasz(`weight=0.25`)는 유지한다.
이는 이미지마다 네 클래스 IoU를 동일 비중으로 평균하는 평가 기준에 CE도 정렬해 보는 실험이다.
서버 CUDA smoke test에서 결측 클래스가 있는 입력에 대해 손실 및 gradient 유한성을 확인했다.

epoch 1/2 image-wise mIoU는 `76.2252%` / `75.7108%` (global `78.5956%` / `78.2334%`)로
하락했다. weighted CE 기반 model257 `77.2071%`보다 크게 낮고, train loss는 감소하지만 valid가
악화되어 epoch 2에서 조기 종료했다. best epoch 1 checkpoint 및 두 epoch의 history는 보존했으며,
성능이 기준보다 낮아 valid probability export와 route probe는 수행하지 않았다.

출력 경로: `$KOMAP_WORKSPACE/outputs/model285_model257_macro_ce/`

## 모델 286: model257 + 이미지별 present-class Soft-IoU 미세조정 (route 미채택)

model257에서 시작해 scale 1·1024 patch·8 crops/image·batch 2·full-image 통계·seed 257·학습률·
4 epoch을 유지한다. weighted CE + Dice 기반을 보존하면서 이미지별 정답 존재 클래스 Soft-IoU
(`weight=0.5`)를 추가한다. 이전 Soft-IoU 시도는 더 오래된 backbone checkpoint에서 수행됐으므로,
이번에는 현재 강한 full-stat model257 초기화를 사용해 재검증한다.

epoch별 image-wise/global mIoU는 `77.3902/79.9529`, `77.3441/80.2906`, `77.3432/80.1827`,
`77.3950/80.1641%`였으며 best epoch 4 단독은 `77.3950%`로 model257 `77.2071%` 대비
`+0.1878%p` 개선됐다.

model225 route와 정확히 비교한 36개 class-wise 혼합 후보 최고는 Primary/Eutectic에 model286을
각각 60%, Al3Ni에 model220 config 6 TENT를 40% 섞은 `78.8001%`(global `81.4254%`)였다.
model225보다 raw `+0.1080%p`지만 19개 시편 grouped-CV 평균 변화 `-0.0055%p`, positive image
`10/20`, group bootstrap 95% CI `[-0.1831, +0.1476]%p`로 0을 포함해 채택하지 않았다.
route reference는 model225 `78.6921%`를 재현했다. test 추론은 수행하지 않았다.

체크포인트·확률·CV 결과: `$KOMAP_WORKSPACE/outputs/model286_model257_imagewise_softiou/`

## 모델 287: model225 component-label / foreground-shape oracle 진단 (완료)

새 학습 전에 region-level class 할당과 Al-vs-secondary 경계 중 어느 쪽에 점수 회복 여지가
있는지 model225 exact route로 검증했다. 예측된 각 secondary class를 따로 4/8-connectivity
component로 나눠 GT majority label을 주는 class-component oracle은 `84.3379/83.9980%`로
baseline `78.6921%`보다 `+5.6458/+5.3060%p`였다. 4-connected 개선의 specimen-group bootstrap
95% CI는 `[+4.6198, +6.6903]%p`다.

GT의 secondary-vs-Al binary shape만 사용하고 phase class는 현 예측을 유지(Al로 놓친 GT secondary만
route 확률로 재분류)한 foreground-shape oracle은 `84.3188%`, delta `+5.6267%p`
(group bootstrap 95% CI `[+4.9819, +6.2686]%p`)였다. 따라서 경계/foreground 분할과 region-level
phase 재분류 양쪽 모두 큰 이론적 headroom이 있지만, 이는 GT 사용 oracle이므로 실제 모델 성능이
아니다. secondary 전체를 하나의 연결 component로 뭉친 naive oracle은 components가 상들을 가로질러
합쳐져 `60.5830%`로 악화했으며 유효한 region oracle로 해석하지 않는다.

이 진단을 근거로 다음은 component-level classifier를 specimen-group CV에서 먼저 시험한다. 통계
결과와 image별 oracle 기록은 다음에 보존했다.

결과: `$KOMAP_WORKSPACE/outputs/model287_region_oracle_diagnostic/results_classwise.json`

## 모델 288: model225 region classifier leave-one-specimen-out probe (미채택)

model225 exact route의 class별 connected components에서 예측 확률, 크기/형상, intensity 통계를
추출하고, component GT majority class를 target으로 학습하는 tabular classifier를 19개 specimen
group leave-one-out으로 평가한다. 이 단계는 valid 내부 grouped-CV 진단이며, 결과가 유망하면 train
OOF component를 사용한 별도 학습으로 넘어간다. GT는 각 fold의 classifier 학습에만 쓰고 held-out
specimen에는 사용하지 않는다.

출력 경로: `$KOMAP_WORKSPACE/outputs/model288_region_classifier_cv/`

19개 specimen LOSO에서 confidence threshold `0.0/0.60/0.75/0.90`별 image-wise 점수는
`77.5106/78.0725/78.2714/78.6086%`였다. 최고 threshold `0.90`도 model225 `78.6921%`보다
`-0.0835%p`(positive image `1/20`, group bootstrap 95% CI `[-0.2168, +0.0004]%p`) 낮았다.
낮은 threshold일수록 components를 과도하게 재분류해 하락했다. region majority oracle의 큰 headroom을
이 32-feature MLP가 회수하지 못했으므로 이 classifier는 채택하지 않는다.

## 모델 289: 이차상-vs-Al binary boundary loss 미세조정 (route 미채택)

model257 full-stat checkpoint에서 시작해 CE+Dice를 유지하고, 네 logits를 이차상(Al3Ni/Primary/
Eutectic) 대 Al의 binary logit으로 묶은 balanced BCE 및 per-image binary Soft-IoU를 추가한다.
BCE는 GT 경계 주변 3px band에 가중치를 높여 foreground-shape oracle에서 확인한 경계 headroom을
직접 겨냥한다. seed·scale·patch·optimizer는 model257 기준을 유지한다.

출력 경로: `$KOMAP_WORKSPACE/outputs/model289_model257_binary_boundary/`

epoch별 image-wise/global mIoU는 `77.4184/80.0075`, `77.2902/80.2992`, `77.2602/80.0908`,
`77.3867/80.1631%`였다. best epoch 1 standalone `77.4184%`는 model257보다 `+0.2113%p`지만,
후속 epoch에서 유지되지 않았다.

model225와 동일한 TENT input을 쓰는 36개 class-wise route 후보의 valid 최고는 Eutectic에
model289를 80% 섞고 Primary/Al3Ni를 model225 설정에 둔 `78.7608%`(global `81.4032%`)였다.
model225보다 raw `+0.0686%p`지만 grouped-CV delta `-0.0709%p`, positive image `14/20`,
group bootstrap 95% CI `[-0.3227, +0.1451]%p`로 0을 포함해 채택하지 않는다. 기준 route 재현은
`78.6921%`로 일치했다. 테스트셋 추론은 하지 않았다.

체크포인트·확률·CV 결과: `$KOMAP_WORKSPACE/outputs/model289_model257_binary_boundary/`

## 모델 290: saved prediction scale-oracle 대 1x null-oracle (완료, 미채택)

이미 저장된 같은 ResNet34 계열 1x/2x/3x valid 확률을 이용해 이미지별 GT-best-scale oracle을
구하고, 후보 수를 동일하게 맞춘 여섯 1x checkpoint/loss-variant oracle과 비교한다. 이 oracle은
배율 선택을 실제 구현하는 모델이 아니라, 이미지마다 다른 scale을 고르는 데 유효 headroom이
있는지의 상한 진단이다. 양쪽 비교 모두 valid GT를 oracle 선택에 사용하므로 그 점수는 채택 점수가
아니다.

6개 후보씩 비교한 scale oracle 평균은 `77.6424%`, 1x null oracle 평균은 `77.6972%`였다.
각 그룹의 최고 단일 후보 대비 oracle 선택 이득은 scale `+0.4349%p`, null `+0.3020%p`로,
scale의 추가 이득은 null보다 `-0.0548%p`(specimen-group bootstrap 95% CI
`[-0.2730, +0.1759]%p`)였다. 후보 수 선택 편향을 넘는 image-specific scale headroom은 확인되지 않아
per-image scale router나 추가 high-resolution 학습은 진행하지 않는다.

결과: `$KOMAP_WORKSPACE/outputs/model290_scale_oracle_vs_null/results.json`

## 모델 291: 이진 foreground + 조건부 상(phase) 계층형 head (미채택)

model287의 foreground-shape oracle이 보여준 Al/이차상 경계 headroom과 class-component oracle의
상 판별 headroom을 구조적으로 분리해 학습한다. model257에서 시작해 기존 4-class 분류 head를
보조 loss로 보존하고, 공용 decoder feature에 (1) 이차상-vs-Al 이진 head 및 (2) 이차상 픽셀에서만
Al3Ni/Primary/Eutectic을 판별하는 조건부 3-class head를 추가한다. 추론 확률은
`P(class)=P(secondary/Al) × P(phase | secondary)`로 결합한다. 새 head는 model257의 4-class
classifier weight에서 warm-start한다.

train split 70장만 사용하고 model257과 같은 1×·1024 patch·8 crops/image·full-image 통계·D4
검증 조건으로 4 epoch 미세조정했다. epoch별 image-wise mIoU는 `77.1369/77.0169/77.3023/77.2208%`
이며 최고 epoch3 `77.3023%`(global `80.2475%`)로 model257 단독 `77.2071%`보다 `+0.0952%p`
였다.

model225 route를 기준으로 model291을 기존 TENT branch와 class-wise 혼합하는 15개 사전 정의 후보를
평가했다. valid raw 최고 `78.7094%`는 기준 `78.6921%`보다 `+0.0173%p`에 불과했다. specimen-group
LOSO에서 fold별 route 선택 후 held-out image 평균 변화는 `-0.1585%p`, positive image `6/20`,
group bootstrap 95% CI `[-0.3285, -0.0168]%p`로 악화했다. 따라서 이 분기는 route에 채택하지 않고
model225를 최고로 유지한다. valid probability 및 그룹 CV 결과는 아래에 보존했다.

출력: `$KOMAP_WORKSPACE/outputs/model291_hierarchical_heads/`
구현: `model291_hierarchical_heads.py`, route 탐색은 `model258_fullstats_route_probe.py`

## 모델 292: 교차-시편 phase-component copy-paste 증강 (미채택)

model291 계층 head는 독립 branch 기준으로만 소폭 개선되고 route CV에서는 미채택됐다. 다음은
구조/loss 재조정 대신 train 시편 다양성을 늘리는 데이터 증강을 시험한다. 서로 다른 시편 그룹의
1024 patch batch pair 사이에서 4-connected Al3Ni/Primary/Eutectic component를 원래 크기로 옮겨
붙이고, 경계 안쪽 5px에서 feature를 feather blend한다. GT mask에는 실제로 붙인 component 영역만
donor phase로 반영한다. 증강은 train split에만 적용하고 valid에는 쓰지 않는다.

model257 초기화, 1×·1024 patch·8 crops/image·full-image 통계·D4 검증 및 동일 optimizer 설정으로
4 epoch 학습했다. epoch별 image-wise mIoU는 `76.8791/77.0478/77.0258/77.0655%`였고, 최고 epoch4
`77.0655%`(global `79.8358%`)는 model257 단독 `77.2071%`보다 `-0.1416%p` 낮았다. 매 epoch
214~235개 component가 붙었지만 검증 향상으로 이어지지 않았다. 기준 branch보다 낮아 model225 route
CV는 수행하지 않고 미채택한다. 로컬/서버 pycompile 및 synthetic tensor로 양방향 cross-specimen
paste smoke test를 통과했으며, 검증 probability도 저장했다.

출력: `$KOMAP_WORKSPACE/outputs/model292_cross_specimen_copy/`
구현: `model292_cross_specimen_component_copy.py`

## 모델 293: train-fitted region classifier (미채택)

모델288에서는 valid 20장의 component로 LOSO classifier를 학습해 표본 수가 작았다. 이번에는
model257 checkpoint를 train 70장에 추론해 class별 4-connected secondary component를 생성하고,
각 component의 majority GT class 및 영역/확률/강도 통계로 MLP를 train split에서 학습한다. region
교정은 model257의 예측 component 안에서만 수행한다.

주의: train 예측은 해당 모델이 학습한 이미지의 in-sample 출력이다. valid 내 train/valid 시편 ID
중복도 별도 표시해 해석한다. train 70장에서 180k 이상 predicted secondary components를 추출해
MLP를 학습하고 valid 20장에 적용했다. threshold `0.75/0.85/0.90/0.95`의 image-wise 점수는
`77.2383/77.2221/77.1958/77.1958%`였다. 저장 확률 fp16에 따른 model257 baseline 재현은
`77.2075%`(기록된 full precision summary `77.2071%`)였다. 가장 좋은 raw 개선 `+0.0308%p`는
positive image `6/20`뿐이다.

valid specimen-group threshold 선택 CV는 `+0.0087%p`, positive `5/20`, 95% CI
`[-0.0591, +0.1102]%p`였다. train과 A-ID가 겹치지 않는 valid 13장에서는 `+0.0457%p`, 겹치는
7장에서는 `-0.0599%p`였다. 더 많은 train component로 학습해도 oracle headroom을 회수하지 못했고,
model225 route 결합으로 이어갈 근거가 부족해 미채택한다.

구현: `model293_train_fitted_region_corrector.py`
출력: `$KOMAP_WORKSPACE/outputs/model293_train_region_corrector/`

## 모델 294: decoder hidden feature mask-pooled region probe (완료, 미채택)

모델287의 component-label oracle과 모델293의 handcrafted region feature 정체를 바탕으로,
model257을 고정한 채 decoder의 마지막 32-channel hidden map을 각 train GT component 안에서
mask-average/std pooling한다. 기존 확률·형태·밝기 component feature 32개와 hidden pooled feature
64개를 이어 3-class region MLP를 train GT component에서 학습하고, valid의 predicted component에
적용한다. 전체 segmentation network는 고정하며 이는 값싼 go/no-go probe다.

train inference는 full-stat 입력·1x·Gaussian tile overlap을 사용하고 valid segmentation base는
model257의 공식 D4 확률 cache를 그대로 쓴다. Threshold는 specimen-group held-out으로 보고하며,
train-valid A-ID 중복/비중복 그룹 성능도 분리한다. group-CV 이득이 작거나 비중복 시편에서 악화하면
full region-head fine-tuning으로 확장하지 않는다.

학습 component 180,865개, feature 96개. 고정 threshold `0.75/0.85/0.90/0.95`에서
image-wise mIoU는 각각 `77.1366/77.1047/77.0876/77.1110%`로, fp16 확률 기준 model257
baseline `77.2075%`보다 모두 낮았다. Specimen-group held-out threshold 선택도 `77.0531%`, 즉
`-0.1544%p`(positive image `8/20`, group bootstrap 95% CI `[-0.3355, +0.0037]%p`)였다.
시편 비중복 13장 delta `-0.0839%p`, 중복 7장 delta `-0.2855%p`로 양쪽 모두 악화했다.
따라서 frozen learned decoder region head는 확장하지 않고 미채택한다.

구현: `model294_decoder_region_probe.py`
출력 경로: `$KOMAP_WORKSPACE/outputs/model294_decoder_region_probe/`

## 모델 295: 이미지 내부 고신뢰 component anchor 진단 (완료, 미채택)

학습형 region 교정기(모델293·294)가 oracle headroom을 회수하지 못한 원인을 좁히기 위해,
학습 없이 같은 이미지 안의 고신뢰 component끼리 비교하는 고정 규칙을 평가한다. 4-connected
secondary component 중 예측 상 평균 확률이 0.90 이상이고 면적이 이미지 component 면적 중앙값 및
32px 이상인 component만 anchor로 삼는다. 위치를 제외하고 확률·불확실성·면적·형태·z/q 밝기/질감
특성을 이미지 내부 robust 표준화한 뒤, anchor 5개 중 4개 이상이 같은 다른 상을 지지할 때만
비-anchor component를 재분류한다. 규칙은 valid 정답을 보지 않고 고정하며, 정답은 제안의 정밀도와
대회식 image-wise mIoU를 사후 진단하는 데만 쓴다. route 학습/valid 튜닝은 하지 않는다.

기준 Model225 route 재현은 `78.6921%`로 일치했다. 고정 교정 후 valid image-wise mIoU는
`70.9398%`(`-7.7523%p`, positive image `3/20`, specimen-group bootstrap 95% CI
`[-12.7669, -3.1122]%p`)로 크게 하락했다. 83,242개 component 중 잘못 예측된 component의
24.20%만 교정됐고, 그 면적 기준 교정률은 14.37%였다. 반면 원래 맞던 component의 35.04%가
오염됐으며, 정답 component 면적의 5.03%가 잘못 바뀌었다. 이미지 내 feature 유사도 투표는 phase
정체성을 보존하지 못하므로 이 계열의 label-free post-hoc 교정은 중단한다.

구현: `model295_within_image_anchor_diagnostic.py`
출력 경로: `$KOMAP_WORKSPACE/outputs/model295_within_image_anchor/`

## 모델 296: region 학습용 train in-sample 예측 편향 진단 (완료)

모델293의 MLP는 model257이 학습에 사용한 train 이미지의 예측 component로 학습됐다. 이 예측이
valid보다 지나치게 쉬우면 region classifier가 train에서만 잘못된 분포를 배운다. model257과 동일한
1x·1024 patch·576 stride·D4 조건으로 train 70장을 다시 추론하고, saved valid 확률과 4-connected
secondary component의 GT-majority 오류율, 면적 가중 오류율, confidence·entropy·margin 분포를
비교한다. valid 시편 중 train A-ID 중복/비중복도 따로 본다. 이 진단은 정답으로 모델을 튜닝하지 않으며,
큰 train-valid gap이 확인될 때만 OOF region 학습 비용을 검토한다.

train in-sample component 오류율은 개수 기준 `27.32%`, 면적 가중 `2.99%`였으나, valid는 각각
`37.10%`, `7.12%`였다. valid 평균 confidence는 train `0.675` 대비 `0.658`, entropy는 `0.747` 대비
`0.769`로 valid에서 더 어렵고 불확실했다. valid 중 train A-ID 비중복 13장은 component 오류율
`31.73%`(면적 가중 `6.52%`), 중복 7장은 `43.59%`(`7.64%`)였다. 특히 비중복 valid에서도
in-sample train보다 오류율이 `4.41%p`, 면적 오류가 `3.54%p` 높아 단순 시편 중복만의 효과는 아니다.
모델293 MLP 학습 입력의 오류율은 valid보다 낮고 면적 기준으로 2.4배 차이가 나므로, OOF component
예측으로 region classifier를 재평가할 근거가 있다. 이 결과만으로 성능 향상을 주장하지 않는다.

기존 `model96_4fold_ensemble` 체크포인트는 재사용할 수 없다. 해당 legacy fold runner는 train과 valid를
합친 90장으로 fold를 나누고, RGB 입력의 구형 model6을 학습했다. 이는 train 70장만으로 만든
model257의 누수 없는 OOF 예측도 아니고, 현재 입력/모델 branch와도 다르므로 새 train-only fold 모델이
필요하다.

구현: `model296_region_train_valid_shift_audit.py`
출력 경로: `$KOMAP_WORKSPACE/outputs/model296_region_shift_audit/`

## 모델 297: train-only 시편 그룹 OOF segmentation branch (완료, 단독 미채택)

모델293 region corrector의 in-sample 편향을 평가하기 위해 train 70장만 4개 A-ID 그룹 fold로 나눈다.
각 fold 모델은 ImageNet ResNet34 초기화만 사용하고 held-out 시편의 mask를 보지 않은 채 8 epoch
학습한다. Model257과 같은 raw/z/CDF full-image 통계 경로를 쓰되, 계산량을 제한한 512px patch로
학습·D4 Gaussian overlap 추론한다.
각 train 이미지의 held-out 확률을 OOF cache에 저장하고, 같은 fold 모델들의 valid 확률 평균도 만든다.
valid mask는 학습/epoch selection에 전혀 쓰지 않는다. 먼저 OOF 및 valid 단독 점수를 확인한 뒤
region corrector나 model225 route 혼합으로 진행할지 결정한다.

네 fold의 held-out image mIoU는 `75.3545%` (18장), `76.2916%` (18장), `81.0904%` (17장),
`76.3342%` (17장)이다. fold image 수를 반영한 전체 OOF 평균은 `77.2264%`; fold 단순평균은
사용하지 않는다. 네 fold valid probability ensemble은 image-wise `74.2546%`로, model225
route보다 낮아 단독 채택하지 않는다. OOF 수치는 train 시편 그룹 교차검증 결과라 valid 최고점과
직접 비교하지 않는다. validation mask는 학습·epoch selection·fold 결정에 쓰지 않았다.

구현: `model297_train_only_oof_branch.py`
출력 경로: `$KOMAP_WORKSPACE/outputs/model297_train_only_oof/`

## 모델 298: train-only OOF region corrector (완료, 개선 미미·미채택)

모델297의 held-out train 확률로 component MLP를 학습하고, specimen-group meta-CV로 confidence
threshold `0.60`을 선택했다. 전체 OOF component로 재학습해 모델297 valid fold-ensemble에 적용한
결과 valid image-wise mIoU는 `74.2546%`에서 `74.2897%`로 `+0.0351%p` 변했다(positive 9/20,
specimen-group bootstrap 95% CI `[-0.0848, +0.1831]%p`). Train meta-CV 변화도 `+0.0354%p`
(95% CI `[-0.0416, +0.1290]%p`)로 0을 포함한다. valid 중 train과 A-ID가 겹치지 않는 13장의
변화는 `+0.0403%p`, 겹치는 7장은 `+0.0253%p`다. 개선 근거가 불확실해 단독 branch로 미채택하고,
model225 route에 대한 소량 class-wise 혼합만 specimen-group held-out으로 추가 확인한다.

구현: `model298_oof_region_corrector.py`
출력 경로: `$KOMAP_WORKSPACE/outputs/model298_oof_region_corrector/`

## 모델 299: Model298 OOF 보정 branch와 Model225 route 결합 (완료, 미채택)

Model225의 Primary·Al3Ni 설정을 고정하고, Model298 corrected probability를 Al3Ni/Primary/
Eutectic 부분집합에 `5/10/20/30%` 섞은 28개 exact 후보를 비교했다. evaluator에서 Model225 기준
image-wise `78.6921%` 재현을 확인했다. in-sample 최고는 Al3Ni에만 Model298을 5% 섞은
`78.6077%`로 기준보다 `-0.0844%p` 낮았다. 19개 specimen-group leave-one-out 선택은 전 fold에서
Model225 기준을 골랐고 held-out delta `0.0000%p`, positive image `0/20`, group bootstrap 95% CI
`[0, 0]%p`였다. 따라서 Model298은 현재 route에 섞지 않는다.

구현: `model258_fullstats_route_probe.py --source-id 298`
결과: `$KOMAP_WORKSPACE/outputs/model299_model298_route_probe/`

## 모델 300: 회전 불확실도 가중 Mean-Teacher consistency (완료, 미채택)

모델297의 각 fold checkpoint를 초기값으로 1 epoch fine-tune했다. held-out train fold의 입력만
image-only dataset으로 제공하고, 해당 mask와 valid mask는 학습 중 열지 않았다. EMA teacher의
0/90/180/270도 예측을 역회전해 평균하고, view 간 KL divergence에 따른 `exp(-4 × KL)` confidence로
student consistency MSE를 가중했다. `lambda=0.5` fold1 pilot과 gradient 기여를 키운 `lambda=5`
네-fold 실험을 수행했다.

lambda=5의 stored OOF 기준 image-wise mIoU는 Model297 `77.2266%`에서 `77.2124%`로
`-0.0142%p` 변했다(positive 32/70, specimen-group bootstrap 95% CI
`[-0.0756, +0.0447]%p`). fold별 OOF 변화는 `-0.0036, +0.0671, +0.0112, -0.1373%p`로
시편에 따라 일관되지 않았다. valid 4-fold probability ensemble은 Model297의 `74.2546%`에서
`74.2646%`로 `+0.0100%p`였으나 positive 12/20, group bootstrap 95% CI
`[-0.0420, +0.0566]%p`로 0을 포함한다. 따라서 이 consistency 설정은 성능 개선 근거가 없어
미채택한다.

구현: `model300_rotation_consistency_pilot.py`
결과: `$KOMAP_WORKSPACE/outputs/model300_rotation_consistency_pilot/aggregate_lambda5.json`

## 모델 301: Model257 boundary-enhanced U-Net++ (구현만 완료, 미학습·대체됨)

Al-Si 미세조직 분할 논문의 Boundary Enhancement Module에서 형태학적 경계 추출과
다중 dilation 문맥 결합 아이디어를 가져와 Model257의 fine-resolution U-Net++ decoder
stage `x0_1/x0_2/x0_3`에 공유 적용한다. dilation rate는 `1/3/5/7`이고, residual projection은
zero-init해 기존 checkpoint 예측을 보존한다. Model257 epoch 4 checkpoint를 실제로 불러온 CPU
검증에서 초기 logits 최대 차이 `0.0`을 확인했다. 학습은 scale 1, patch 1024, 8 crops/image,
batch 2, weighted CE+Dice, 4 epoch 설정으로 준비했다. [RUCS-BE-MT 논문](https://www.sciencedirect.com/org/science/article/pii/S1546221826001864)

Model257 checkpoint는 모든 train 시편으로 학습된 가중치이고, 원 구현은 valid mIoU로 epoch을
선택한다. 따라서 이 checkpoint로 grouped OOF 성능을 주장할 수 없으며, 반복적인 valid 선택에 따른
낙관 편향도 우려된다. 이 방식의 학습은 시작하지 않았고, held-out train groups를 지킨 Model297
fold checkpoint에서 이어 학습하는 Model302로 대체한다. 아래는 구현·parity 확인뿐이며 성능 개선을
의미하지 않는다.

구현: `model301_boundary_enhanced_finetune.py`
기준 checkpoint: `$KOMAP_WORKSPACE/outputs/model257_full_image_stats_1x/best.pt`

## 모델 302: Model297 train-only grouped OOF + boundary enhancement (P100 완료, 미채택)

Model297의 네 specimen-group OOF checkpoint를 각각 그대로 이어 받아, 같은 fold의 train groups에만
Model301 boundary enhancement를 4 epoch 미세조정한다. 각 checkpoint가 보지 않은 held-out groups로
OOF 예측을 만들고, fold별 검증 checkpoint 선택은 하지 않고 고정된 마지막 epoch을 쓴다. valid masks는
네 fold 학습·예측이 모두 끝난 뒤 최종 측정에만 열어, validation 기반 epoch 선택 누수를 막는다.

비교의 주 지표는 Model297 대비 paired train-OOF image-wise mIoU 변화와 specimen-group bootstrap
95% CI다. valid 4-fold probability ensemble 변화는 별도 참고값으로 보고 OOF 근거 없이 채택하지 않는다.
Model297 fold1 checkpoint로 실제 state-dict 호환을 확인했고, baseline 대비 초기 logits 최대 차이 `0.0`을
확인했다. 로컬·P100 원격 py_compile 및 CPU smoke test를 통과했고, 로컬 Mac MPS에서도 batch 2,
128×128 forward/backward와 65% allocator cap 설정을 smoke-test해 finite logits/gradient를 확인했다.
사용자가 P100 실행을 선택했다. 첫 GPU0 시도는 65% 메모리 한도(`10.33GiB`)에서 약 `10.56GiB` 할당
중 OOM으로 첫 epoch 전 종료됐으며, 부분 폴더를 `model302_boundary_oof_gpu0_oom/`에 보존했다.
이후 GPU6에 고정하고 메모리 한도를 90%로 높여 네 fold 전체를 완료했다. 1024 patch, 8 crops/image,
batch 2, 4 epochs/fold이며 fold별 학습 loss는 아래와 같다.

| Fold | held-out 이미지 | Epoch train loss (1→4) | 경과시간 |
|---|---:|---|---:|
| 1 | 18 | 0.48423 → 0.47072 → 0.45742 → 0.45132 | 938.5초 |
| 2 | 18 | 0.49507 → 0.47446 → 0.46279 → 0.45736 | 901.9초 |
| 3 | 17 | 0.52612 → 0.50776 → 0.49577 → 0.49180 | 912.0초 |
| 4 | 17 | 0.47127 → 0.46218 → 0.44861 → 0.44421 | 914.7초 |

최종 train-only grouped OOF image-wise mIoU는 Model297 `77.2266%`에서 Model302 `77.4562%`로
`+0.2296%p`였다(positive images `41/70`, specimen-group bootstrap 95% CI
`[-0.3177, +0.7440]%p`). 신뢰구간이 0을 포함해 개선은 불확실하다. 20장 valid 4-fold probability
ensemble은 `74.2546%`에서 `74.9864%`로 `+0.7318%p`였지만, 이는 보조 결과로만 본다. 기존 최고
valid score `78.7209%`를 넘지 못했으므로 Model302는 최고 모델로 채택하지 않는다. valid masks는
네 fold의 학습과 예측이 모두 끝난 뒤 점수 계산에만 사용했다.

실행기는 `auto/cuda/mps/cpu` 장치를 지원한다.

기존 Model287 valid oracle의 image별 진단에서 A71은 기준 `46.64%`/class-component oracle `55.71%`로
oracle headroom이 상대적으로 작다. A153(`72.71→81.88%` shape oracle), A74(`73.97→80.57%`),
A11 step2(`74.00→81.98%`), A67(`74.46→81.53%`)에는 더 큰 shape/class-region 회복 여지가 있다.
이는 valid GT를 쓴 비실현 상한일 뿐이며 학습이나 route 선택에 사용하지 않는다. 한 장만 크게 고쳐
80%를 기대하기보다 여러 hard image에서 paired OOF 개선이 일관적인지 본다.

구현: `model302_boundary_oof_finetune.py`
기준 체크포인트: `$KOMAP_WORKSPACE/outputs/model297_train_only_oof/fold{1..4}/last.pt`
기준 OOF: `$KOMAP_WORKSPACE/outputs/model297_train_only_oof/train_oof_probs.npz`
결과: `$KOMAP_WORKSPACE/outputs/model302_boundary_oof/results.json`
fold 로그: `$KOMAP_WORKSPACE/outputs/model302_boundary_oof_gpu6.log`

## 모델 303: Model302 low-LR grouped OOF continuation (P100 완료, 보완성 평가 중)

Model302 네 fold의 마지막 checkpoint에서 각 fold별 4 epoch을 추가 미세조정한다(누적 8 epoch).
encoder/decoder/native/boundary 학습률은 Model302 설정의 1/4로 낮추고, 같은 held-out specimen groups와
같은 train OOF 이미지에서 Model302와 paired 비교한다. valid는 최종 4-fold ensemble 측정에만 사용하며
학습·epoch 선택에는 사용하지 않는다. 목적은 Model302의 네 fold 모두 마지막 epoch까지 train loss가
계속 내려간 데서 추가 학습 여지를 확인하는 것이다. 긍정 결과가 있어도 grouped OOF와 group bootstrap
95% CI를 기준으로 채택 여부를 판단한다.

P100 GPU6, 1024 patch, 8 crops/image, batch 2, 90% memory cap으로 네 fold를 완료했다. fold1/2/3/4의
추가 4 epoch train loss는 `0.44789→0.44439`, `0.45261→0.44964`, `0.48468→0.48421`,
`0.43368→0.43711`이었다. train-only grouped OOF image-wise mIoU는 Model302 `77.4562%`에서
Model303 `77.5096%`로 `+0.0533%p` 변했고 70장 중 46장이 개선됐다. specimen-group bootstrap 95% CI는
`[+0.0155, +0.0909]%p`로 양수지만 개선폭은 작다. valid 4-fold ensemble은 `75.0472%`로
Model302 대비 `+0.0608%p`였다. 현재 최고 route `78.7209%`에는 미치지 못해 단독 채택하지 않는다.
Model304에서 이 branch를 Model225 route에 class-wise 5–50% 혼합한 43개 exact 후보로 평가했다.
기준 route는 `78.6921%`로 재현됐고 in-sample 최고도 기준 route 그대로였다. 19-group leave-one-out은
모든 fold에서 Model225 기준을 선택해 held-out delta `0.0000%p`(positive `0/20`, bootstrap CI
`[0,0]%p`)였다. Model303 branch는 현재 route를 보완하지 못해 미채택한다.

구현: `model302_boundary_oof_finetune.py` (`--model-id`/baseline ID/LR multiplier 추가)
결과 경로: `$KOMAP_WORKSPACE/outputs/model303_boundary_oof_8ep/`
실행 로그: `$KOMAP_WORKSPACE/outputs/model303_boundary_oof_8ep_gpu6.log`

## 모델 304: Model303 boundary branch의 Model225 route 보완성 검증 (P100 완료, 미채택)

Model303 fold-ensemble 확률을 기존 최고 route인 Model225에 class-wise로 5–50% 혼합하는 43개
exact 후보를 검증했다. Model225 기준 image-wise mIoU `78.6921%`는 재현됐고, in-sample 최고도
기준 route 자체였다. 19개 specimen-group leave-one-out에서도 모든 fold가 Model225를 선택해 held-out
변화 `0.0000%p`, positive image `0/20`, group-bootstrap 95% CI `[0.0000, 0.0000]%p`였다.
Model303 boundary branch는 최고 route를 보완하지 못했다.

결과: `$KOMAP_WORKSPACE/outputs/model304_model303_route_probe/results.json`

## 모델 305: Model257 full-data checkpoint + boundary enhancement (완료, 미채택)

Model257의 전체 train 학습 checkpoint에 zero-init shared multi-dilation boundary module을 붙여 4 epoch
미세조정했다(1024 patch, 8 crops/image, batch 2, weighted CE+Dice). `--fixed-final-epoch` 모드로
훈련 중 valid mask 평가와 checkpoint 선택을 생략하고 마지막 epoch을 저장한 뒤에만 valid를 측정했다.
train loss는 `0.35199 → 0.34778 → 0.34061 → 0.33993`으로 감소했지만, valid image-wise mIoU는
Model257 `77.2075%`에서 `76.6797%`로 `-0.5278%p` 하락했다(9/20 images 개선). 클래스별 image-mean
IoU 변화는 Al3Ni `-1.1917%p`, Primary Si `-0.2739%p`, Al `-0.0173%p`, Eutectic Si `-0.6289%p`로
주요 두 Si 상이 악화했다. 단독 모델로 미채택하며, 확률 보완 여부만 Model306에서 확인한다.

결과: `$KOMAP_WORKSPACE/outputs/model305_model257_boundary_fixed4/summary.json`
체크포인트: `$KOMAP_WORKSPACE/outputs/model305_model257_boundary_fixed4/last.pt`
구현: `model301_boundary_enhanced_finetune.py --fixed-final-epoch`

## 모델 306: Model305 boundary branch의 Model225 route 보완성 검증 (P100 완료, 미채택)

Model305 전체 train 학습 branch를 기존 Model225 class-wise route의 후보 source로 추가하고 43개 exact
혼합·보정 후보를 평가했다. In-sample 최고는 Eutectic Si에 Model305 확률을 50% 혼합한 후보로
`78.6938%`였지만, 기준 `78.6921%`보다 겨우 `+0.0017%p`였다. 19개 specimen-group leave-one-out
결과는 `-0.2601%p`, positive `1/20`, bootstrap 95% CI `[-0.5508, -0.0483]%p`로 악화했다.
따라서 Model305 branch는 route에 포함하지 않는다.

결과: `$KOMAP_WORKSPACE/outputs/model306_model305_route_probe/results.json`

## 모델 307: 확률만 쓰는 image-wise expected-IoU decoder (P100 완료, 미채택)

Model303 train grouped-OOF 확률만으로 이미지별 class bias를 좌표 탐색했다. 각 후보는 저해상도 확률의
plug-in ratio-of-expectations mIoU를 최대화하고, Al은 고정 기준 클래스로 두었다. 모든 이미지의 bias를
확정한 뒤에만 train GT mask를 열어 채점했다. 따라서 mask를 통한 per-image bias 선택은 하지 않았다.

OOF image-wise mIoU가 `77.5096%`에서 `76.4843%`로 `-1.0252%p` 하락했고, positive image는 `10/70`,
specimen-group bootstrap 95% CI는 `[-1.3485, -0.6934]%p`였다. Eutectic Si class-mean IoU 변화가
`-2.4764%p`로 가장 컸다. 보조 측정인 공식 valid도 `75.0472%`에서 `73.7333%`로 `-1.3139%p`
하락했다. 확률만으로 선택한 기대-IoU surrogate가 실제 image-wise mIoU로 일반화하지 않아 미채택한다.
class 확률의 보정 오차와 픽셀 간 공간 상관을 surrogate가 반영하지 못한 탓일 수 있다(가능성에 대한 해석).

구현: `model307_expected_iou_decoder.py`
결과: `$KOMAP_WORKSPACE/outputs/model307_expected_iou_decoder/results.json`

## 모델 308: frozen DINOv2 다층 dense feature 잔차 branch (P100 완료, route 미채택)

과거 Model4의 frozen DINOv2-Small + 단순 저해상도 decoder(`42.3769%`)와 구별되도록, Model303의
specimen-group OOF CNN checkpoint는 동결하고 DINOv2-Small의 3/6/9/12번째 block patch feature를
작은 residual decoder로 변환해 CNN logits에 더했다. residual 마지막 층을 0 초기화해 시작 logits가
기준 CNN과 정확히 같음을 확인했다. 사전학습 DINOv2 patch 특징과 중간층 추출은 공식 구현을 사용했다.
[DINOv2 공식 코드·가중치는 Apache-2.0 라이선스](https://github.com/facebookresearch/dinov2#license).

네 fold 모두 Model303의 기록된 group split 및 checkpoint로 시작했고, valid mask는 모든 학습·예측과
checkpoint 결정을 끝낸 뒤에만 평가했다. baseline과 candidate 모두 512px patch, stride 384, Gaussian
overlap 및 D4 TTA를 동일하게 적용했다. 70장 train OOF image-wise mIoU는 같은 512px baseline
`76.7647%`에서 DINO branch `76.9273%`로 `+0.1626%p`였다(41/70 positive, specimen-group bootstrap
95% CI `[+0.0003, +0.3500]%p`). class-mean IoU 변화(Al3Ni, Primary Si, Al, Eutectic Si)는
`[-0.0450, +0.5099, +0.0939, +0.0917]%p`로 Primary Si 개선에 대부분 기댄다.

공식 valid 20장도 같은 512px 조건에서 `73.2189%` → `73.4223%` (`+0.2034%p`, positive 9/20)이지만
group-bootstrap 95% CI `[-0.1164, +0.5795]%p`는 0을 포함한다. valid class별 변화는
`[-0.1810, +0.2102, +0.1092, +0.6750]%p`로 OOF 양상과 다소 다르다. 512px 점수는 기존 1024px
Model303 결과나 최고 route `78.7209%`와 직접 비교할 수 없다. branch를 채택하기 전에 고정 Model225
route에 섞었을 때 specimen-group leave-one-out에서도 이득인지 Model309에서 확인한다.

P100 GPU6에서 약 1626초(27분) 실행했다. 구현: `model308_dinov2_dense_fusion_oof.py`
결과: `$KOMAP_WORKSPACE/outputs/model308_dinov2_dense_fusion/results.json`
로그: `$KOMAP_WORKSPACE/outputs/model308_dinov2_dense_fusion_gpu6.log`

## 모델 309: Model308 DINO branch의 Model225 route 보완성 (P100 서버 완료, 미채택)

Model308 valid 확률을 기존 Model225 route에 class-wise 혼합하는 43개 exact 후보를 19-specimen-group
leave-one-out으로 평가했다. 첫 시도에는 잘못하여 Model217의 5-step TENT 확률을 route 구성에 넣어
Model225 기준 점수를 재현하지 못했다. 이 결과는 폐기하고, Model225에서 실제 사용한 Model220 config 6
(20-step TENT, lr `5e-5`; `valid_probs_06.npz`)으로 바로잡아 재실행했다. 수정된 evaluator는 Model225
기준 image-wise mIoU `78.6921%`를 정확히 재현했다.

43개 후보 중 in-sample 최고도 기준 Model225 route 그 자체였다. 19개 held-out specimen 모두 기준 route를
선택했고, held-out 변화는 `0.0000%p`, positive image `0/20`, group-bootstrap 95% CI `[0, 0]%p`였다.
따라서 DINO branch는 현 최고 route를 보완하지 못해 미채택한다.

잘못된 입력을 쓴 초기 probe: `$KOMAP_WORKSPACE/outputs/model309_model308_route_probe/`
수정 완료 결과: `$KOMAP_WORKSPACE/outputs/model309_model308_route_probe_tent20/results.json`

## 모델 310: Model308 DINO branch의 원래 1024px scale 전이 (P100 완료, 미채택)

512px로 학습된 Model308 adapter를 재학습 없이 원래 최고 모델 추론 조건인 1024px patch/stride 864/
Gaussian overlap/D4에 적용했다. 기준으로 다시 생성한 CNN 예측은 저장된 Model303 OOF 및 valid 확률과
평균·최대 absolute 차이가 모두 `0.0`으로 정확히 일치해 scale 비교 parity를 확인했다.

Train OOF image-wise mIoU는 1024px baseline `77.5096%`에서 DINO branch `77.4823%`로
`-0.0272%p`였다(31/70 positive, group-bootstrap 95% CI `[-0.1854, +0.1376]%p`). 공식 valid는
`75.0472%`에서 `75.0194%`로 `-0.0278%p`였다(6/20 positive, 95% CI
`[-0.2604, +0.2329]%p`). 512px 학습 branch는 1024px 적용 시 일반화 이득이 없었으며 Model309 route
CV도 개선되지 않아 DINO residual 접근은 종료·미채택한다.

P100 GPU6 추론은 약 1343초(22분)였다. 채점 단계의 mask 경로 전달 버그는 저장된 전체 확률로만 재채점해
수정했으며, inference를 반복하지 않았다. 구현: `model310_dinov2_1024_scale_probe.py`
결과: `$KOMAP_WORKSPACE/outputs/model310_dinov2_1024_scale_probe/results.json`

## 모델 311: Eutectic Si train-OOF hard crop mining (P100 완료, 미채택)

Model303의 네 fold checkpoint를 이어 학습하면서, 각 train 이미지에 대해 해당 specimen을 제외해
만든 grouped-OOF 확률로 Eutectic Si false-negative/false-positive 위치를 mining했다. 이미지당 epoch당
기존 random crop 6개에 OOF hard crop 2개를 추가하되 총 8개 crop 수는 유지했다(1024px patch, batch 2,
4 epoch/fold). Valid mask는 학습·hard crop 선택·fold 판단에 사용하지 않고, 전체 예측이 끝난 뒤 보조
점수 산출에만 사용했다.

70장 grouped OOF image-wise mIoU는 Model303 `77.5096%`에서 `77.5029%`로 `-0.0066%p` 변했다
(positive `35/70`, specimen-group bootstrap 95% CI `[-0.1421, +0.1220]%p`). 사실상 동률이며 개선
증거는 없다. 클래스별 변화(Al3Ni, Primary Si, Al, Eutectic Si)는 `[+0.0355, +0.0036, +0.0232,
-0.0888]%p`로, 집중 개선 대상인 Eutectic Si도 소폭 하락했다. 공식 valid는 `75.0472%`에서
`75.0778%`로 `+0.0306%p`였지만, 20장 보조 valid의 작은 변동만으로 채택하지 않는다. 따라서 OOF hard
crop mining은 현재 설정에서 효과가 확인되지 않아 미채택한다.

P100 GPU6 네 fold 학습·추론에 약 `3871초`(64.5분) 소요. 구현: `model311_eutectic_oof_hard_mining.py`
결과: `$KOMAP_WORKSPACE/outputs/model311_eutectic_oof_hard_mining/results.json`
확률: `train_oof_probs.npz`, `valid_fold_ensemble_probs.npz`

## 모델 312: EfficientNet-B4 3배 해상도 (P100, 20/28 epoch 조기 종료)

팀 실험 문서의 3배 확대 결과(3 seeds 평균 `78.79%`)를 현재 B4 학습기에 맞춰 점검한다. Model280의 4배
실험은 계획된 7,840 optimizer updates 중 5,600 updates(20/28 epoch)에서 중단됐고 최고 valid
image-wise mIoU가 `71.3317%`였다. 이번에는 seed·구조·patch 크기·crop sampling·augmentation·LR를
Model280과 맞추고 배율만 4배에서 3배로 바꿔 끝까지 학습한다. 설정은 EfficientNet-B4 U-Net,
1536px patch, 4 crops/image, batch 1, uniform-area sampling, metallography augmentation, LR `5e-5`,
400-update warmup + cosine, 28 epochs(7,840 optimizer updates), D4 TTA/Gaussian overlap이다.
image-wise mIoU를 4 epoch마다 평가·checkpoint 선택에 사용하며 empty union은 0으로 처리한다.
test 추론은 하지 않는다. 다만 이 실행은 팀 점수의 정확한 재현은 아니다. 팀 문서는 3-seed 평균,
whole-image reflect-pad + D4 추론, GT에 없는 클래스를 image-wise 평균에서 제외하는 규칙을 보고한다.
Model312는 단일 seed, 1536/1152 tile + Gaussian overlap 추론을 쓰고, 빈 클래스 IoU를 0으로 둔다.
valid 20장 중 Primary Si가 없는 이미지는 A71 한 장이라 결측 처리만으로 생길 수 있는 차이는 최대
`1.25%p`; 따라서 그 규칙만으로 큰 성능 격차를 설명할 수 없다. Model312 점수는 현재 학습·타일 추론
recipe 안에서의 3배 확대 성능으로 해석하고, 팀 기록과 직접 동급 비교하지 않는다.

epoch4/8/12/16/20 train loss는 `0.6141/0.5230/0.4958/0.4860/0.4849`이고 valid image-wise mIoU는
`69.638/70.953/70.368/70.503/69.676%`다. 최고 epoch8보다 epoch20이 `-1.277%p` 낮고, epoch16에서
회복하지 못한 뒤 epoch20에서 다시 하락했다. epoch20 global class IoU는 Al3Ni `70.40%`, Primary Si
`82.68%`, Al `91.87%`, Eutectic Si `49.39%`다. 같은 seed의 Model280 4배 최고는 `71.332%`
(epoch16)로 본 run과 유사한 낮은 수준이었다. 두 고해상도 B4 run 모두 최고 route Model225
`78.6921%`에 크게 못 미쳐, 후반 8 epoch는 중단했다. best checkpoint는 epoch8, last checkpoint는
epoch20이며 test 추론은 하지 않았다.

실행: P100 GPU2 (`CUDA_VISIBLE_DEVICES=2`)
진행 로그: `$KOMAP_WORKSPACE/outputs/model312_effb4_3x_match_gpu2.log`
결과 경로: `$KOMAP_WORKSPACE/outputs/model312_effb4_3x_match/`

## 모델 313: Model312 체크포인트의 1792px 타일 추론 (valid only)

Model312 best(epoch8)을 고정하고 scale 3, D4/Gaussian을 유지한 채 추론 tile/stride만
1536/1152에서 1792/1344로 키웠다. fixed class-3 bias 0 기준 image-wise mIoU는
`71.2653%`로 Model312 epoch8 `70.9530%` 대비 `+0.3123%p`였다(global `75.0738%`). valid에 맞춰
bias를 다시 고르면 `71.7080%`가 되지만, 이는 같은 20장에 대한 calibration 점수이므로 일반화 이득으로
보지 않는다. 타일 문맥 확장은 작은 개선만 보였다. valid 확률은 저장했고 test 추론은 생략했다.

결과: `$KOMAP_WORKSPACE/outputs/model313_context_1792/`

## 모델 314: Model312 체크포인트의 whole-image D4 추론 (valid only)

팀 문서의 whole-image reflect-pad + D4 추론을 분리 검증하기 위해 Model312 best(epoch8)을 고정하고
추론만 whole-image로 바꿨다(scale 3, 32 배수 reflect pad, D4). fixed class-3 bias 0 기준 image-wise
mIoU는 `70.5038%`(global `74.6954%`)로, 같은 체크포인트의 1536px 타일 기준 `70.9530%`보다
`-0.4492%p`, 1792px 타일보다 `-0.7615%p`였다. 따라서 낮은 단독 점수의 주원인이 타일 문맥 부족이라는
가설은 지지되지 않았다. 큰 valid 이미지(1440×1024)를 포함해 전체 20장 추론이 P100에서 완료됐고,
test mask는 생성하지 않았다. valid 확률을 Model225 route에 추가할지 specimen-group CV를 수행 중이다.

결과: `$KOMAP_WORKSPACE/outputs/model314_whole_image/`
구현: `encoder_structure_search.py --inference-mode whole-image`

Model314 whole-image 확률을 Model225 route의 Al3Ni/Primary/Eutectic source로 넣는 15개 exact
class-wise 조합(gamma `0.30/0.50`)도 비교했다. 최고는 기존 Model225 route 그대로 `78.6921%`였고,
19개 specimen-group leave-one-out fold 모두 기존 route를 선택했다(held-out 변화 `0.0000%p`,
positive image `0/20`, group-bootstrap 95% CI `[0.0000, 0.0000]%p`). 따라서 B4 고해상도 branch는
단독 성능뿐 아니라 최고 route 보완성도 확인되지 않아 채택하지 않는다.

route 결과: `$KOMAP_WORKSPACE/outputs/model314_whole_image/route_probe.json`

## 모델 315: Model257 ResNet34 native-stem whole-image D4 추론 (P100, 미채택)

Model314와 달리 전체 이미지 통계로 학습한 Model257 체크포인트를 대상으로, `Model239`의
full-resolution native stem을 포함한 정확한 구조로 체크포인트를 strict-load하고 scale 1 전체 이미지에
D4 확률 평균을 적용했다. valid 20장 image-wise mIoU는 `73.9255%`, global mIoU는 `78.1414%`로,
기존 Model257 patch/stride D4 추론의 `77.2071%`보다 `-3.2816%p` 낮았다. 최대 이미지 1440×1024에서
추론은 가능했지만 CUDA peak allocated memory가 약 `15.89 GB`(14.79 GiB)로 높았다. 결과적으로
전체 이미지 문맥을 한 번에 처리하는 것이 이 branch에는 도움이 되지 않았다.

저장 확률을 Model225 운영 route에 넣어 Al3Ni/Eutectic/Primary class-wise 14개 혼합 후보를 추가
비교했다. 20장 in-sample 최고와 19개 specimen-group leave-one-out 모두 기존 Model225 route였고,
CV 변화 `0.0000%p`, positive image `0/20`, group-bootstrap 95% CI `[0.0000, 0.0000]%p`였다.
따라서 Model315 추론은 독립 branch와 route 보완성 양쪽 모두 개선 근거가 없어 미채택한다. test는
추론하지 않았다.

P100 GPU2 추론은 20장 약 72초. 구현: `model315_model257_whole_image.py`
확률·요약: `$KOMAP_WORKSPACE/outputs/model315_model257_whole_image_valid/`
route 결과: `$KOMAP_WORKSPACE/outputs/model315_wholeimage_route/results.json`

## 모델 316: 클래스 균형 reliable-pixel TENT (P100 완료, 미채택)

Model195를 이미지별로 reset하고 BN affine만 20 step, lr `5e-5`로 unlabeled adaptation했다.
기존 entropy 평균과 달리 각 pseudo-class에서 entropy 하위 75% 픽셀만 선택하고, 선택된 클래스별
평균 entropy를 동일 가중해 다수 클래스가 gradient를 독점하지 않게 했다. 각 이미지 mask는 적응과
추론이 끝난 뒤에만 평가에 사용했다. 이는 EATA의 신뢰도 기반 샘플 선택에서 착안한 segmentation용
pixel-level 변형이며, Fisher anti-forgetting 항은 이미지마다 모델을 reset하므로 적용하지 않았다.

독립 valid branch는 image-wise `76.6111%`, global `79.3284%`로 기존 20-step TENT
(`76.7921%`)보다 낮았다. 확률을 Model225에 넣은 14개 class-wise 후보도 기준 route
`78.6921%`보다 높지 않았다. 19개 specimen-group CV delta는 `-0.0145%p`, positive image `0/20`,
group-bootstrap 95% CI `[-0.0457, 0.0000]%p`로 비개선이었다. 따라서 신뢰 픽셀 선별·클래스 균형
adaptation은 현재 설정에서 기존 TENT보다 유효하지 않아 미채택한다. test는 추론하지 않았다.

P100 GPU2 valid 적응·추론은 약 192초. 구현: `model316_reliable_pixel_tent.py`
확률·요약: `$KOMAP_WORKSPACE/outputs/model316_reliable_tent_valid/`
route 결과: `$KOMAP_WORKSPACE/outputs/model316_reliable_tent_route/results.json`

## 모델 317: EfficientNet-B4 1배 해상도 재검증 (P100 GPU2, 14/28 epoch 중단)

도메인 적응·증류와 별도로 B4 기본 분할 학습을 다시 확인했다. ImageNet 사전학습 EfficientNet-B4 U-Net,
512px patch, scale 1, 이미지별 균등 4 crop, batch 1, LR `5e-5`, warmup 400 updates, CE+Dice,
동일 클래스 가중치, metallography augmentation, D4/Gaussian 타일 추론을 사용했다. 검증은 4 epoch마다
실행했고 test 추론은 생략했다.

epoch 4/8/12의 fixed-bias image-wise mIoU는 `65.2347/66.7332/66.5161%`였고, 저장된 최고는 epoch 8의
`66.7332%`다. epoch 12에서 개선이 멈춰 14/28 epoch에서 중단했다. `best.pt`는 epoch 8, `last.pt`는
epoch 14까지 저장되어 있으며 최종 calibration/summary와 test 산출물은 만들지 않았다. 현 최고 route
Model225 `78.6921%`에 미달하므로 채택하지 않는다.

이 결과는 팀 EfficientNet-B4 기준선과 직접 같은 조건의 재현으로 보지 않는다. 팀 문서는 원본 면적에
비례해 crop을 배분하고 마지막 checkpoint를 평가에 사용했다고 기록했지만, 이 run의 `uniform_area`는
각 이미지에서 같은 횟수로 crop을 뽑았다. 공식 클래스 색상-이름 매핑은 코드와 범례가 일치함을 확인했다.
따라서 다음 비교는 적응/증류가 아니라 crop 표집 프로토콜과 희귀상(Eutectic Si) 경계 오류를 겨냥한
지도학습 설정에서 시작한다.

결과: `$KOMAP_WORKSPACE/outputs/model317_effb4_1x_7840/`

## 모델 318: EfficientNet-B4 3배 해상도 면적비례 crop 표집 (P100 GPU2, 8/28 epoch 중단)

팀 실험의 원본 픽셀 면적 비례 crop 표집을 재현하기 위해 Model312와 같은 seed `280`, EfficientNet-B4,
1536px patch/scale 3, batch 1, 280 crops/epoch, LR `5e-5`, warmup 400 updates,
기본 hard class weights, CE+Dice, metallography augmentation 및 D4/Gaussian 추론을 유지하고 표집 방식만
`uniform_area`에서 `area_weighted`로 변경했다. 기존의 이미지별 고정 배정 대신 매 crop마다 이미지 픽셀
면적에 비례해 이미지를 복원추출한 뒤 이미지 안에서 중심점을 균일 추출한다. 입력 shape와 클래스 인덱스를 확인하는 smoke test를
통과한 뒤 P100 GPU2에서 학습했다.

Model318의 fixed-bias image-wise mIoU는 epoch4 `70.9250%`, epoch8 `68.5019%`였다. 같은 조건의 Model312는
epoch4 `69.6376%`, epoch8 `70.9530%`다. 따라서 면적비례 표집은 epoch4에서 `+1.2874%p`였지만 epoch8에는
Model312보다 `-2.4511%p`로 역전했다. epoch4 global class IoU 변화(Al3Ni/Primary Si/Al/Eutectic Si)는
`-0.42/+2.06/-0.21/+1.28%p`였으나, epoch8에는 `-2.62/-7.34/+0.21/-0.49%p`로 Primary Si가 크게
하락했다. 학습 loss가 낮아져도 validation mIoU가 유지되지 않아 epoch8에서 중단했다. 최고 checkpoint는
epoch4이며, epoch4 valid 확률은 별도 추론으로 저장했다. best checkpoint의 valid상 bias sweep 최고
`71.2385%`(Al bias `-0.35`)는 같은 valid에 맞춘 값이라 독립 성능으로 보지 않는다. test 추론은 하지 않았다.

구현: `improve_five_causes.py`의 `area_weighted` sampling mode
결과: `$KOMAP_WORKSPACE/outputs/model318_effb4_3x_area_weighted/`
epoch4 valid 확률: `$KOMAP_WORKSPACE/outputs/model318_area_weighted_best_e4_valid/`

## 모델 319: Model318/312 확률의 Model225 route 보완성 (P100 완료, 미채택)

Model318 epoch4 및 Model312 epoch8의 동일 추론 프로토콜(1536px tile, stride 1152, D4/Gaussian) valid
확률을 생성해 Model225 기준 route와 비교했다. 두 B4 branch 중 하나 또는 양쪽을 Al3Ni/Primary Si/
Eutectic Si 클래스별로 30%/50% 혼합하는 35개 exact 후보를 평가했다. 기준 Model225 image-wise
mIoU `78.6921%`를 재현했고, in-sample 최고도 기준 route 자체였다. 19개 specimen-group leave-one-out에서
모든 fold가 기준 route를 선택했으며 held-out delta `0.0000%p`, positive image `0/20`, group-bootstrap
95% CI `[0.0000, 0.0000]%p`였다. 따라서 표집 변경 branch는 단독 성능과 기존 route 보완성 모두 채택 근거가
없다.

구현: `model254_highres_route_probe.py` (candidate ID 인자 추가)
결과: `$KOMAP_WORKSPACE/outputs/model319_area_weighted_route_probe/results.json`
Model312 대조 확률: `$KOMAP_WORKSPACE/outputs/model319_uniform_area_control_e8_valid/`
test 추론은 생략했다.

## 모델 320: Al3Ni/Eutectic Si 전역 문맥 게이트 grouped-OOF 진단 (P100 CPU, 미채택)

도메인 적응·증류와 별도의 구조 개선 가설로, 이미지 전체 조성/밝기 문맥이 Al3Ni와 Eutectic Si의
혼동 보정에 유효한지 저비용으로 확인했다. Model297의 train-only specimen-group OOF 확률을 고정 입력으로
사용하고, 네 개 specimen-group fold 각각에서 held-out 그룹을 제외해 이진 logistic expert를 학습했다.
비교 대상은 픽셀 단위 확률·밝기·국소 통계만 입력한 expert와, 여기에 이미지 밝기 분위수, 예측 클래스
비율/평균 확률 및 각 상의 밝기·국소 변동 통계를 더한 expert다. 수정은 기준 예측에서 Al3Ni 또는
Eutectic Si로 분류된 픽셀에만 적용했다. 학습 표본도 두 GT 상에서 클래스별 최대 384픽셀로 균형 추출했다.
평가는 train 70장/59개 specimen 그룹 전체의 4-class image-wise mIoU이며, 95% 구간은 그룹 paired
bootstrap 10,000회로 계산했다. 공식 valid와 test는 학습·선택에 사용하지 않았다.

기준 Model297 OOF mIoU `77.2266%`에 비해 pixel-only expert는 `76.4177%`(`-0.8090%p`, 그룹 bootstrap
95% CI `[-1.0453, -0.4854]%p`, 개선 이미지 `12/70`), global-context expert는 `76.3990%`
(`-0.8276%p`, 95% CI `[-1.1180, -0.4655]%p`, 개선 이미지 `14/70`)으로 모두 명확히 하락했다.
전역 문맥 추가와 pixel-only 간 차이도 `-0.0186%p`(95% CI `[-0.2238, +0.1814]%p`)로 사실상
구분되지 않았다. 네 fold 모두 두 logistic optimizer가 수렴했으며, 각 fold 점수도 기준보다 낮았다.

따라서 이 간단한 이미지 문맥 보정에는 일반화 근거가 없고, 이를 근거로 대규모 global-context decoder를
학습할 이유도 부족해 해당 방향은 보류한다. 다만 이 결과는 logistic 후처리 게이트만 검증했으므로,
학습된 multi-scale 문맥 구조 전체가 무효라고 단정하지는 않는다. 다음 개선은 추가 DA/증류보다 지도학습
오류 분석—특히 어려운 표본의 라벨·상 경계 일관성, 희귀 상 혼동의 재현 가능한 train-OOF 보정성—을
우선해 근거가 확인되는 가설만 확장한다.

구현: `model320_global_context_gate.py`
결과: `$KOMAP_WORKSPACE/outputs/model320_global_context_gate/results.json`

## 팀 코드 ZIP 분석: `KoMaP_code.zip` (2026-09-26)

압축 파일에는 22개 코드·문서 파일(가중치/checkpoint와 예측 결과는 없음)이 들어 있다. 팀이 기록한
최고 `79.12%`는 3배 EfficientNet-B4 세 시드와 4배 한 시드의 probability ensemble이며, 팀의
`common.py`는 이미지별 네 class IoU 중 union이 0인 항목을 `NaN`으로 만들어 평균에서 제외한다.
우리 기본 기록은 empty union을 0으로 두므로 같은 Model225 route도 `78.6921%`다. 팀 방식처럼 결측
항목을 빼면 `79.4693%`다. 실제로 A71 한 장에서 Primary Si GT와 예측이 모두 없어 이 IoU만 `0/0`이다.
따라서 팀의 `79.12%`와 우리 `78.6921%`를 그대로 비교하면 안 되며, 결측 제외 기준에서는 기존 Model225가
`0.3493%p` 높다. 대회 산식은 결측 class의 처리(0/제외/1)를 명시하지 않으므로, 고정 4-class/empty=0과
empty 제외 값을 함께 기록한다. 팀 최고 점수는 ZIP 안에 prediction/checkpoint가 없어 독립 재채점하지 못했다.

모델·학습 recipe도 다르다. 팀 상위 분기는 SMP U-Net + EfficientNet-B4, scale 3, crop 1536, batch 2,
16,000 optimizer iterations, AdamW LR `3e-4`, unweighted CE+Dice이고, 전체 resized image D4 추론을 쓴다.
증강은 affine scale `0.7–1.4`와 임의 회전, D4, brightness/contrast, gamma, blur/noise다. 반면 우리의
최고 Model225는 ResNet34 기반 지도 branch와 20-step TENT rare-class 확률을 class-wise route로 합친
방식이다. 입력 정규화도 팀은 회색 3채널에 동일 mean/std (`0.449/0.226`)를 쓰고, 우리 RGB 경로는
ImageNet 채널별 통계를 적용해 온 차이가 있다. 우리가 별도로 시험한 B4 3배 runs는 patch 1536이어도
batch 1, hard class weights, LR
`5e-5`, 짧은 학습량/중간 checkpoint 평가였으므로 팀의 16k-step recipe 재현이 아니다(예: Model312는
20/28 epoch=5,600/7,840 planned updates에서 중단, best standalone `70.9530%`). 이 낮은 결과만으로
팀 B4 접근을 배제하지 않는다.

팀 코드의 `queue_round7`/`run_seeds.sh`에는 per-image width 4000 배율 정규화와 uniform image sampling도
후보로 있으나, ZIP의 실험 기록에는 그 결과가 없어 검증된 `79.12%`에 포함된 방법으로 취급하지 않는다.
다음 팀-B4 재검증은 고정 3배·긴 학습·batch 2·강한 affine/광도 증강·grayscale 공통 정규화를 맞춘 단일
branch부터 하고, 결과 확률을 Model225 route에 더해 specimen-group leave-one-out으로 보완성을 판단한다.
그 후에만 width-normalization/uniform-sampling 조합을 확장한다.

## 모델 321: Model303 boundary branch 연속 회전 증강 grouped-OOF (P100 완료, 미채택)

일반적인 D4 변환으로 만들 수 없는 임의 각도 회전이 현미경 시야 방향 변화에 강건성을 더하는지
확인했다. Model303의 specimen-group 4-fold OOF checkpoint에서 각 fold를 4 epoch씩 이어 학습했고,
1024px patch, batch 2, LR 배수 0.25, hard class weights와 CE+Dice는 고정한 채 입력에는 bilinear,
마스크에는 nearest-neighbor로 동일한 무작위 각도(±180°)를 적용했다. 각 train 이미지는 자신이 학습에
들어가지 않은 fold 모델의 예측만 OOF로 받았으며, checkpoint는 사전 고정한 마지막 epoch를 사용했다.
따라서 valid mask는 학습이나 선택에 쓰지 않았다. 점수는 고정 4-class image-wise mIoU, empty union=0이다.

70장 grouped-OOF에서 Model303 `77.5096%` → Model321 `77.2998%`로 `-0.2097%p`였고, 개선 이미지는
`26/70`, specimen-group paired-bootstrap 95% CI는 `[-0.4859, +0.0713]%p`였다. 20장 valid fold ensemble도
Model303 `75.0472%` → Model321 `74.6942%`(`-0.3530%p`)로 하락했다. OOF 구간은 0을 포함하지만 두
평가 모두 평균 변화가 음수이므로 임의 회전은 현 설정에서 채택하지 않는다.

결과: `$KOMAP_WORKSPACE/outputs/model321_rotation_aug_oof/results.json`
확률: `train_oof_probs.npz`, `valid_fold_ensemble_probs.npz` (동일 디렉터리)

## Validation mIoU 결측 클래스 정책 전수 재채점 (2026-09-26)

공식 산식의 `0/0` 규칙이 미정이라 앞으로 validation 결과에는 `empty union=0`과
`empty class 제외(NaN skip)`를 함께 적는다. 같은 per-image mean-of-four-class-IoUs 정의를
유지하고, 결측 처리만 바꾼다.

P100의 191개 `results.json`에서 20장 per-image score vector 5,002개를 다시 평균했다.
기존 보고값과 `1e-9` 이상 차이 나는 항목은 0개, 읽기 오류도 0개였다(동일 설정의
`best_image`/`best_exclude`/`records` 중복 포함). 별도로 188개 NPZ를 살펴 20개 validation
이미지 예측이 모두 들어 있는 111개 probability archive를 mask와 다시 대조해 두 정책 점수를
계산했다. 모델 채널 순서 차이는 팀 B4 채널 `[Al, Al3Ni, Primary Si, Eutectic Si]`와 기존
모델 채널 `[Al3Ni, Primary Si, Al, Eutectic Si]`를 분리해 처리했다.

추가 전수 재실행 결과도 `miou_policy_audit/rerun_20260926_user/`에 별도 보관했다. 범위와
오류 건수는 같았고 최고 route는 두 결측 정책 모두 Model270(`78.7209% / 79.4978%`), 최고
단독 archive는 Model289(`77.4181% / 78.0528%`)로 유지됐다. 당시 학습 중인 Model322는
완성된 valid 예측이 없어 두 재집계에 포함되지 않았다.

최고 raw validation route는 Model270이며 `empty=0` **78.7209%**, `empty=skip` **79.4978%**다.
하지만 Model225 기준 route와 비교한 19-specimen-group leave-one-out 변화가 `-0.1320%p`,
95% CI `[-0.3548, +0.0574]%p`, 개선 이미지 `9/20`이므로 채택하지 않았다. 현재 일반화 기준으로
유지하는 Model225는 두 정책에서 각각 **78.6921% / 79.4693%**다. 완전한 단독 probability archive
중 최고는 Model289 `77.4181% / 78.0528%`였으며, 고점 route의 저장 점수 벡터와 혼동하지 않는다.

상세 표와 재채점 스크립트: [miou_policy_audit.md](miou_policy_audit.md),
`recompute_miou_policy.py`, `recompute_miou_recorded_results.py`.

## 모델 322: 팀 EfficientNet-B4 3× grouped-OOF 재현 (완료)

팀 제공 3× 레시피를 4-fold specimen-group CV로 재현했다: EfficientNet-B4 U-Net, crop 1536, batch 2, AdamW `3e-4`, unweighted CE+Dice, affine/D4/광도 증강, fold당 16,000 updates. P100 호환을 위해 BF16을 FP16+GradScaler로 바꾸고, 공식 valid를 checkpoint 선택에 쓰지 않았다.

- 70장 grouped OOF: image-wise mIoU empty-union=0 `78.5351%`, NaN-skip `78.7825%`.
- 20장 valid 단독: empty-union=0 `77.85%`, NaN-skip `78.57%`.
- 약한 Model303 parent와 grouped OOF 비교: 단독 `78.53%` 대 `77.51%`; CV에서 고른 branch 혼합 route `79.35%` (+`1.84%p`, 95% CI `[+1.33,+2.40]%p`, 64/70장 개선). 이는 현재 최고 Model270 route와의 비교가 아니다.
- fold1 18장 preliminary 비교(Model303 parent): empty=0 `−0.40%p` (95% CI `[-1.48,+0.62]`), NaN-skip `+0.56%p` (95% CI `[-1.33,+3.45]`); 단일 fold라 결론 근거로 쓰지 않는다.

구현: [model322_team_b4/KoMaP](../experiments/model322_team_b4/KoMaP/) · OOF 결과: [JSON 결과](../results/outputs/model322_team_b4_group_oof/results.json)

## 모델 323: Model321 회전증강 branch를 Model270 route에 추가 (완료, 미채택)

Model322의 train OOF 비교 기준은 Model303 parent이므로, 기존 최고 raw route인 Model270에
보조 branch가 실제로 이득을 주는지 별도로 확인했다. `model323_model270_branch_route_probe.py`로
Al3Ni/Primary/Eutectic subset에 Model321 valid 확률을 30%·50% 섞는 14개 후보와 100% 대체 후보를
평가했다. Model270 기준 확률은 model195 Primary `0.75`, 20-step TENT rare branch(model220 설정,
`valid_probs_06.npz`) Al3Ni `0.40`, model269 Eutectic `0.50` 및 고정 prior/edge 후처리다.
처음에는 5-step TENT archive를 잘못 연결해 baseline이 재현되지 않았으나, 잘못된 실행은
`invalid_tent5_diagnostic.json`으로 격리하고 20-step archive로 다시 계산했다.

정정된 baseline은 `empty=0` `78.7209%`, `empty=skip` `79.4978%`로 Model270 기록과 일치했다.
두 정책 모두 후보 최고는 기준 route 자체였고, 19개 validation specimen-group held-out에서
모든 fold가 기준을 선택했다(평균 변화 `0.0000%p`, positive `0/20`, group-bootstrap 95% CI
`[0.0000, 0.0000]%p`). 따라서 Model321 회전증강 branch는 Model270을 보완하지 못한다.
이는 validation 내부 group-CV이지 독립 test 보장은 아니다.

구현: [model323_model270_branch_route_probe.py](../experiments/historical_scripts/model323_model270_branch_route_probe.py)
결과: [Model321 route probe JSON](../results/outputs/model323_model270_model321_fusion_tent20/route_probe.json)
P100 원본: `$KOMAP_WORKSPACE/outputs/model323_model270_model321_fusion_tent20/route_probe.json`

후속으로 Al branch까지 포함한 32개 후보를 평가했다. raw 최고는 두 정책 모두 baseline Model270
그대로였고, 19개 validation specimen-group held-out route 선택 변화는 두 정책에서 모두
`-0.0384%p`(positive `0/20`, group-bootstrap 95% CI `[-0.1213, 0.0000]%p`)였다.
19 folds 중 A153만 Al 50% 혼합을 골랐으며, held-out 전체 평균은 오히려 하락했다. 따라서
Model321의 Al-inclusive fusion도 미채택이다. 이 결과는 Model322에 같은 route evaluator를 쓸 때
Al을 후보에서 빼지 않도록 확인해 준다.

4-class 결과: [route_probe_all4.json](../results/outputs/model323_model270_model321_fusion_tent20/route_probe_all4.json)
Model322 B4 확률도 생성되면 team channel order를 main 순서로 재배열해 같은 32개 후보를 평가한다.
이 Model323 실험은 우리 Model321 branch와 Model270의 융합이며, 팀원 모델과 우리 모델 간 융합은 아니다.

## 모델 324 후보: GRAF-inspired Primary-Si inter-particle gap weighting (사전검사 후 미실행)

Al-Si 미세조직의 인접 Primary-Si 입자 사이 간극을 morphology로 찾고 gap-region CE를 높이는
GRAF/GRWS 아이디어는 기존 실험 기록에 없었다. 원 논문은 인접 입자 병합을 줄이기 위해 gap 영역
가중치를 사용했고, 저자 실험에서 weight 10을 사용했다. 특허 설명은 각 Primary-Si instance를
개별 팽창한 뒤 쌍별 팽창 교집합을 합쳐 gap mask를 만드는 방식이다. [GRAF 논문](https://doi.org/10.1016/j.commatsci.2023.112620),
[gap-region 방법 특허](https://patents.google.com/patent/CN113808140B/en).

초기 native-resolution train GT 사전 점검에서는 56/70장에 gap 픽셀이 있었지만, 실제 Model302
학습 최소 변 길이 1024와 반경 5px 정의로 train OOF를 다시 점검했다. 70장·59 specimen group에서
Model302의 baseline OOF image-mean mIoU는 `77.4562%`, gap 영역을 정답으로 완벽 교정하는
oracle도 `77.4675%`에 그쳤다(**+0.0113%p**, specimen-group bootstrap 95% CI
`[+0.0076, +0.0153]%p`). 학습 해상도에서 gap 픽셀은 평균 `0.0382%`였고 그중 오분류 비율은
`10.88%`였다. 검증 정답을 이용한 조정 없이 train OOF만 쓴 상한 진단에서도 사전 기준 `+0.3%p`에
크게 못 미쳐 Model324 학습은 취소했다. native-resolution 비율 `0.0929%`는 실제 학습 해상도
효과를 반영하지 못한 예비 통계로만 취급한다.

구현(`model302_boundary_oof_finetune.py`)과 no-training oracle 진단
(`model324_gap_oracle_audit.py`)은 보존한다. oracle 결과가 작아 추가 학습은 하지 않는다.
결과 JSON: [results.json](../results/outputs/model324_gap_oracle_model302_oof/results.json).

## Model270 검증 오류 분해 (진단 전용; 모델 학습 아님)

Model270 route를 저장된 확률 archive로 재구성해 기준 image-mean mIoU `78.7209%`를 정확히
재현했다. 검증 20장·19 specimen group에서 클래스별 image-mean IoU는 Al3Ni `75.15%`,
Primary Si `86.04%`, Al `93.74%`, Eutectic Si `59.95%`였다. GT에 존재하지만 예측이 전혀 없는
클래스는 한 건도 없어, 주된 문제는 클래스 누락보다 경계/영역 혼동이다. 한 클래스를 GT로 완벽
교정하는 진단상 상한 기여는 Eutectic Si `+14.04%p`, Al3Ni `+12.98%p`, Al `+7.27%p`,
Primary Si `+4.08%p` 순이었다. 이는 달성 가능한 개선치가 아닌 오류 집중도 참고치다.

GT 경계를 반경 1px 무시하는 비공식 민감도 진단은 `87.21%`(+8.49%p), 2px는 `84.59%`,
3px는 `81.16%`였다. 대회 지표를 바꾸는 결과가 아니라 경계 픽셀 민감성을 보여주는 참고값이다.
진단에서는 validation GT로 모델·route·threshold를 조정하지 않았다. 최저 이미지도 Primary Si
GT가 없는 경우였으며, empty-union=0 정책 때문에 해당 class IoU는 0으로 들어간다.

결과: [model325_model270_error_audit.json](../results/outputs/model325_model270_error_audit/results.json)
구현: [model325_model270_error_audit.py](../experiments/historical_scripts/model325_model270_error_audit.py)

## 모델 325 설계: 팀 B4 재현 + Model270 확률 융합

Model322의 grouped-OOF 학습·validation 확률이 완성되는 즉시, 팀원 제공 EfficientNet-B4 U-Net
레시피의 재현 모델과 우리 Model270 route를 결합한다. 팀 기록상 최고 79.12%는 3× 3-seed와 4×
branch의 앙상블이며 팀 evaluator의 empty-class NaN-skip 값이다(우리의 empty=0 점수와 직접 비교 불가).
공유된 `KoMaP_code.zip`에는 그 checkpoint들이 포함돼 있지 않다.
따라서 우선 Model322의 4-fold 3× B4 재현 예측을 팀 방식 branch로 사용하며, 원본 팀 checkpoint
앙상블과 동일한 가중치라는 뜻은 아니다.
기존 `model323_model270_branch_route_probe.py`의 고정 Model270 구성 및 팀→주 채널 재배열을
재사용하고, 4개 클래스 부분집합·혼합비 0.30/0.50 및 B4 전체 대체를 평가한다. 결측 클래스는
IoU 0과 NaN-skip 정책을 모두 계산하며, image-wise 네 클래스 평균을 유지한다. Validation의
최고 단일 route 점수만으로 채택하지 않고 specimen-group held-out CV 개선과 CI도 함께 판정한다.

자동 순차 실행기: [run_model325_team_b4_fusion_after322.sh](../experiments/runners/run_model325_team_b4_fusion_after322.sh)
평가 결과는 `$KOMAP_WORKSPACE/outputs/model325_team_ours_fusion/route_probe.json`에 저장한다.

## 모델 325 결과: 팀 B4 재현 + Model270 확률 융합 (완료, 일반화 개선 미확정)

첫 실행은 팀 B4 확률 archive의 이미지 키(`A117_step4_sample2`)와 기존 branch archive의 키
(`A117_step4_sample2_image`) 표기 차이로 중단됐다. 의미상 같은 ID임을 확인해 `_image` 접미사 정규화 및
키 집합 일치 검사를 추가했고, 네 archive 모두 동일한 valid 20개 ID를 포함하는 것을 검증한 뒤 재실행했다.
구현: [model323_model270_branch_route_probe.py](../experiments/historical_scripts/model323_model270_branch_route_probe.py)

Model270 기준 valid는 empty-union=0 `78.72%`, NaN-skip `79.50%`다. 32개 후보 중 같은 valid set에서 가장 높은
조합은 Al3Ni·Primary Si·Al 확률에 B4를 각 50% 섞고 Eutectic Si는 기존 Model269 branch로 유지한 route로,
`79.16%` / `80.00%`(반올림; empty=0 / NaN-skip)를 얻었다. 다만 이는 같은 valid를 사용해 고른 in-sample 최고점이다.
19개 specimen-group leave-one-group-out route 선택의 평균 개선은 empty=0 `+0.382%p`(14/20장 개선),
group-bootstrap 95% CI `[-0.111,+0.898]%p`였다. CI가 0을 포함하므로 Model270 대비 안정적인 이득은 입증되지 않았고
앙상블 route는 아직 채택하지 않는다. 전체 결과: `$KOMAP_WORKSPACE/outputs/model325_team_ours_fusion/route_probe.json`.

## 모델 326: 팀 B4 recipe + full-image-stat 입력

EfficientNet-B4 U-Net에 `[raw grayscale, full-image z-score, full-image empirical CDF]`를 입력하고, 나머지는 Model322의 3×·crop1536·batch2·AdamW·CE+Dice 레시피로 고정했다. 4-fold specimen-group OOF는 fold당 16,000 updates이며 validation mask로 중간 선택하지 않았다. 이 16,000은 비교 기준을 맞추기 위해 상속한 budget이지 update 수 탐색으로 검증된 최적값이 아니다.

- 단독 valid: empty=0 `78.0400%`, NaN-skip `78.7090%`; 70장 OOF empty=0 `78.5297%`.
- Model322과 OOF paired 차이: empty=0 `−0.0050%p`, 95% group-bootstrap CI `[-0.2850,+0.2915]%p` (개선 35/70); 단독 입력 변경 효과는 확인되지 않았다.
- Model270 class-wise fusion valid: empty=0 `79.1937%`, NaN-skip `80.0186%`; Model270 대비 group-CV `+0.4728%p`, 95% CI `[+0.0933,+0.8838]%p`, 15/20장 개선. 현재 robust한 최고 후보이나 empty=0 80%에는 미달한다.

구현/결과: [model326_hybrid_fullstats](../experiments/model326_hybrid_fullstats/KoMaP/) · [JSON 결과](../results/outputs/model326_hybrid_fullstats_group_oof/results.json)

## 모델 327: B4 + Model270 클래스별 혼합비 탐색 (CPU 완료, 미채택)

322의 valid B4 확률과 Model270 route를 이용해 기존 325의 후보군을 미세화한다. Al3Ni·Primary Si·Al의
B4 혼합 가중치를 서로 독립적으로 `{0.30, 0.50, 0.70}`에서 조합하고, Eutectic Si는 검증된 Model269
branch를 그대로 유지한다. 기존 32개 route를 포함해 총 59개 설정을 동일한 image-wise empty=0 지표로
평가하며, 19개 specimen-group leave-one-group-out 선택 및 group bootstrap CI를 함께 본다.
이 후보군은 Model325 탐색에서 얻은 클래스 선택 정보를 사용한 후속 탐색이라 valid 재사용에 따른 탐색 편향이
있을 수 있다. 따라서 raw 최고점만으로 채택하지 않고 grouped held-out 근거와 함께 탐색 결과로 보고한다.

전용 옵션 `--classwise-grid`를 [model323_model270_branch_route_probe.py](../experiments/historical_scripts/model323_model270_branch_route_probe.py)에
추가했고 compile/help 및 동일 20개 이미지 키 집합 preflight를 통과했다. 실행은 `CUDA_VISIBLE_DEVICES=`와
낮은 CPU 우선순위로 시작해 GPU 학습 자원을 점유하지 않았다. 결과:
`$KOMAP_WORKSPACE/outputs/model327_b4_classwise_route/route_probe.json`.
후보 59개를 모두 평가했다. in-sample 최고는 Al3Ni B4 `0.50`, Primary Si B4 `0.70`, Al B4 `0.70` 혼합이고
Eutectic Si는 Model269 branch 유지로, valid empty-union=0 image-mean mIoU `79.1898%`(NaN-skip `80.0330%`)다.
이는 같은 valid에서 고른 최고점이며 공식 empty=0 기준으로도 80% 미만이다. specimen-group held-out route 개선은
`+0.4265%p`, positive `15/20`, group-bootstrap 95% CI `[-0.0574,+0.8919]%p`로 0을 포함한다.
18/19 fold가 동일한 클래스별 가중치를 선택했지만 독립 개선 증거가 아직 충분하지 않아 미채택 상태다.

## 모델 328: B4 target-width 4000 + image-uniform crop (P100 실행 중)

팀 코드에 남아 있으나 이전에 검증되지 않은 두 제안(per-image resize width `4000`, crop image sampling
`uniform`)을 묶어 평가한다. 이는 두 요인의 개별 효과를 분리하는 실험은 아니지만, image-wise 평균 지표에 맞춰
이미지별 학습 기회를 같게 하고 미세조직 크기/배율 편차를 정규화하는 후속 후보다. B4 U-Net, 1536 crop,
batch 2, AdamW `3e-4`, CE+Dice 및 16,000 update의 나머지 팀 recipe를 유지하며, Model322와 같은 4-fold
specimen-group OOF 및 20장 valid 평가를 쓴다. 각 fold 학습 중 valid 평가는 끄고 마지막 update를 사용한다.
Model328 단독 평가는 기존 grouped-CV 비교(Model303 parent)로 수행하고, 완료 직후 CPU-only Model270
확률 route 탐색도 자동 실행하도록 후속 실행기를 보완했다. 따라서 새 B4 branch의 단독 성능뿐 아니라
현재 최고 route와의 valid 점수 및 specimen-group held-out 개선·신뢰구간까지 함께 판정한다.

Model326+Model270 route가 empty=0 `80%` 및 grouped-CV 95% CI 하한 `>0` gate를 통과하지 못해 시작했다.
Model328 완료 후 같은 gate로 판정하며, 미통과 시에만 Model329를 이어서 실행한다.

실행기: [run_model328_width4000_uniform_group_oof.sh](../experiments/runners/run_model328_width4000_uniform_group_oof.sh)
순차 평가기: [queue_model328_after326.sh](../experiments/runners/queue_model328_after326.sh)

## 모델 329 조건부 설계: full-stat 입력 + width4000 + uniform sampling

Model326의 `[raw, full-image z-score, full-image CDF]` 입력 표현과 Model328의 per-image `target_width=4000`,
image-uniform crop sampling을 결합한다. EfficientNet-B4 U-Net, crop 1536, batch 2, AdamW `3e-4`, CE+Dice,
fold당 16,000 updates 및 specimen-group 4-fold OOF 프로토콜을 유지한다. 이 결합 실험은 개별 효과 분리보다
세 기법의 조합이 80% 목표에 도달하는지 검증하는 후속 시도다.

Model328+Model270 route가 empty-union=0 valid `80%` 이상과 grouped-CV 95% CI 하한 `>0`을 함께 만족하지
못할 때만 실행하도록 Model328 완료 후 queue에 연결했다. 시작 전 P100 GPU 6의 14.5 GiB 여유를 3회 연속
확인하고 100-step 학습 및 full-image 추론 smoke를 통과시킨 뒤 본 학습에 진입한다. 학습·OOF·valid 완료 후
Model270 route를 CPU 평가한다.

초기 기록 정정: `predict.py`의 `scores.json` 필드 `mean_miou`는 NaN-skip이며 공식 empty-union=0이 아니다. 따라서 이 절에 먼저 적었던 fold1 `77.9351% / 76.8802%`와 `+1.0549%p`는 NaN-skip 수치였고 empty=0으로 잘못 표기했다. 아래 값은 저장된 예측 mask와 GT mask를 직접 다시 비교해 official empty=0으로 산출했다.

fold 1의 동일 held-out 18장에서 Model322 `75.9170%` → Model329 `76.9348%`, paired `+1.0178%p` (10/18장 개선)였다. fold 2 18장은 Model322 `77.3068%` → Model329 `78.4686%`, `+1.1618%p` (11/18장 개선)였다. 두 fold 36장 합계는 Model322 `76.6119%` → Model329 `77.7017%`, paired `+1.0898%p`, 21/36장 개선, 30 specimen groups, group-bootstrap 95% CI `[+0.4014,+1.8531]%p`다.

클래스 순서 `[Al, Al3Ni, Primary Si, Eutectic Si]`에서 paired class-IoU 변화는 각각 `+0.192`, `+1.761`, `+0.187`, `+2.219%p`다. 95% group-bootstrap CI는 Al `[-0.135,+0.580]`, Al3Ni `[+0.714,+2.992]`, Primary Si `[-0.317,+0.788]`, Eutectic Si `[+0.853,+3.718]%p`다. 이는 train-only OOF의 두 fold 중간 결과(36/70장)이지 valid 점수나 최종 grouped OOF 결론이 아니며, 전체 Model329 성능 또는 80% 달성을 뜻하지 않는다.

재현 코드: [model340_partial_oof_metric_audit.py](../experiments/model340_partial_oof_metric_audit.py); [JSON 결과](../results/outputs/model340_partial_oof_audit/folds1_2.json).

실행기: [run_model329_hybrid_width4000_uniform_group_oof.sh](../experiments/runners/run_model329_hybrid_width4000_uniform_group_oof.sh)

## 모델 330: 팀 train 픽셀 빈도 기반 class-prior correction (미채택)

팀원이 설명한 방식대로 train mask 70장의 클래스 픽셀 비율을 구해 역수를 class weight로 두고, 현재 최고 valid route인 Model327의 최종 클래스 확률을 weight로 나눈 뒤 argmax했다. 클래스 순서는 팀 `[Al, Al3Ni, Primary Si, Eutectic Si]`에서 route `[Al3Ni, Primary Si, Al, Eutectic Si]`로 재배열했다. 기존 route의 최종 logits에 `-log(class_weight)`를 더하는 것은 확률 나눗셈 후 argmax와 동치다.

train 픽셀 비율(팀 순서)은 `[57.28%, 26.31%, 9.43%, 6.98%]`, 역빈도 weight는 `[1.746, 3.800, 10.606, 14.333]`이다. 재현 검증에서 보정 전 Model327 점수가 기록과 정확히 일치했다: image-mean mIoU, empty-union=0 기준 `79.1898%`(NaN-skip `80.0330%`). 보정 적용 후 `74.8745%`(NaN-skip `75.6938%`)로 각각 `-4.3153%p`, `-4.3392%p` 하락했다.

클래스별 image-mean IoU(empty=0)는 Al3Ni `76.36→74.41%`, Primary Si `86.37→85.13%`, Al `94.01→91.96%`, Eutectic Si `60.02→48.00%`였다. 특히 이미 약한 Eutectic Si에서 역빈도 weight 나눗셈이 예측을 크게 억제해 악화됐다. 따라서 이 보정은 Model327에 적용하지 않는다. 이는 valid 20장 결과이며 test 점수는 아니다.

모델 329 번호는 이미 full-stat + width4000 + uniform 학습 대기 작업에 배정되어 있어, 이 진단은 모델 330으로 정정했다.
평가 스크립트: [model330_train_prior_correction.py](../experiments/historical_scripts/model330_train_prior_correction.py)
결과: [result.json](../results/outputs/model330_train_prior_correction/result.json)

## 모델 331: B4 class-wise 혼합률 미세 탐색 (미채택)

157개 route를 비교했다. raw 최고는 Model322 B4를 Al3Ni `0.50`, Primary Si `0.70`, Al `0.80` 혼합하고 Eutectic Si에는 Model269 `0.50`을 유지한 설정이다. valid 20장 점수는 empty=0 `79.1914%`, NaN-skip `80.0359%`로 Model327보다 각각 `+0.0016%p`, `+0.0029%p`뿐이다. Model270 대비 specimen-group CV 변화는 empty=0 `+0.441%p` (95% CI `[-0.070,+0.960]`, 16/20장 개선), NaN-skip `+0.483%p` (95% CI `[-0.112,+1.156]`)로 CI가 모두 0을 포함한다. 따라서 미채택이며, NaN-skip 숫자만으로 80% 달성 또는 robust 개선이라 하지 않는다.

결과: [route_probe.json](../results/outputs/model331_b4_fine_gamma_route/route_probe.json)

## 모델 332: Model322 grouped-OOF Eutectic Si 오차 기하 감사 (CPU, 진단)

공식 validation은 전혀 열지 않고, Model322의 70장 train grouped-OOF 확률과 train masks만 사용했다(59 specimen groups, class order `[Al, Al3Ni, Primary Si, Eutectic Si]`, empty-union IoU=0). 전체 image-wise mIoU는 `78.5346%`; 이미지별 class-IoU 평균은 Al `92.7727%`, Al3Ni `79.3328%`, Primary Si `86.3145%`, Eutectic Si `55.7186%`다. Eutectic Si class mean의 specimen-group bootstrap 95% CI는 `[52.3836%, 59.1447%]`다.

Eutectic GT 내부 경계를 8-connectivity로 추출하고 2px 팽창한 띠에서 GT Eutectic pixels의 `95.71%`, Eutectic false-negative pixels의 `93.85%`가 나타났다. 반면 Eutectic false-positive pixels 중 그 띠 안에 든 비율은 `36.72%`뿐이다. 즉 얇은 Eutectic 상의 누락은 경계 주변에 집중되지만 오탐의 다수(`63.28%`)는 GT Eutectic 경계에서 2px보다 떨어져 있어 경계 손실만으로 설명되지 않는다. 경계 띠/바깥의 pooled Eutectic IoU는 각각 `65.2259%`/`12.8812%`다.

가장 큰 양방향 혼동은 Al3Ni다. GT Eutectic을 놓친 픽셀은 Al3Ni `746,385`(FN의 약 `53.3%`), Al `402,633`, Primary Si `250,904`개다. 예측 Eutectic 오탐의 원래 GT는 Al3Ni `603,093`(FP의 약 `53.6%`), Al `265,838`, Primary Si `256,151`개다. 따라서 다음 해석은 Eutectic needle boundary와 Al3Ni 질감/상 구분을 함께 봐야 하며, 단순 class bias나 경계 보조 loss의 반복으로 해결된다고 가정하지 않는다. Model307의 확률-only expected-IoU decoder는 OOF 및 valid에서 악화했고, Al3Ni–Eutectic pairwise margin/clDice/boundary 실험도 과거에 robust 개선을 보이지 않아 재실행하지 않는다. 현재 대기 중인 width4000+uniform 실험이 상/조직 크기 표현을 바꾸는 효과를 먼저 검증하며, Model326/328 OOF 완료 뒤 동일 감사를 반복 비교한다.

구현: [model332_eutectic_oof_error_geometry.py](../experiments/historical_scripts/model332_eutectic_oof_error_geometry.py)
결과: [audit.json](../results/outputs/model332_eutectic_error_geometry/audit.json)

Claude Code Opus 자문(읽기 전용)도 반영했다. 제안된 확률-only per-image expected-IoU decoder는 Model307에서 이미 grouped OOF `-1.0252%p`, valid `-1.3139%p`로 악화했고, 이미지별 bias oracle도 Model172에서 최대 `+0.5260%p`에 그쳐 재실행하지 않는다. 지표 정렬형 loss는 Model283/284/286에서 valid standalone 소폭 상승이 있었으나 기존 route와 grouped-CV 일반화 이득은 확인되지 않았다. 현 Model326→328→조건부329 queue가 끝나도 robust 80% gate를 통과하지 못하면 후속 후보(Model333)는 Model322 각 fold checkpoint를 시작점으로 per-image Lovasz fine-tune과 동일 update 수 CE+Dice control을 4-fold paired OOF로 비교한다. 채택 조건은 specimen-group bootstrap CI 하한 `>0`, Eutectic Si class IoU 상승, 다른 class 평균 하락 `0.2%p` 이내이며 official valid는 OOF에서 선택한 설정의 최종 1회 확인에만 쓴다. HRNet seed ensemble은 Model327 OOF 확률 부재로 현 단계에서 뒤로 미룬다.

## 모델 333: Model322 B4 checkpoint paired Lovasz fine-tuning (조건부 대기)

Model326/328/329 이후에도 robust 80% gate를 통과하지 못할 때만 실행하도록 준비했다. 각 fold에서 동일한 Model322 checkpoint로부터 두 branch를 초기화해 loss 효과를 분리한다: control은 CE+Dice, candidate는 CE+Dice+`0.25 ×` per-image present-class Lovasz-Softmax다. 두 arm 모두 EfficientNet-B4 U-Net, ImageNet normalization, gray3 입력, scale 3, crop 1536, batch 2, AdamW `2e-5`, 2,000 updates 및 동일 seed의 deterministic worker를 쓴다. Lovasz 계산량을 제한하기 위해 이미지당 131,072개 픽셀을 균등 복원추출해 surrogate를 계산한다.

4-fold specimen-group OOF를 각각 만들고, `evaluate_group_cv.py`로 Lovasz와 동일-init control의 grouped held-out route를 비교한다. Lovasz 확률은 Model270 route와도 CPU-only로 비교하며, empty-union=0 image-mean mIoU와 group bootstrap CI를 기록한다. valid는 모든 fold 학습·OOF 이후 평가하고 test 추론은 하지 않는다. trainer의 shell/Python 문법, Lovasz forward/backward, 원격 `--help`, Model322 fold checkpoint 766-tensor strict load를 확인했다.

실행기: [run_model333_b4_lovasz_group_oof.sh](../experiments/runners/run_model333_b4_lovasz_group_oof.sh); 조건부 대기기: [queue_model333_after328.sh](../experiments/runners/queue_model333_after328.sh). 조건부 실행은 Model326→328→329 평가가 끝난 뒤 robust 80% gate를 확인한다. 코드는 [Model326/333 source](../experiments/model326_hybrid_fullstats/KoMaP/)에 있다.

## 모델 334: Model322/326 B4 확률 평균 + Model270 class-wise route

Model322 팀 B4와 Model326 full-stat B4의 valid 확률을 이미지별 50:50 평균해 후보 branch를 만들었다. 이를 Model270 route에 넣고 59개 class-wise 설정을 비교했다. 최고 설정은 `[Al3Ni, Primary Si, Al, Eutectic Si]` 순서로 B4 결합 branch를 `[0.50, 1.00, 0.50]` 비율로 섞고, Eutectic Si는 기존 Model269 branch `0.50`을 유지한 구성이다. 나머지 Model270 prior, beta, edge 및 cleanup 설정은 고정했다.

- valid 20장 image-wise mIoU: empty-union=0 `79.2353%`; empty-union 제외(NaN-skip) `80.0687%`.
- Model270 대비 specimen-group leave-one-group-out route 선택 결과: `+0.4642%p`, 15/20장 개선, group-bootstrap 95% CI `[+0.0433,+0.9137]%p` (19 groups).
- 이 route 후보 탐색은 같은 valid 20장의 점수를 활용했으며, NaN-skip 80%를 공식 empty=0 기준의 목표 달성으로 해석하지 않는다. 이 결과만으로 test 일반화가 입증된 것은 아니다.

결과: [route_probe.json](../results/outputs/model334_b4_equalblend_route/route_probe.json); 확률 입력 archive는 보존하지 않았다.

## 모델 335: Model270 × B4 branch 2,401개 exact route 전수 채점

Model322, Model326, 그리고 두 모델의 50:50 평균 확률을 Model270 route의 각 class에 독립적으로 적용하는 모든 discrete 조합을 채점했다. class마다 기존 Model270 유지 또는 세 branch 중 하나를 gamma `0.50`/`1.00`으로 혼합하는 7개 선택지를 두었고, 4개 class 전체 `7^4 = 2,401` routes를 만들었다. Eutectic Si도 포함했다. 모든 조합은 동일한 train-only prior/calibration과 exact Gaussian·bilateral spatial refinement 및 Primary component cleanup을 거친다. 점수 산출은 CUDA를 비활성화한 CPU 16-worker 실행이며 train/inference checkpoint는 새로 만들지 않았다.

| 선택 기준 | Al3Ni | Primary Si | Al | Eutectic Si | Global mIoU | Image-wise empty=0 | Image-wise empty 제외 |
|---|---|---|---|---|---:|---:|---:|
| strict image-wise 최고 | 322/326 평균 `0.50` | 326 `1.00` | 322 `0.50` | Model270 유지 | 81.8794% | **79.2903%** | 80.1248% |
| NaN-skip / global 최고 | 322/326 평균 `0.50` | 326 `1.00` | 322 `1.00` | Model270 유지 | **81.9359%** | 79.2898% | **80.1316%** |

Model270의 기존 Eutectic 설정은 Model269 gamma `0.50`이며 그대로 유지됐다. strict 최고 route는 Model334 대비 empty=0 `+0.0549%p`, NaN-skip `+0.0561%p`; Model270 기준 대비 각각 `+0.5693%p`, `+0.6270%p`였다. Global 최고 route는 Model334 global 최고 `81.8677%`보다 `+0.0682%p`다. Empty-union=0 기준 공식 목표 80%에는 아직 `0.7097%p` 부족하다.

평가 정책 차이 원인을 valid mask·route score 배열로 추가 확인했다. 20장 중 A71 한 장만 strict와 제외 점수가 다르며, 해당 mask에는 Primary Si가 없다. NaN-skip 최고 route에서 A71은 strict `50.5104%`, empty union 제외 `67.3471%`로 전체 점수 차이 `0.8418%p`를 설명한다. 이 Primary Si는 양쪽 모두 GT·예측 union이 0이므로 empty=0 정책 아래 이 클래스 점수는 개선 대상이 아니며, A71의 다른 세 class IoU와 나머지 이미지 개선이 필요하다.

19개 validation specimen group을 leave-one-group-out으로 route 선택한 held-out 비교도 기록했다. Model270 대비 image-wise strict 변화는 `+0.4564%p`(14/20장 양수, group-bootstrap 95% CI `[+0.0456,+0.8978]%p`), NaN-skip은 `+0.5060%p`(15/20장 양수, CI `[+0.0524,+1.0264]%p`)였다. Pooled global 비교 변화는 `+0.5484%p`(CI `[+0.2796,+0.8593]%p`)다. 이 검증은 route 선택 일반화를 보조 확인하지만, 2,401개 grid를 같은 validation pool로 비교했으므로 독립 test 평가가 아니다. Model328/329 같은 다른 branch는 이번 grid에 포함되지 않았다.

구현: [model335_exhaustive_b4_route_search.py](../experiments/historical_scripts/model335_exhaustive_b4_route_search.py)
결과: [results.json](../results/outputs/model335_exhaustive_b4_routes/results.json); 로그는 보존하지 않았다.

## 모델 336: 팀 HRNet-W48 + U-Net++ 레시피 재현 (P100 대기)

비공개 [`CNUKoMaP/HRNet_best`](https://github.com/CNUKoMaP/HRNet_best) commit
`97adcdf337ab78fb7ec1469fe48943ff9a6363ea`의 모델 구조와 핵심 학습법을 별도 디렉터리에 보존해
재현한다. 구조는 ImageNet 사전학습 HRNet-W48 encoder + SMP U-Net++ decoder와 학습 전용
class-agnostic boundary head다. 256px 원본 crop을 512px로 2배 확대하고, 희소상 중심 patch sampling,
unweighted `0.5 CE + 0.5 Dice + 1.0 boundary BCE/Dice`, AdamW `3e-4` (encoder/decoder 동일 LR),
cosine schedule 70 epochs를 적용한다. P100에서는 micro-batch 4, accumulation 2로 effective batch 8을
맞춘다. 추론은 원본 256px tile/170px overlap, Gaussian stitching, horizontal/vertical flip 4-way TTA다.

팀 README의 valid 20장 `81.77%`는 pooled/global confusion-matrix mIoU이고 image-mean 점수가 아니다.
따라서 이번 결과는 우리 공식 이미지별 4-class 산식으로 다시 계산하며, empty-union=0/제외/1 점수와
pooled 점수를 명시적으로 분리한다. 우선 `train`으로만 학습하고 `valid`를 최종 평가하며, 마지막에
`test` 입력에 대한 마스크를 생성한다. 기존 Model270/B4 route에 대한 유효성은 저장한 valid 확률맵으로
별도 탐색하되, valid로 고른 최고점은 독립 test 성능처럼 해석하지 않는다.

기존 HRNet 실험은 Model8 HRNet-W18 단순 FPN, Model14 W32 context head, Model15 W48 context head,
Model48 W18 native patch였으며, 이번의 HRNet-W48 + U-Net++ 조합과 달랐다. 서버 환경에서 HRNet-W48
U-Net++ forward shape `[1,4,64,64]` 확인, 약 76.96M parameter 구조 생성, pretrained encoder 로드를
확인했다. 실행기는 Model326→328→329→조건부333 GPU 큐가 끝난 뒤 P100 여유를 확인하고 순차 실행하도록
대기 중이다.

관련 `sppark` commit `e0d71b1`은 현재 DINOv3 global-context + local-detail CNN 코드를 포함하고, 문서에는
HRNet 특징과 stride-2/stride-1 detail 경로가 설계 후보로 적혀 있다. 보관된 `accuracy_v1` CNN baseline은
image-mean `78.4729%`지만, 해당 HRNet 설계는 측정 결과가 아니다. 이번에는 먼저 팀의 완성된 HRNet-W48
레시피를 통제된 branch로 재현한다. 성능 확인 뒤 필요하면 sppark식 원본 전체 문맥/로컬 세부 융합과 Model270
class-wise 확률 route를 차례로 비교한다.

Model336 평가 후에는 Model337 CPU-only route sweep을 잇는다. 네 상마다 Model270 유지 또는 HRNet 확률을
`0.25/0.50/0.75/1.00` 비율로 섞는 `5^4=625` 조합을 정확한 공간 후처리까지 채점한다. 사전에 이미지 순서,
정답 마스크 일치, 클래스 순서 및 확률 합을 확인하며, raw valid 최고점과 specimen-group leave-one-group-out
선택 결과를 나눠 기록한다.

코드/설정: [model336_hrnet_best](../experiments/model336_hrnet_best/)
route 평가기: [model337_hrnet_model270_routes.py](../experiments/historical_scripts/model337_hrnet_model270_routes.py)
서버 결과 예정 경로: `$KOMAP_WORKSPACE/outputs/model336_hrnet_best/`

## 모델 338: Model335 expert 혼합률 fine-sweep

Model335 strict-best route의 네 class 확률원을 고정하고 한 class씩 혼합률을 `0.1`~`1.0`으로 바꾸는 38개 고유 설정을 exact CPU 평가했다. 모델 학습이나 새 추론은 없고, 저장된 확률맵과 동일한 calibration/spatial refinement를 재사용했다.

공식 empty-union=0 최고는 **79.3400%**로, Model335 strict-best `79.2903%`보다 `+0.0497%p` 상승했다. 80% 목표에는 아직 `0.6600%p` 부족하다.

| 상 | 확률원 | 혼합률 |
|---|---|---:|
| Al3Ni | Model322/326 평균 | 0.50 |
| Primary Si | Model326 | 1.00 |
| Al | Model322 | 0.50 |
| Eutectic Si | Model269 | **0.90** |

참고로 NaN-skip 최고는 `80.1685%` (Eutectic gamma `0.80`), pooled global 최고는 `81.9667%` (strict-best와 같은 gamma `0.90` route)다. 공식 순위에는 strict empty=0만 사용한다. Model338 strict-best의 A71 strict score는 `49.6248%`, empty class 제외 score는 `66.1664%`; mask에서 비어 있고 모델도 예측하지 않은 Primary Si가 정책 차이의 원인이다.

19개 specimen group leave-one-out route 선택은 Model270 대비 strict `+0.5884%p`, 18/20 images positive, group-bootstrap 95% CI `[+0.2305,+0.9916]%p`였다. Route 탐색이 같은 valid pool에 의존하므로 독립 test 일반화 점수는 아니다. Model329 branch는 아직 포함되지 않았다.

코드: [model338_fine_gamma_route_search.py](../experiments/historical_scripts/model338_fine_gamma_route_search.py); 결과: [results.json](../results/outputs/model338_fine_gamma_route_search/results.json); 로그는 보존하지 않았다.

## 모델 339: Model338 네 클래스 혼합률 동시 탐색 (완료, 채택 보류)

Model338은 한 번에 한 클래스의 혼합률만 바꿨기 때문에 클래스 간 확률 정규화·argmax 상호작용을 탐색하지 못했다. Model338 공식 strict-best의 확률원과 calibration/spatial refinement는 고정하고 네 클래스 혼합률을 동시에 바꾸는 400개 조합을 CPU exact scorer로 비교한다. 기준에는 Model270 원본 route를 포함한다.

| 클래스 | 고정 확률원 | 탐색 gamma |
|---|---|---|
| Al3Ni | Model322/326 평균 | `0.3–0.7`, 0.1 간격 |
| Primary Si | Model326 | `0.7–1.0`, 0.1 간격 |
| Al | Model322 | `0.3–0.7`, 0.1 간격 |
| Eutectic Si | Model269 | `0.7–1.0`, 0.1 간격 |

각 조합을 exact spatial refinement까지 적용해 점수화하고 official image-wise empty-union=0, empty 제외 참고값 및 leave-one-specimen-group-out 선택 결과를 계산한다. 탐색 점수는 같은 valid 20장에서 생성되므로 raw 최고는 validation 선택 편향을 포함하며, grouped held-out CI와 함께 판정한다. 새 학습이나 GPU 추론은 하지 않는다. 사전 5-route smoke/preflight는 통과했다.

401개 route 탐색 공식 최고는 **79.3683%** (NaN-skip `80.2062%`, pooled global `82.0064%`)이며, Model338의 strict `79.3400%`보다 raw valid에서 `+0.0283%p`다. 최고 설정은 Al3Ni Model322/326 평균 `0.6`, Primary Si Model326 `1.0`, Al Model322 `0.7`, Eutectic Si Model269 `0.8`이다.

하지만 Model338 strict route를 고정 비교군으로 둔 19-group leave-one-specimen-out route selection 변화는 `−0.0139%p`, 개선 이미지 `9/20`, group-bootstrap 95% CI `[-0.0613,+0.0352]%p`로 0을 포함한다. 따라서 `+0.0283%p`는 같은 valid set의 raw 탐색 최고일 뿐이고, 일반화 개선 근거가 없어 Model338을 보수적 채택 기준으로 유지한다. Model339 route는 최고 관측값 표에는 남기되 안정적으로 우월한 route로 채택하지 않는다.

탐색은 CPU-only 저우선순위로 완료했다. 당시에는 Model329 branch를 포함하지 않았으며, 이후 완료된 Model329와의 626-setting 결합 탐색은 Model341에 기록한다.

코드: [model339_joint_gamma_route_search.py](../experiments/model339_joint_gamma_route_search.py); 직접 비교 감사 코드: [model339_grouped_cv_parent_audit.py](../experiments/model339_grouped_cv_parent_audit.py); 결과: [results.json](../results/outputs/model339_joint_gamma_route_search/results.json); Model338 paired group-CV: [cv_vs_model338.json](../results/outputs/model339_joint_gamma_route_search/cv_vs_model338.json)

## 모델 340: Model329 full OOF 공식 지표 재감사 (완료)

Model329의 초기 fold 비교에서 `predict.py` 결과 JSON의 `mean_miou`가 NaN-skip인데도 empty-union=0으로 잘못 표기된 것을 확인했다. Model340은 Model322와 Model329의 저장된 예측 mask를 각 fold GT와 직접 다시 비교하고, union=0 class IoU를 0으로 두어 공식 image-wise 지표를 재계산했다.

fold1은 18장 기준 Model322 `75.9170%` → Model329 `76.9348%` (`+1.0178%p`, 10/18장 개선), fold2는 18장 기준 `77.3068%` → `78.4686%` (`+1.1618%p`, 11/18장 개선)이다. 두 fold 합계 36/70장은 `76.6119%` → `77.7017%`, paired `+1.0898%p`, 21/36장 개선, 30 specimen groups, group-bootstrap 95% CI `[+0.4014,+1.8531]%p`였다.

`[Al, Al3Ni, Primary Si, Eutectic Si]` 평균 class-IoU 변화는 `[+0.192,+1.761,+0.187,+2.219]%p`다. Al3Ni CI `[+0.714,+2.992]`, Eutectic Si CI `[+0.853,+3.718]%p`는 양수이고, Al/Primary Si CI는 0을 포함한다. 이 partial audit 이후 네 fold 모두 완료해 70장 train OOF와 20장 valid를 재평가했다. Model329 단독 valid 공식 empty=0 image-mean은 **78.8334%** (absent class 제외 참고값 `79.5353%`), class별 `[Al, Al3Ni, Primary Si, Eutectic Si]`는 `[94.052,75.131,86.482,59.669]%`다. Train OOF 단독은 `79.1233%`; OOF에서 선택한 네 클래스 gamma `0.5`는 `79.3974%`다. Model303 parent 대비 specimen-group held-out route 선택은 `+1.8875%p`, 67/70장 개선, 59 groups, group-bootstrap 95% CI `[+1.4921,+2.2940]%p`였다.

Model270 expert 경로에 Model329를 섞은 별도 exact route probe의 strict valid 최고는 **79.2615%** (NaN-skip 참고 `80.07697%`)다. Model270 대비 group-held-out 선택 변화는 `+0.4305%p`, 17/20장 개선, 19 groups, 95% CI `[+0.1571,+0.7144]%p`로 양수지만 80% 목표 및 기존 raw 최고 Model339 `79.3683%`에는 미달한다. 따라서 이 점수만으로 기존 최고를 교체하지 않는다.

Model329 학습/OOF/valid 및 Model270 probe는 완료했다. Model341은 Model339 전문가 route에 Model329를 더하는 626-setting 탐색을 실행 중이며, Model342 OOF-fitted blend가 그 뒤를 기다린다. Model333 paired CE+Dice control 대 Lovasz fine-tuning도 시작했고 Model336 HRNet-W48/U-Net++는 Model333 뒤에서 대기 중이다.

재현 코드: [model340_partial_oof_metric_audit.py](../experiments/model340_partial_oof_metric_audit.py); partial audit: [folds1_2.json](../results/outputs/model340_partial_oof_audit/folds1_2.json); full OOF/valid: [results.json](../results/outputs/model329_fullstats_width4000_uniform_group_oof/results.json); Model270 route: [route_probe.json](../results/outputs/model329_model270_fusion/route_probe.json).
