# model_results0924_1-4 — 백본·적용 기법·상별 성능 종합

현재 `model_results_0924_1~3`은 S0·S1·S2의 학습 결과이다. 이 문서는 네 모델 비교를 위해 현재 참고 중인 [0921 ResNet34 U-Net++ + Residual ASPP](model_results_0921_1.md)를 네 번째 **기존 비교 모델**로 포함했다. S3 경계 손실 실험은 아직 실행되지 않았다.

## 1. 평가 기준

- 평가 데이터: 동일한 official validation **20장**.
- **image mIoU**: 각 이미지에서 네 클래스 IoU를 평균한 뒤, 그 값을 20장에 대해 평균한다.
- **상별 평균 IoU**: 해당 클래스의 이미지별 IoU를 20장에 대해 평균한다. 아래 표의 네 상별 평균을 다시 평균하면 image mIoU가 된다.
- `TP+FP+FN=0`인 클래스는 IoU=1로 계산한다. 정답에 없지만 오검출이 있으면 IoU=0이다.
- 모든 수치는 **%**, 차이는 **%p**다. 전체 픽셀을 합쳐 계산한 Global mIoU는 비교 지표로 사용하지 않았다.
- 주 비교표는 모두 **TTA 없는 Plain 평가**다. D4 결과는 같은 모델의 추가 추론 조건으로 따로 표시했다.

## 2. 네 모델 종합 비교

| 모델 / 결과 출처 | 학습 백본 | 구조 및 적용한 변경 | image mIoU | Primary Si 평균 IoU | Al3Ni 평균 IoU | Eutectic Si 평균 IoU | Al 평균 IoU |
|---|---|---|---:|---:|---:|---:|---:|
| **S0 · 0924_1** | EfficientNet-B4, ImageNet 사전학습 | 기본 U-Net, 입력 **1배**, CE + Dice, halo를 둔 패치 학습·타일 추론 | 70.5977 | 86.4877 | 58.9426 | 45.6697 | 91.2906 |
| **S1 · 0924_2** | EfficientNet-B4, ImageNet 사전학습 | S0와 같은 구조·학습 예산, 원본 시야를 유지하며 입력 **2배 확대** | 73.1106 | 88.2350 | 61.8263 | 50.1214 | 92.2596 |
| **S2 · 0924_3** | EfficientNet-B4, ImageNet 사전학습 | S0와 같은 구조·학습 예산, 원본 시야를 유지하며 입력 **3배 확대** | 73.3606 | 83.9165 | 64.8231 | 52.2603 | 92.4426 |
| **기존 ResNet34 모델 · 0921_1** | ResNet34, ImageNet 사전학습 | U-Net++ + Residual ASPP + 원해상도 stem, gray/z-score/CDF 입력, 단계별 학습, EMA/Raw 검증 선택 | **77.0150** | 84.7294 | **72.6095** | **56.6180** | **94.1030** |

**ResNet34 행은 기존 보고서의 수치를 그대로 옮긴 값이 아니라, 저장된 이미지별 confusion matrix에서 위 기준으로 다시 계산한 값이다.** 원본 보고서와의 차이는 5절에 설명했다. 모델 추론이나 재학습은 수행하지 않았다.

## 3. 모델에 적용한 내용

### S0 / S1 / S2: 입력 배율을 비교한 단일 모델 실험

**공통 구조**

- torchvision EfficientNet-B4의 ImageNet `IMAGENET1K_V1` 사전학습 encoder와 기본 U-Net decoder를 사용했다.
- Encoder의 stride 2/4/8/16/32 특징을 skip connection으로 연결한다. Decoder는 bilinear upsampling, 특징 연결, Conv-BN-ReLU 두 번으로 구성하며 채널은 256/128/64/32/16이다.
- 흑백 이미지를 3채널로 복제하고 ImageNet mean/std로 정규화한다.
- Encoder BatchNorm의 running statistics를 고정하고, 메모리 절약을 위한 activation checkpointing을 적용했다. Encoder 가중치와 BN affine parameter는 학습한다.
- 출력은 단일 4클래스 segmentation head다. ASPP·경계 보조 head·앙상블은 추가하지 않았다.

**모델별 차이**

| 모델 | 원본 중심 시야 | 배율 적용 후 중심 입력 | halo 포함 실제 입력 | 변경 목적 |
|---|---:|---:|---:|---|
| S0 | 512×512 | 512×512 | 640×640 | 기본 해상도 기준선 |
| S1 | 512×512 | 1024×1024 | 1280×1280 | 시야를 유지하며 미세 조직을 더 큰 입력으로 표현 |
| S2 | 512×512 | 1536×1536 | 1920×1920 | 3배 확대가 미세 상과 경계 분할에 주는 효과 확인 |

입력 확대는 보간이며 초해상도 복원 모델을 사용한 것은 아니다. Halo는 원본 기준 사방 64px의 주변 문맥으로, 학습 손실은 중심 영역에서 계산한다. 각 모델은 동일한 ImageNet 초기 가중치와 seed에서 독립적으로 학습했다. S1/S2가 S0 학습 가중치를 이어받은 것은 아니다.

**공통 학습·추론 조건**

- Seed 2026, 8,000 optimizer updates, 유효 배치 8, micro-batch 1.
- AdamW: encoder LR 1e-4, decoder LR 3e-4, weight decay 0.01, warmup 250 + cosine, BF16.
- 손실: CE + Dice. 이번 세 모델은 경계 가중치가 0이며, 별도 경계 손실을 적용하지 않았다.
- 증강: D4 회전·반전과 약한 밝기·대비·감마. 학습 증강의 D4와 평가 시 TTA는 구분하며, 이번 평가에서는 TTA를 껐다.
- 마지막 8,000 update checkpoint로 평가했다. Validation 점수로 best checkpoint나 EMA를 선택하지 않았다.
- 추론: native core512/halo64, 50% overlap, Gaussian 가중 결합. 확대된 클래스 확률을 원본 크기로 복원한 뒤 argmax한다.
- S0는 1 GPU에서 2,000 updates를 수행한 뒤 6 GPU로 이어 학습했다. S1/S2는 처음부터 6 GPU다. 따라서 S0 대비 비교에는 GPU 수 변경에 따른 난수·BN 실행 이력 차이가 포함된다.

### 기존 비교 모델: ResNet34 U-Net++ + Residual ASPP

- **백본:** ImageNet 사전학습 ResNet34.
- **원해상도 stem:** stride 1의 32채널 분기를 추가해 입력 해상도의 특징을 유지한다.
- **U-Net++ decoder:** 여러 깊이의 특징을 밀집된 skip connection으로 결합한다.
- **Residual ASPP:** bottleneck에서 dilation 1/2/4/8 및 global pooling 분기를 결합하고 잔차로 더한다. 서로 다른 크기의 주변 문맥을 처리하기 위한 구성이다.
- **입력:** 단순 흑백 복제 대신 `[gray, image z-score, image CDF]` 3채널을 사용한다.
- **학습:** CE + Dice, AdamW, 유효 배치 8, BF16. 512 패치 2,000 → 768 패치 1,500 → full-frame 1,000 updates의 총 4,500 updates 일정이다.
- **EMA 및 checkpoint 선택:** EMA decay 0.99를 적용하고, 학습 중 validation에서 Raw/EMA 및 best checkpoint를 선택했다. S0/S1/S2의 마지막 checkpoint 고정 방식과 다르다.
- **추론:** full-frame Plain을 주 비교표에 사용했다. 동일 모델의 D4 8방향 TTA 결과는 아래에 별도로 제시한다.

| 기존 ResNet34 모델의 추론 조건 | image mIoU | Primary Si 평균 IoU | Al3Ni 평균 IoU | Eutectic Si 평균 IoU | Al 평균 IoU |
|---|---:|---:|---:|---:|---:|
| Plain | 77.0150 | 84.7294 | 72.6095 | 56.6180 | 94.1030 |
| D4 TTA | 77.6475 | 85.4387 | 73.0543 | 57.8807 | 94.2165 |

이 모델은 입력 표현, 구조, 학습 일정, checkpoint 선택, 추론 방식이 모두 다르다. 표의 차이를 백본 하나의 효과로 해석할 수는 없다.

## 4. 결과 해석

- **1배→2배:** image mIoU가 **+2.5129%p** 상승했다. 특히 Eutectic Si는 +4.4517%p, Al3Ni는 +2.8837%p 개선됐다.
- **2배→3배:** image mIoU는 **+0.2501%p**, Al3Ni는 +2.9968%p, Eutectic Si는 +2.1389%p 개선됐다.
- **S2의 Primary Si 평균 하락은 A71에 크게 영향받는다.** GT에 Primary Si가 없는 A71에서 S2가 2픽셀을 오검출해 해당 클래스 IoU가 1에서 0으로 바뀌었다. 이 항이 전체 image mIoU에 주는 차이는 1.25%p다. Primary Si가 존재하는 19장에서는 S1 87.6158 → S2 88.3332%로 개선됐다.
- 새 단일 U-Net 세 모델 중 관측 최고는 S2다. 다만 한 seed의 validation 결과이며, 3배의 안정적인 우월성을 확정한 것은 아니다.
- 같은 평가 공식을 적용한 기존 ResNet34 Plain은 S2보다 **+3.6543%p** 높다. 주된 차이는 Al3Ni와 Eutectic Si다. 추가 구조의 효과를 확인하려면 나머지 입력·학습·추론 조건을 맞춘 실험이 필요하다.
- 세 모델의 경계 및 외곽 오류에 대한 자세한 진단은 [0924_4 종합 분석](model_results_0924_4/model_results_0924_4.md)에 있다.

## 5. 0921 원본 보고서와 수치가 다른 이유

기존 0921 코드와 저장된 CSV를 확인한 결과, 두 가지 집계 차이가 있었다.

1. 기존 `image_mean_miou`는 **GT에 있는 클래스만** 이미지별 평균에 포함했다. A71은 Primary Si GT가 없어서, 이 클래스에 오검출이 있어도 평균에서 빠졌다. 본 표는 지침대로 네 클래스를 항상 포함한다. 저장된 confusion matrix에서 다시 계산하면 Plain **77.6752 → 77.0150%**, D4 **78.3186 → 77.6475%**가 된다.
2. 기존 summary의 `class_ious`는 validation 전체 confusion matrix를 합쳐 계산한 상별 IoU였다. 본 표의 상별 값은 **이미지별 클래스 IoU를 20장 평균**한 값이다.

따라서 원본 보고서의 전체 mIoU와 상별 값을 그대로 혼합하지 않았다. 원본 보고서·CSV·모델은 수정하지 않았으며, 이 종합 문서에서만 비교 기준을 통일했다.

## 6. 수치 및 설정 출처

- S0: [보고서](model_results_0924_1/model_results_0924_1.md), [점수](model_results_0924_1/summary.json), [설정](model_results_0924_1/request.json).
- S1: [보고서](model_results_0924_2/model_results_0924_2.md), [점수](model_results_0924_2/summary.json), [설정](model_results_0924_2/request.json).
- S2: [보고서](model_results_0924_3/model_results_0924_3.md), [점수](model_results_0924_3/summary.json), [설정](model_results_0924_3/request.json).
- 공통 분석: [0924_4](model_results_0924_4/model_results_0924_4.md).
- 기존 ResNet34: [원본 보고서](model_results_0921_1.md), [학습 설정](../experiments/resnet34_unetplusplus_aspp_v1/outputs/runs/resolved_config.json), [Plain 이미지별 혼동행렬](../experiments/resnet34_unetplusplus_aspp_v1/outputs/evaluations/plain/per_image_metrics.csv), [D4 이미지별 혼동행렬](../experiments/resnet34_unetplusplus_aspp_v1/outputs/evaluations/d4/per_image_metrics.csv), [기존 집계 코드](../experiments/resnet34_unetplusplus_aspp_v1/model_lab/evaluation.py).
