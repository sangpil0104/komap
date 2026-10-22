# model_results_0920 — 100개 모델 실험 결과

상태: SMOKE 기능 점검 — 본 실험 아님

주 지표는 공식 검증 20장의 이미지별 평균 mIoU입니다. Global mIoU는 별도 보조 지표입니다. 값은 %, 변화는 pp입니다.

구성: 추천 1개 + 기존/동료 결과 기반 19개 + 별도 탐색 80개. 100개의 서로 다른 실험 설정이며 모두 서로 다른 원 논문 모델 100종이라는 뜻은 아닙니다.

각 행은 추가 학습 중 최고 plain 검증 점수로 선택한 best_trained.pt를 plain/D4로 평가한 결과입니다. best.pt는 학습 전 update 0까지 포함한 최고 plain 체크포인트입니다. 부모 가중치가 선택된 행은 별도로 표시합니다.

기존 기준 CNN+D4: 이미지 평균 78.6865, Global 81.4989.
이전 seed2028+D4의 관측 최고: 이미지 평균 78.7858, Global 81.5082. 빈 클래스 처리와 A71 민감도를 함께 확인해야 합니다.

이번 관측 최고 후보: **m019_short_hard_patch** — 이미지 평균 **78.6869**, Global **81.4989**, 기준 대비 **+0.0004 pp**.
이전 이미지 평균 최고 78.7858 대비 -0.0988 pp. 관측 검증 최고이며 독립 테스트 성능을 뜻하지 않습니다.

## D4 순위

| 순위 | 실험 | 그룹 | 이미지 mIoU | Global mIoU | 기준 Δpp | 공통19장 Δpp | 개선 장수 | 초기 가중치 유지 | 학습 분 |
|---:|---|---|---:|---:|---:|---:|---:|---|---:|
| 1 | m019_short_hard_patch | local19 | 78.6869 | 81.4989 | +0.0004 | +0.0003 | 11/20 | 아니오 | 0.29 |
| 2 | m037_skip_gate_v1 | explore80 | 78.6868 | 81.4983 | +0.0003 | +0.0005 | 10/20 | 아니오 | 0.15 |
| 3 | m064_boundary_005 | explore80 | 78.6866 | 81.4989 | +0.0002 | +0.0003 | 11/20 | 아니오 | 0.16 |
| 4 | m016_aspp_w64_staged | local19 | 78.6865 | 81.4979 | -0.0000 | +0.0000 | 9/20 | 아니오 | 0.35 |
| 5 | m086_erase | explore80 | 78.6864 | 81.4979 | -0.0000 | -0.0001 | 10/20 | 아니오 | 0.21 |
| 6 | m023_ppm_v1 | explore80 | 78.6863 | 81.4972 | -0.0001 | -0.0000 | 10/20 | 아니오 | 0.17 |
| 7 | m015_aspp_w32_staged | local19 | 78.6863 | 81.4984 | -0.0002 | -0.0002 | 9/20 | 아니오 | 0.34 |
| 8 | m048_focal_0p5 | explore80 | 78.6862 | 81.4985 | -0.0003 | -0.0002 | 10/20 | 아니오 | 0.17 |
| 9 | m062_aux_01 | explore80 | 78.6859 | 81.4978 | -0.0005 | -0.0005 | 8/20 | 아니오 | 0.18 |
| 10 | m071_sgd_001 | explore80 | 78.6856 | 81.4975 | -0.0009 | -0.0008 | 8/20 | 아니오 | 0.19 |
| 11 | m001_recommended_fullframe | recommended1 | 78.6854 | 81.4973 | -0.0010 | -0.0010 | 8/20 | 아니오 | 0.18 |
| 12 | m091_scratch_residual_w32 | explore80 | 54.3904 | 58.4461 | -24.2961 | -25.1758 | 0/20 | 아니오 | 0.40 |
| 13 | m100_scratch_attention_w32 | explore80 | 33.6148 | 34.1504 | -45.0717 | -46.8747 | 0/20 | 아니오 | 0.39 |
| 14 | m097_scratch_convnext_w32 | explore80 | 15.6736 | 15.5110 | -63.0129 | -65.1811 | 0/20 | 아니오 | 0.31 |
| 15 | m094_scratch_separable_w32 | explore80 | 7.3063 | 7.5215 | -71.3802 | -73.2189 | 0/20 | 아니오 | 0.24 |

## 상위 후보의 클래스별 IoU

| 실험 | 초정 Si | Al3Ni | 공정 Si | Al |
|---|---:|---:|---:|---:|
| m019_short_hard_patch | 86.3952 | 74.4028 | 59.8499 | 94.0998 |
| m037_skip_gate_v1 | 86.3956 | 74.4025 | 59.8493 | 94.1000 |
| m064_boundary_005 | 86.3948 | 74.4032 | 59.8482 | 94.1004 |
| m016_aspp_w64_staged | 86.3945 | 74.4026 | 59.8485 | 94.1002 |
| m086_erase | 86.3948 | 74.4016 | 59.8495 | 94.0999 |
| m023_ppm_v1 | 86.3948 | 74.4021 | 59.8489 | 94.0995 |
| m015_aspp_w32_staged | 86.3949 | 74.4025 | 59.8479 | 94.0999 |
| m048_focal_0p5 | 86.3942 | 74.4022 | 59.8482 | 94.1001 |
| m062_aux_01 | 86.3940 | 74.4007 | 59.8494 | 94.0997 |
| m071_sgd_001 | 86.3948 | 74.4005 | 59.8474 | 94.0997 |

## 평가 해석

- 기준 모델보다 높은 D4 후보: 3/15개.
- 초기 가중치를 plain 기준으로 넘지 못한 후보: 0/15개.
- 완료 모델의 학습 시간 합: 3.71분. 프로세스 시작·최종 평가 시간 등은 별도입니다.
- warm-start와 scratch는 시작 가중치 및 학습 예산이 다릅니다. 구조 효과만의 공정한 비교로 해석하지 마세요.
- 100개 후보와 여러 체크포인트를 같은 validation에서 선택했습니다. 최고값은 개발 점수이며 그룹 CV/새 검증셋과 반복 seed로 확인해야 합니다.
- A71의 빈 클래스 처리에 따른 점수 변화를 보기 위해 동일 19장 변화와 클래스별 유효 이미지 수도 CSV에 기록합니다.
- 공식 train/validation 사이 이미지 ID 중복은 없지만 일부 A-ID 그룹은 공유됩니다. 다른 fold는 해당 fold의 새 부모 모델부터 학습해야 합니다.
- 동료 문서의 79.48은 Global mIoU, 75.83은 이미지 평균입니다. 동일 평가 CSV·클래스 매핑이 없어 직접적인 통제 비교는 아닙니다.
- test 10장에는 정답이 없으므로 테스트 정확도는 보고하지 않습니다. 기존 모델은 자동 교체하지 않습니다.

## 100개 설정 및 실행 상태

| 번호 | 실험 | 상태 | 초기화 | 설명 |
|---:|---|---|---|---|
| 1 | m001_recommended_fullframe | 완료 | baseline | 추천: 부모 CNN에서 패치 없이 전체 영상 600 updates |
| 2 | m002_ff_short | 미완료 | baseline | 전체 영상 300 updates, LR 1e-6 |
| 3 | m003_ff_gentle_long | 미완료 | baseline | 전체 영상 1000 updates, LR 1e-6 |
| 4 | m004_ff_lr5e6 | 미완료 | baseline | 전체 영상 600 updates, LR 5e-6 |
| 5 | m005_ff_lr1e5 | 미완료 | baseline | 전체 영상 1000 updates, LR 1e-5 |
| 6 | m006_ema999 | 미완료 | baseline | 추천 일정 + EMA 0.999 |
| 7 | m007_ema95 | 미완료 | baseline | 추천 일정 + EMA 0.95 |
| 8 | m008_decoder_only | 미완료 | baseline | encoder 고정, decoder와 classifier 미세조정 |
| 9 | m009_head_only | 미완료 | baseline | classifier만 미세조정 |
| 10 | m010_encoder_slow | 미완료 | baseline | encoder LR은 decoder의 0.1배 |
| 11 | m011_seed2028_parent | 미완료 | seed2028 | 과거 seed2028 최고점에서 전체 영상 미세조정 |
| 12 | m012_photo_parent | 미완료 | photo | 과거 photo_mild 모델에서 전체 영상 미세조정 |
| 13 | m013_fullframe_parent | 미완료 | fullframe | 과거 fullframe 모델에서 전체 영상 미세조정 |
| 14 | m014_mild_photo | 미완료 | baseline | 전체 영상에서 약한 광도 증강 확률 0.3 |
| 15 | m015_aspp_w32_staged | 완료 | baseline | ASPP width 32: 새 블록 선행 학습 후 분리 LR |
| 16 | m016_aspp_w64_staged | 완료 | baseline | ASPP width 64: 새 블록 선행 학습 후 분리 LR |
| 17 | m017_light_boundary_ce | 미완료 | baseline | 경계 픽셀 CE 가중치 +0.2 |
| 18 | m018_light_class_weights | 미완료 | baseline | 강한 가중치 대신 CE [1,1.05,1.1,1] |
| 19 | m019_short_hard_patch | 완료 | baseline | 낮은 LR 768 hard patch 200 후 전체 영상 600 |
| 20 | m020_aux_light | 미완료 | baseline | 낮은 비중 0.05의 다중 해상도 보조 학습 |
| 21 | m021_aspp_v1 | 미완료 | baseline | aspp 경량 변형 1, 전체 영상 학습 |
| 22 | m022_aspp_v2 | 미완료 | baseline | aspp 경량 변형 2, 전체 영상 학습 |
| 23 | m023_ppm_v1 | 완료 | baseline | ppm 경량 변형 1, 전체 영상 학습 |
| 24 | m024_ppm_v2 | 미완료 | baseline | ppm 경량 변형 2, 전체 영상 학습 |
| 25 | m025_large_kernel_v1 | 미완료 | baseline | large_kernel 경량 변형 1, 전체 영상 학습 |
| 26 | m026_large_kernel_v2 | 미완료 | baseline | large_kernel 경량 변형 2, 전체 영상 학습 |
| 27 | m027_axial_v1 | 미완료 | baseline | axial 경량 변형 1, 전체 영상 학습 |
| 28 | m028_axial_v2 | 미완료 | baseline | axial 경량 변형 2, 전체 영상 학습 |
| 29 | m029_bottleneck_v1 | 미완료 | baseline | bottleneck 경량 변형 1, 전체 영상 학습 |
| 30 | m030_bottleneck_v2 | 미완료 | baseline | bottleneck 경량 변형 2, 전체 영상 학습 |
| 31 | m031_se_v1 | 미완료 | baseline | se 경량 변형 1, 전체 영상 학습 |
| 32 | m032_se_v2 | 미완료 | baseline | se 경량 변형 2, 전체 영상 학습 |
| 33 | m033_eca_v1 | 미완료 | baseline | eca 경량 변형 1, 전체 영상 학습 |
| 34 | m034_eca_v2 | 미완료 | baseline | eca 경량 변형 2, 전체 영상 학습 |
| 35 | m035_cbam_v1 | 미완료 | baseline | cbam 경량 변형 1, 전체 영상 학습 |
| 36 | m036_cbam_v2 | 미완료 | baseline | cbam 경량 변형 2, 전체 영상 학습 |
| 37 | m037_skip_gate_v1 | 완료 | baseline | skip_gate 경량 변형 1, 전체 영상 학습 |
| 38 | m038_skip_gate_v2 | 미완료 | baseline | skip_gate 경량 변형 2, 전체 영상 학습 |
| 39 | m039_decoder_refine_v1 | 미완료 | baseline | decoder_refine 경량 변형 1, 전체 영상 학습 |
| 40 | m040_decoder_refine_v2 | 미완료 | baseline | decoder_refine 경량 변형 2, 전체 영상 학습 |
| 41 | m041_highres_v1 | 미완료 | baseline | highres 경량 변형 1, 전체 영상 학습 |
| 42 | m042_highres_v2 | 미완료 | baseline | highres 경량 변형 2, 전체 영상 학습 |
| 43 | m043_channel_adapter_v1 | 미완료 | baseline | channel_adapter 경량 변형 1, 전체 영상 학습 |
| 44 | m044_channel_adapter_v2 | 미완료 | baseline | channel_adapter 경량 변형 2, 전체 영상 학습 |
| 45 | m045_ce_half | 미완료 | baseline | 손실 탐색: ce_half |
| 46 | m046_dice_one | 미완료 | baseline | 손실 탐색: dice_one |
| 47 | m047_lovasz_quarter | 미완료 | baseline | 손실 탐색: lovasz_quarter |
| 48 | m048_focal_0p5 | 완료 | baseline | 손실 탐색: focal_0.5 |
| 49 | m049_focal_1p0 | 미완료 | baseline | 손실 탐색: focal_1.0 |
| 50 | m050_focal_2p0 | 미완료 | baseline | 손실 탐색: focal_2.0 |
| 51 | m051_tversky_0p3 | 미완료 | baseline | 손실 탐색: tversky_0.3 |
| 52 | m052_tversky_0p7 | 미완료 | baseline | 손실 탐색: tversky_0.7 |
| 53 | m053_generalized_dice | 미완료 | baseline | 손실 탐색: generalized_dice |
| 54 | m054_smooth_0p01 | 미완료 | baseline | 손실 탐색: smooth_0.01 |
| 55 | m055_smooth_0p03 | 미완료 | baseline | 손실 탐색: smooth_0.03 |
| 56 | m056_smooth_0p05 | 미완료 | baseline | 손실 탐색: smooth_0.05 |
| 57 | m057_hard_pixels_0p75 | 미완료 | baseline | 손실 탐색: hard_pixels_0.75 |
| 58 | m058_hard_pixels_0p5 | 미완료 | baseline | 손실 탐색: hard_pixels_0.5 |
| 59 | m059_hard_pixels_0p25 | 미완료 | baseline | 손실 탐색: hard_pixels_0.25 |
| 60 | m060_edge_ce_half | 미완료 | baseline | 손실 탐색: edge_ce_half |
| 61 | m061_edge_ce_one | 미완료 | baseline | 손실 탐색: edge_ce_one |
| 62 | m062_aux_01 | 완료 | baseline | 손실 탐색: aux_01 |
| 63 | m063_aux_02 | 미완료 | baseline | 손실 탐색: aux_02 |
| 64 | m064_boundary_005 | 완료 | baseline | 손실 탐색: boundary_005 |
| 65 | m065_wd_zero | 미완료 | baseline | 최적화 탐색: wd_zero |
| 66 | m066_wd_0001 | 미완료 | baseline | 최적화 탐색: wd_0001 |
| 67 | m067_wd_001 | 미완료 | baseline | 최적화 탐색: wd_001 |
| 68 | m068_wd_05 | 미완료 | baseline | 최적화 탐색: wd_05 |
| 69 | m069_beta2_99 | 미완료 | baseline | 최적화 탐색: beta2_99 |
| 70 | m070_beta1_95 | 미완료 | baseline | 최적화 탐색: beta1_95 |
| 71 | m071_sgd_001 | 완료 | baseline | 최적화 탐색: sgd_001 |
| 72 | m072_sgd_003 | 미완료 | baseline | 최적화 탐색: sgd_003 |
| 73 | m073_adam | 미완료 | baseline | 최적화 탐색: adam |
| 74 | m074_amsgrad | 미완료 | baseline | 최적화 탐색: amsgrad |
| 75 | m075_constant_lr | 미완료 | baseline | 최적화 탐색: constant_lr |
| 76 | m076_poly_lr | 미완료 | baseline | 최적화 탐색: poly_lr |
| 77 | m077_brightness | 미완료 | baseline | 데이터 탐색: brightness |
| 78 | m078_contrast | 미완료 | baseline | 데이터 탐색: contrast |
| 79 | m079_gamma | 미완료 | baseline | 데이터 탐색: gamma |
| 80 | m080_noise | 미완료 | baseline | 데이터 탐색: noise |
| 81 | m081_blur | 미완료 | baseline | 데이터 탐색: blur |
| 82 | m082_drop_zscore | 미완료 | baseline | 데이터 탐색: drop_zscore |
| 83 | m083_drop_cdf | 미완료 | baseline | 데이터 탐색: drop_cdf |
| 84 | m084_sharpen | 미완료 | baseline | 데이터 탐색: sharpen |
| 85 | m085_local_contrast | 미완료 | baseline | 데이터 탐색: local_contrast |
| 86 | m086_erase | 완료 | baseline | 데이터 탐색: erase |
| 87 | m087_sample_al3ni | 미완료 | baseline | 데이터 탐색: sample_al3ni |
| 88 | m088_sample_eutectic | 미완료 | baseline | 데이터 탐색: sample_eutectic |
| 89 | m089_scratch_residual_w16 | 미완료 | scratch | 처음부터 학습: residual U-Net, base width 16 |
| 90 | m090_scratch_residual_w24 | 미완료 | scratch | 처음부터 학습: residual U-Net, base width 24 |
| 91 | m091_scratch_residual_w32 | 완료 | scratch | 처음부터 학습: residual U-Net, base width 32 |
| 92 | m092_scratch_separable_w16 | 미완료 | scratch | 처음부터 학습: separable U-Net, base width 16 |
| 93 | m093_scratch_separable_w24 | 미완료 | scratch | 처음부터 학습: separable U-Net, base width 24 |
| 94 | m094_scratch_separable_w32 | 완료 | scratch | 처음부터 학습: separable U-Net, base width 32 |
| 95 | m095_scratch_convnext_w16 | 미완료 | scratch | 처음부터 학습: convnext U-Net, base width 16 |
| 96 | m096_scratch_convnext_w24 | 미완료 | scratch | 처음부터 학습: convnext U-Net, base width 24 |
| 97 | m097_scratch_convnext_w32 | 완료 | scratch | 처음부터 학습: convnext U-Net, base width 32 |
| 98 | m098_scratch_attention_w16 | 미완료 | scratch | 처음부터 학습: attention U-Net, base width 16 |
| 99 | m099_scratch_attention_w24 | 미완료 | scratch | 처음부터 학습: attention U-Net, base width 24 |
| 100 | m100_scratch_attention_w32 | 완료 | scratch | 처음부터 학습: attention U-Net, base width 32 |

## 재현 자료

- 실험 코드: `experiments/hundred_v1/`; 원본 모델·소스·결과 보존.
- 설정: `experiments/hundred_v1/configs/resolved/` 및 각 run의 `resolved_config.yaml`.
- 세부 결과: `experiments/hundred_v1/outputs/reports/results.csv` (plain/D4, 클래스 점수, 체크포인트 경로).
- 로그/체크포인트: `experiments/hundred_v1/outputs/runs/<실험>/`.
- 출처의 아이디어를 경량 변형한 자체 구현입니다. 공식 사전학습 백본이나 논문 전체 모델을 재현한 것은 아닙니다.

## 참고 문헌

- [aspp](https://arxiv.org/abs/1706.05587)
- [ppm](https://arxiv.org/abs/1612.01105)
- [se](https://arxiv.org/abs/1709.01507)
- [eca](https://arxiv.org/abs/1910.03151)
- [cbam](https://arxiv.org/abs/1807.06521)
- [attention](https://arxiv.org/abs/1804.03999)
- [convnext](https://arxiv.org/abs/2201.03545)
- [unet](https://arxiv.org/abs/1505.04597)
- [focal](https://arxiv.org/abs/1708.02002)
- [tversky](https://arxiv.org/abs/1706.05721)
- [gdice](https://arxiv.org/abs/1707.03237)
- [ohem](https://arxiv.org/abs/1604.03540)
- [torch](https://docs.pytorch.org/docs/stable/optim.html)
- [ce](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
