# 100개 실험 목록

추천 1개 + 기존 결과 기반 19개 + 새로운 탐색 80개. 상세 실행 설정은 `configs/resolved/`에 있습니다.

| 번호 | ID | 그룹 | 계열 | 초기화 | 추가 updates | 설명 | 근거 |
|---:|---|---|---|---|---:|---|---|
| 1 | `m001_recommended_fullframe` | recommended1 | local | baseline | 600 | 추천: 부모 CNN에서 패치 없이 전체 영상 600 updates | 현재 실험 관찰 |
| 2 | `m002_ff_short` | local19 | local | baseline | 300 | 전체 영상 300 updates, LR 1e-6 | 현재 실험 관찰 |
| 3 | `m003_ff_gentle_long` | local19 | local | baseline | 1000 | 전체 영상 1000 updates, LR 1e-6 | 현재 실험 관찰 |
| 4 | `m004_ff_lr5e6` | local19 | local | baseline | 600 | 전체 영상 600 updates, LR 5e-6 | 현재 실험 관찰 |
| 5 | `m005_ff_lr1e5` | local19 | local | baseline | 1000 | 전체 영상 1000 updates, LR 1e-5 | 현재 실험 관찰 |
| 6 | `m006_ema999` | local19 | local | baseline | 600 | 추천 일정 + EMA 0.999 | 현재 실험 관찰 |
| 7 | `m007_ema95` | local19 | local | baseline | 600 | 추천 일정 + EMA 0.95 | 현재 실험 관찰 |
| 8 | `m008_decoder_only` | local19 | local | baseline | 600 | encoder 고정, decoder와 classifier 미세조정 | 현재 실험 관찰 |
| 9 | `m009_head_only` | local19 | local | baseline | 600 | classifier만 미세조정 | 현재 실험 관찰 |
| 10 | `m010_encoder_slow` | local19 | local | baseline | 600 | encoder LR은 decoder의 0.1배 | 현재 실험 관찰 |
| 11 | `m011_seed2028_parent` | local19 | local | seed2028 | 600 | 과거 seed2028 최고점에서 전체 영상 미세조정 | 현재 실험 관찰 |
| 12 | `m012_photo_parent` | local19 | local | photo | 600 | 과거 photo_mild 모델에서 전체 영상 미세조정 | 현재 실험 관찰 |
| 13 | `m013_fullframe_parent` | local19 | local | fullframe | 600 | 과거 fullframe 모델에서 전체 영상 미세조정 | 현재 실험 관찰 |
| 14 | `m014_mild_photo` | local19 | local | baseline | 600 | 전체 영상에서 약한 광도 증강 확률 0.3 | 현재 실험 관찰 |
| 15 | `m015_aspp_w32_staged` | local19 | local | baseline | 900 | ASPP width 32: 새 블록 선행 학습 후 분리 LR | 현재 실험 관찰 |
| 16 | `m016_aspp_w64_staged` | local19 | local | baseline | 900 | ASPP width 64: 새 블록 선행 학습 후 분리 LR | 현재 실험 관찰 |
| 17 | `m017_light_boundary_ce` | local19 | local | baseline | 600 | 경계 픽셀 CE 가중치 +0.2 | 현재 실험 관찰 |
| 18 | `m018_light_class_weights` | local19 | local | baseline | 600 | 강한 가중치 대신 CE [1,1.05,1.1,1] | 현재 실험 관찰 |
| 19 | `m019_short_hard_patch` | local19 | local | baseline | 800 | 낮은 LR 768 hard patch 200 후 전체 영상 600 | 현재 실험 관찰 |
| 20 | `m020_aux_light` | local19 | local | baseline | 600 | 낮은 비중 0.05의 다중 해상도 보조 학습 | 현재 실험 관찰 |
| 21 | `m021_aspp_v1` | explore80 | architecture | baseline | 1200 | aspp 경량 변형 1, 전체 영상 학습 | [aspp](https://arxiv.org/abs/1706.05587) |
| 22 | `m022_aspp_v2` | explore80 | architecture | baseline | 1200 | aspp 경량 변형 2, 전체 영상 학습 | [aspp](https://arxiv.org/abs/1706.05587) |
| 23 | `m023_ppm_v1` | explore80 | architecture | baseline | 1200 | ppm 경량 변형 1, 전체 영상 학습 | [ppm](https://arxiv.org/abs/1612.01105) |
| 24 | `m024_ppm_v2` | explore80 | architecture | baseline | 1200 | ppm 경량 변형 2, 전체 영상 학습 | [ppm](https://arxiv.org/abs/1612.01105) |
| 25 | `m025_large_kernel_v1` | explore80 | architecture | baseline | 1200 | large_kernel 경량 변형 1, 전체 영상 학습 | [convnext](https://arxiv.org/abs/2201.03545) |
| 26 | `m026_large_kernel_v2` | explore80 | architecture | baseline | 1200 | large_kernel 경량 변형 2, 전체 영상 학습 | [convnext](https://arxiv.org/abs/2201.03545) |
| 27 | `m027_axial_v1` | explore80 | architecture | baseline | 1200 | axial 경량 변형 1, 전체 영상 학습 | 자체 탐색 가설 |
| 28 | `m028_axial_v2` | explore80 | architecture | baseline | 1200 | axial 경량 변형 2, 전체 영상 학습 | 자체 탐색 가설 |
| 29 | `m029_bottleneck_v1` | explore80 | architecture | baseline | 1200 | bottleneck 경량 변형 1, 전체 영상 학습 | 자체 탐색 가설 |
| 30 | `m030_bottleneck_v2` | explore80 | architecture | baseline | 1200 | bottleneck 경량 변형 2, 전체 영상 학습 | 자체 탐색 가설 |
| 31 | `m031_se_v1` | explore80 | architecture | baseline | 1200 | se 경량 변형 1, 전체 영상 학습 | [se](https://arxiv.org/abs/1709.01507) |
| 32 | `m032_se_v2` | explore80 | architecture | baseline | 1200 | se 경량 변형 2, 전체 영상 학습 | [se](https://arxiv.org/abs/1709.01507) |
| 33 | `m033_eca_v1` | explore80 | architecture | baseline | 1200 | eca 경량 변형 1, 전체 영상 학습 | [eca](https://arxiv.org/abs/1910.03151) |
| 34 | `m034_eca_v2` | explore80 | architecture | baseline | 1200 | eca 경량 변형 2, 전체 영상 학습 | [eca](https://arxiv.org/abs/1910.03151) |
| 35 | `m035_cbam_v1` | explore80 | architecture | baseline | 1200 | cbam 경량 변형 1, 전체 영상 학습 | [cbam](https://arxiv.org/abs/1807.06521) |
| 36 | `m036_cbam_v2` | explore80 | architecture | baseline | 1200 | cbam 경량 변형 2, 전체 영상 학습 | [cbam](https://arxiv.org/abs/1807.06521) |
| 37 | `m037_skip_gate_v1` | explore80 | architecture | baseline | 1200 | skip_gate 경량 변형 1, 전체 영상 학습 | [attention](https://arxiv.org/abs/1804.03999) |
| 38 | `m038_skip_gate_v2` | explore80 | architecture | baseline | 1200 | skip_gate 경량 변형 2, 전체 영상 학습 | [attention](https://arxiv.org/abs/1804.03999) |
| 39 | `m039_decoder_refine_v1` | explore80 | architecture | baseline | 1200 | decoder_refine 경량 변형 1, 전체 영상 학습 | 자체 탐색 가설 |
| 40 | `m040_decoder_refine_v2` | explore80 | architecture | baseline | 1200 | decoder_refine 경량 변형 2, 전체 영상 학습 | 자체 탐색 가설 |
| 41 | `m041_highres_v1` | explore80 | architecture | baseline | 1200 | highres 경량 변형 1, 전체 영상 학습 | 자체 탐색 가설 |
| 42 | `m042_highres_v2` | explore80 | architecture | baseline | 1200 | highres 경량 변형 2, 전체 영상 학습 | 자체 탐색 가설 |
| 43 | `m043_channel_adapter_v1` | explore80 | architecture | baseline | 1200 | channel_adapter 경량 변형 1, 전체 영상 학습 | 자체 탐색 가설 |
| 44 | `m044_channel_adapter_v2` | explore80 | architecture | baseline | 1200 | channel_adapter 경량 변형 2, 전체 영상 학습 | 자체 탐색 가설 |
| 45 | `m045_ce_half` | explore80 | loss | baseline | 1200 | 손실 탐색: ce_half | 자체 탐색 가설 |
| 46 | `m046_dice_one` | explore80 | loss | baseline | 1200 | 손실 탐색: dice_one | 자체 탐색 가설 |
| 47 | `m047_lovasz_quarter` | explore80 | loss | baseline | 1200 | 손실 탐색: lovasz_quarter | 자체 탐색 가설 |
| 48 | `m048_focal_0p5` | explore80 | loss | baseline | 1200 | 손실 탐색: focal_0.5 | [focal](https://arxiv.org/abs/1708.02002) |
| 49 | `m049_focal_1p0` | explore80 | loss | baseline | 1200 | 손실 탐색: focal_1.0 | [focal](https://arxiv.org/abs/1708.02002) |
| 50 | `m050_focal_2p0` | explore80 | loss | baseline | 1200 | 손실 탐색: focal_2.0 | [focal](https://arxiv.org/abs/1708.02002) |
| 51 | `m051_tversky_0p3` | explore80 | loss | baseline | 1200 | 손실 탐색: tversky_0.3 | [tversky](https://arxiv.org/abs/1706.05721) |
| 52 | `m052_tversky_0p7` | explore80 | loss | baseline | 1200 | 손실 탐색: tversky_0.7 | [tversky](https://arxiv.org/abs/1706.05721) |
| 53 | `m053_generalized_dice` | explore80 | loss | baseline | 1200 | 손실 탐색: generalized_dice | [gdice](https://arxiv.org/abs/1707.03237) |
| 54 | `m054_smooth_0p01` | explore80 | loss | baseline | 1200 | 손실 탐색: smooth_0.01 | [ce](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) |
| 55 | `m055_smooth_0p03` | explore80 | loss | baseline | 1200 | 손실 탐색: smooth_0.03 | [ce](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) |
| 56 | `m056_smooth_0p05` | explore80 | loss | baseline | 1200 | 손실 탐색: smooth_0.05 | [ce](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) |
| 57 | `m057_hard_pixels_0p75` | explore80 | loss | baseline | 1200 | 손실 탐색: hard_pixels_0.75 | [ohem](https://arxiv.org/abs/1604.03540) |
| 58 | `m058_hard_pixels_0p5` | explore80 | loss | baseline | 1200 | 손실 탐색: hard_pixels_0.5 | [ohem](https://arxiv.org/abs/1604.03540) |
| 59 | `m059_hard_pixels_0p25` | explore80 | loss | baseline | 1200 | 손실 탐색: hard_pixels_0.25 | [ohem](https://arxiv.org/abs/1604.03540) |
| 60 | `m060_edge_ce_half` | explore80 | loss | baseline | 1200 | 손실 탐색: edge_ce_half | 자체 탐색 가설 |
| 61 | `m061_edge_ce_one` | explore80 | loss | baseline | 1200 | 손실 탐색: edge_ce_one | 자체 탐색 가설 |
| 62 | `m062_aux_01` | explore80 | loss | baseline | 1200 | 손실 탐색: aux_01 | 자체 탐색 가설 |
| 63 | `m063_aux_02` | explore80 | loss | baseline | 1200 | 손실 탐색: aux_02 | 자체 탐색 가설 |
| 64 | `m064_boundary_005` | explore80 | loss | baseline | 1200 | 손실 탐색: boundary_005 | 자체 탐색 가설 |
| 65 | `m065_wd_zero` | explore80 | optimizer | baseline | 1200 | 최적화 탐색: wd_zero | [torch](https://docs.pytorch.org/docs/stable/optim.html) |
| 66 | `m066_wd_0001` | explore80 | optimizer | baseline | 1200 | 최적화 탐색: wd_0001 | [torch](https://docs.pytorch.org/docs/stable/optim.html) |
| 67 | `m067_wd_001` | explore80 | optimizer | baseline | 1200 | 최적화 탐색: wd_001 | [torch](https://docs.pytorch.org/docs/stable/optim.html) |
| 68 | `m068_wd_05` | explore80 | optimizer | baseline | 1200 | 최적화 탐색: wd_05 | [torch](https://docs.pytorch.org/docs/stable/optim.html) |
| 69 | `m069_beta2_99` | explore80 | optimizer | baseline | 1200 | 최적화 탐색: beta2_99 | [torch](https://docs.pytorch.org/docs/stable/optim.html) |
| 70 | `m070_beta1_95` | explore80 | optimizer | baseline | 1200 | 최적화 탐색: beta1_95 | [torch](https://docs.pytorch.org/docs/stable/optim.html) |
| 71 | `m071_sgd_001` | explore80 | optimizer | baseline | 1200 | 최적화 탐색: sgd_001 | [torch](https://docs.pytorch.org/docs/stable/optim.html) |
| 72 | `m072_sgd_003` | explore80 | optimizer | baseline | 1200 | 최적화 탐색: sgd_003 | [torch](https://docs.pytorch.org/docs/stable/optim.html) |
| 73 | `m073_adam` | explore80 | optimizer | baseline | 1200 | 최적화 탐색: adam | [torch](https://docs.pytorch.org/docs/stable/optim.html) |
| 74 | `m074_amsgrad` | explore80 | optimizer | baseline | 1200 | 최적화 탐색: amsgrad | [torch](https://docs.pytorch.org/docs/stable/optim.html) |
| 75 | `m075_constant_lr` | explore80 | optimizer | baseline | 1200 | 최적화 탐색: constant_lr | [torch](https://docs.pytorch.org/docs/stable/optim.html) |
| 76 | `m076_poly_lr` | explore80 | optimizer | baseline | 1200 | 최적화 탐색: poly_lr | [torch](https://docs.pytorch.org/docs/stable/optim.html) |
| 77 | `m077_brightness` | explore80 | data | baseline | 1200 | 데이터 탐색: brightness | 자체 탐색 가설 |
| 78 | `m078_contrast` | explore80 | data | baseline | 1200 | 데이터 탐색: contrast | 자체 탐색 가설 |
| 79 | `m079_gamma` | explore80 | data | baseline | 1200 | 데이터 탐색: gamma | 자체 탐색 가설 |
| 80 | `m080_noise` | explore80 | data | baseline | 1200 | 데이터 탐색: noise | 자체 탐색 가설 |
| 81 | `m081_blur` | explore80 | data | baseline | 1200 | 데이터 탐색: blur | 자체 탐색 가설 |
| 82 | `m082_drop_zscore` | explore80 | data | baseline | 1200 | 데이터 탐색: drop_zscore | 자체 탐색 가설 |
| 83 | `m083_drop_cdf` | explore80 | data | baseline | 1200 | 데이터 탐색: drop_cdf | 자체 탐색 가설 |
| 84 | `m084_sharpen` | explore80 | data | baseline | 1200 | 데이터 탐색: sharpen | 자체 탐색 가설 |
| 85 | `m085_local_contrast` | explore80 | data | baseline | 1200 | 데이터 탐색: local_contrast | 자체 탐색 가설 |
| 86 | `m086_erase` | explore80 | data | baseline | 1200 | 데이터 탐색: erase | 자체 탐색 가설 |
| 87 | `m087_sample_al3ni` | explore80 | data | baseline | 1200 | 데이터 탐색: sample_al3ni | 자체 탐색 가설 |
| 88 | `m088_sample_eutectic` | explore80 | data | baseline | 1200 | 데이터 탐색: sample_eutectic | 자체 탐색 가설 |
| 89 | `m089_scratch_residual_w16` | explore80 | scratch | scratch | 4500 | 처음부터 학습: residual U-Net, base width 16 | [unet](https://arxiv.org/abs/1505.04597) |
| 90 | `m090_scratch_residual_w24` | explore80 | scratch | scratch | 4500 | 처음부터 학습: residual U-Net, base width 24 | [unet](https://arxiv.org/abs/1505.04597) |
| 91 | `m091_scratch_residual_w32` | explore80 | scratch | scratch | 4500 | 처음부터 학습: residual U-Net, base width 32 | [unet](https://arxiv.org/abs/1505.04597) |
| 92 | `m092_scratch_separable_w16` | explore80 | scratch | scratch | 4500 | 처음부터 학습: separable U-Net, base width 16 | [unet](https://arxiv.org/abs/1505.04597) |
| 93 | `m093_scratch_separable_w24` | explore80 | scratch | scratch | 4500 | 처음부터 학습: separable U-Net, base width 24 | [unet](https://arxiv.org/abs/1505.04597) |
| 94 | `m094_scratch_separable_w32` | explore80 | scratch | scratch | 4500 | 처음부터 학습: separable U-Net, base width 32 | [unet](https://arxiv.org/abs/1505.04597) |
| 95 | `m095_scratch_convnext_w16` | explore80 | scratch | scratch | 4500 | 처음부터 학습: convnext U-Net, base width 16 | [convnext](https://arxiv.org/abs/2201.03545) |
| 96 | `m096_scratch_convnext_w24` | explore80 | scratch | scratch | 4500 | 처음부터 학습: convnext U-Net, base width 24 | [convnext](https://arxiv.org/abs/2201.03545) |
| 97 | `m097_scratch_convnext_w32` | explore80 | scratch | scratch | 4500 | 처음부터 학습: convnext U-Net, base width 32 | [convnext](https://arxiv.org/abs/2201.03545) |
| 98 | `m098_scratch_attention_w16` | explore80 | scratch | scratch | 4500 | 처음부터 학습: attention U-Net, base width 16 | [unet](https://arxiv.org/abs/1505.04597) |
| 99 | `m099_scratch_attention_w24` | explore80 | scratch | scratch | 4500 | 처음부터 학습: attention U-Net, base width 24 | [unet](https://arxiv.org/abs/1505.04597) |
| 100 | `m100_scratch_attention_w32` | explore80 | scratch | scratch | 4500 | 처음부터 학습: attention U-Net, base width 32 | [unet](https://arxiv.org/abs/1505.04597) |
