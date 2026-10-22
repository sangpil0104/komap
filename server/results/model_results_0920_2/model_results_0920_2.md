# model_results_0920 — 100개 모델 실험 결과

상태: 진행 중 / 일부 완료: 99/100

주 지표는 공식 검증 20장의 이미지별 평균 mIoU입니다. Global mIoU는 별도 보조 지표입니다. 값은 %, 변화는 pp입니다.

구성: 추천 1개 + 기존/동료 결과 기반 19개 + 별도 탐색 80개. 100개의 서로 다른 실험 설정이며 모두 서로 다른 원 논문 모델 100종이라는 뜻은 아닙니다.

각 행은 추가 학습 중 최고 plain 검증 점수로 선택한 best_trained.pt를 plain/D4로 평가한 결과입니다. best.pt는 학습 전 update 0까지 포함한 최고 plain 체크포인트입니다. 부모 가중치가 선택된 행은 별도로 표시합니다.

기존 기준 CNN+D4: 이미지 평균 78.6865, Global 81.4989.
이전 seed2028+D4의 관측 최고: 이미지 평균 78.7858, Global 81.5082. 빈 클래스 처리와 A71 민감도를 함께 확인해야 합니다.

이번 관측 최고 후보: **m011_seed2028_parent** — 이미지 평균 **78.8004**, Global **81.5255**, 기준 대비 **+0.1140 pp**.
이전 이미지 평균 최고 78.7858 대비 +0.0147 pp. 관측 검증 최고이며 독립 테스트 성능을 뜻하지 않습니다.

## D4 순위

| 순위 | 실험 | 그룹 | 이미지 mIoU | Global mIoU | 기준 Δpp | 공통19장 Δpp | 개선 장수 | 초기 가중치 유지 | 학습 분 |
|---:|---|---|---:|---:|---:|---:|---:|---|---:|
| 1 | m011_seed2028_parent | local19 | 78.8004 | 81.5255 | +0.1140 | -0.5347 | 8/20 | 예 | 2.41 |
| 2 | m013_fullframe_parent | local19 | 78.7992 | 81.3118 | +0.1127 | -0.4737 | 7/20 | 아니오 | 2.45 |
| 3 | m063_aux_02 | explore80 | 78.6928 | 81.4473 | +0.0063 | -0.0386 | 8/20 | 예 | 4.79 |
| 4 | m085_local_contrast | explore80 | 78.6898 | 81.5626 | +0.0033 | -0.0155 | 8/20 | 아니오 | 4.88 |
| 5 | m009_head_only | local19 | 78.6888 | 81.4992 | +0.0023 | +0.0019 | 13/20 | 아니오 | 1.29 |
| 6 | m012_photo_parent | local19 | 78.6882 | 81.4169 | +0.0017 | +0.0218 | 7/20 | 아니오 | 2.47 |
| 7 | m008_decoder_only | local19 | 78.6859 | 81.4786 | -0.0006 | +0.0034 | 11/20 | 예 | 1.99 |
| 8 | m006_ema999 | local19 | 78.6804 | 81.4877 | -0.0061 | -0.0052 | 7/20 | 예 | 2.45 |
| 9 | m062_aux_01 | explore80 | 78.6800 | 81.4562 | -0.0065 | -0.0294 | 10/20 | 예 | 4.81 |
| 10 | m016_aspp_w64_staged | local19 | 78.6770 | 81.4797 | -0.0094 | -0.0095 | 6/20 | 예 | 3.16 |
| 11 | m010_encoder_slow | local19 | 78.6760 | 81.4661 | -0.0105 | -0.0062 | 7/20 | 예 | 2.43 |
| 12 | m015_aspp_w32_staged | local19 | 78.6759 | 81.4801 | -0.0106 | -0.0111 | 8/20 | 예 | 3.16 |
| 13 | m059_hard_pixels_0p25 | explore80 | 78.6758 | 81.4599 | -0.0107 | -0.0051 | 9/20 | 예 | 4.87 |
| 14 | m054_smooth_0p01 | explore80 | 78.6677 | 81.4289 | -0.0188 | -0.0079 | 8/20 | 예 | 4.91 |
| 15 | m078_contrast | explore80 | 78.6673 | 81.4644 | -0.0192 | -0.0054 | 7/20 | 예 | 4.80 |
| 16 | m073_adam | explore80 | 78.6666 | 81.4562 | -0.0198 | -0.0130 | 8/20 | 예 | 4.77 |
| 17 | m064_boundary_005 | explore80 | 78.6661 | 81.4894 | -0.0204 | -0.0188 | 10/20 | 예 | 4.84 |
| 18 | m020_aux_light | local19 | 78.6657 | 81.4582 | -0.0207 | -0.0250 | 10/20 | 예 | 2.46 |
| 19 | m050_focal_2p0 | explore80 | 78.6642 | 81.3948 | -0.0223 | -0.0156 | 10/20 | 예 | 4.89 |
| 20 | m055_smooth_0p03 | explore80 | 78.6634 | 81.3229 | -0.0231 | -0.0358 | 10/20 | 예 | 4.85 |
| 21 | m049_focal_1p0 | explore80 | 78.6613 | 81.4397 | -0.0252 | -0.0209 | 9/20 | 예 | 5.05 |
| 22 | m087_sample_al3ni | explore80 | 78.6606 | 81.4662 | -0.0259 | -0.0143 | 9/20 | 예 | 4.78 |
| 23 | m005_ff_lr1e5 | local19 | 78.6594 | 81.4669 | -0.0271 | -0.0159 | 7/20 | 예 | 3.98 |
| 24 | m014_mild_photo | local19 | 78.6588 | 81.4658 | -0.0277 | -0.0143 | 6/20 | 예 | 2.46 |
| 25 | m017_light_boundary_ce | local19 | 78.6586 | 81.4407 | -0.0279 | -0.0120 | 9/20 | 예 | 2.49 |
| 26 | m002_ff_short | local19 | 78.6565 | 81.4497 | -0.0300 | -0.0264 | 8/20 | 예 | 1.27 |
| 27 | m088_sample_eutectic | explore80 | 78.6554 | 81.4503 | -0.0311 | -0.0185 | 10/20 | 예 | 4.75 |
| 28 | m048_focal_0p5 | explore80 | 78.6543 | 81.4270 | -0.0322 | -0.0199 | 7/20 | 예 | 4.92 |
| 29 | m058_hard_pixels_0p5 | explore80 | 78.6530 | 81.4493 | -0.0335 | -0.0168 | 6/20 | 예 | 4.88 |
| 30 | m076_poly_lr | explore80 | 78.6519 | 81.4604 | -0.0346 | -0.0234 | 7/20 | 예 | 4.78 |
| 31 | m033_eca_v1 | explore80 | 78.6515 | 81.4478 | -0.0350 | -0.0202 | 8/20 | 예 | 4.82 |
| 32 | m057_hard_pixels_0p75 | explore80 | 78.6508 | 81.4435 | -0.0356 | -0.0197 | 6/20 | 예 | 4.92 |
| 33 | m022_aspp_v2 | explore80 | 78.6506 | 81.4497 | -0.0358 | -0.0211 | 7/20 | 예 | 4.83 |
| 34 | m079_gamma | explore80 | 78.6506 | 81.4386 | -0.0359 | -0.0224 | 7/20 | 예 | 4.83 |
| 35 | m074_amsgrad | explore80 | 78.6503 | 81.4468 | -0.0362 | -0.0214 | 8/20 | 예 | 4.81 |
| 36 | m026_large_kernel_v2 | explore80 | 78.6499 | 81.4488 | -0.0366 | -0.0218 | 8/20 | 예 | 4.81 |
| 37 | m067_wd_001 | explore80 | 78.6497 | 81.4466 | -0.0368 | -0.0220 | 8/20 | 예 | 4.82 |
| 38 | m070_beta1_95 | explore80 | 78.6490 | 81.4481 | -0.0375 | -0.0223 | 8/20 | 예 | 4.80 |
| 39 | m007_ema95 | local19 | 78.6478 | 81.4514 | -0.0387 | -0.0235 | 8/20 | 예 | 2.43 |
| 40 | m075_constant_lr | explore80 | 78.6477 | 81.4488 | -0.0388 | -0.0244 | 8/20 | 예 | 4.83 |
| 41 | m052_tversky_0p7 | explore80 | 78.6473 | 81.4986 | -0.0392 | -0.0258 | 9/20 | 예 | 4.91 |
| 42 | m001_recommended_fullframe | recommended1 | 78.6462 | 81.4554 | -0.0403 | -0.0263 | 5/20 | 예 | 2.66 |
| 43 | m071_sgd_001 | explore80 | 78.6453 | 81.4429 | -0.0412 | -0.0298 | 9/20 | 예 | 4.96 |
| 44 | m018_light_class_weights | local19 | 78.6440 | 81.4284 | -0.0424 | -0.0224 | 8/20 | 예 | 2.51 |
| 45 | m024_ppm_v2 | explore80 | 78.6432 | 81.4532 | -0.0432 | -0.0314 | 6/20 | 예 | 4.90 |
| 46 | m047_lovasz_quarter | explore80 | 78.6423 | 81.4437 | -0.0441 | -0.0285 | 6/20 | 예 | 4.80 |
| 47 | m077_brightness | explore80 | 78.6419 | 81.4408 | -0.0446 | -0.0291 | 6/20 | 예 | 4.78 |
| 48 | m086_erase | explore80 | 78.6417 | 81.4776 | -0.0448 | -0.0506 | 10/20 | 예 | 4.76 |
| 49 | m036_cbam_v2 | explore80 | 78.6415 | 81.4378 | -0.0450 | -0.0315 | 7/20 | 예 | 4.86 |
| 50 | m025_large_kernel_v1 | explore80 | 78.6397 | 81.4477 | -0.0468 | -0.0350 | 7/20 | 예 | 4.82 |
| 51 | m040_decoder_refine_v2 | explore80 | 78.6388 | 81.4331 | -0.0477 | -0.0314 | 5/20 | 예 | 5.43 |
| 52 | m069_beta2_99 | explore80 | 78.6380 | 81.4351 | -0.0485 | -0.0328 | 5/20 | 예 | 4.80 |
| 53 | m027_axial_v1 | explore80 | 78.6378 | 81.4344 | -0.0487 | -0.0333 | 5/20 | 예 | 4.81 |
| 54 | m041_highres_v1 | explore80 | 78.6378 | 81.4349 | -0.0487 | -0.0331 | 5/20 | 예 | 5.05 |
| 55 | m043_channel_adapter_v1 | explore80 | 78.6375 | 81.4344 | -0.0490 | -0.0334 | 5/20 | 예 | 4.81 |
| 56 | m080_noise | explore80 | 78.6372 | 81.4392 | -0.0492 | -0.0335 | 6/20 | 예 | 4.79 |
| 57 | m032_se_v2 | explore80 | 78.6372 | 81.4346 | -0.0493 | -0.0338 | 5/20 | 예 | 4.80 |
| 58 | m042_highres_v2 | explore80 | 78.6371 | 81.4342 | -0.0493 | -0.0337 | 5/20 | 예 | 4.81 |
| 59 | m066_wd_0001 | explore80 | 78.6369 | 81.4345 | -0.0496 | -0.0341 | 5/20 | 예 | 4.78 |
| 60 | m034_eca_v2 | explore80 | 78.6367 | 81.4339 | -0.0498 | -0.0341 | 5/20 | 예 | 4.80 |
| 61 | m028_axial_v2 | explore80 | 78.6366 | 81.4340 | -0.0499 | -0.0345 | 5/20 | 예 | 4.82 |
| 62 | m065_wd_zero | explore80 | 78.6365 | 81.4341 | -0.0499 | -0.0343 | 5/20 | 예 | 4.92 |
| 63 | m060_edge_ce_half | explore80 | 78.6362 | 81.4117 | -0.0503 | -0.0326 | 8/20 | 예 | 4.91 |
| 64 | m031_se_v1 | explore80 | 78.6362 | 81.4341 | -0.0503 | -0.0351 | 5/20 | 예 | 4.81 |
| 65 | m039_decoder_refine_v1 | explore80 | 78.6361 | 81.4335 | -0.0504 | -0.0344 | 5/20 | 예 | 5.13 |
| 66 | m037_skip_gate_v1 | explore80 | 78.6358 | 81.4340 | -0.0506 | -0.0350 | 5/20 | 예 | 4.99 |
| 67 | m023_ppm_v1 | explore80 | 78.6358 | 81.4466 | -0.0507 | -0.0393 | 6/20 | 예 | 4.84 |
| 68 | m021_aspp_v1 | explore80 | 78.6355 | 81.4331 | -0.0510 | -0.0358 | 5/20 | 예 | 4.95 |
| 69 | m038_skip_gate_v2 | explore80 | 78.6353 | 81.4332 | -0.0511 | -0.0358 | 5/20 | 예 | 5.00 |
| 70 | m068_wd_05 | explore80 | 78.6343 | 81.4306 | -0.0522 | -0.0369 | 5/20 | 예 | 4.79 |
| 71 | m035_cbam_v1 | explore80 | 78.6340 | 81.4311 | -0.0524 | -0.0375 | 5/20 | 예 | 4.81 |
| 72 | m072_sgd_003 | explore80 | 78.6317 | 81.4475 | -0.0548 | -0.0364 | 6/20 | 예 | 4.72 |
| 73 | m044_channel_adapter_v2 | explore80 | 78.6314 | 81.4426 | -0.0551 | -0.0442 | 6/20 | 예 | 4.83 |
| 74 | m061_edge_ce_one | explore80 | 78.6283 | 81.3903 | -0.0582 | -0.0391 | 9/20 | 예 | 4.88 |
| 75 | m019_short_hard_patch | local19 | 78.6270 | 81.4655 | -0.0595 | -0.0611 | 13/20 | 예 | 2.89 |
| 76 | m056_smooth_0p05 | explore80 | 78.6264 | 81.2262 | -0.0600 | -0.0801 | 10/20 | 예 | 5.05 |
| 77 | m046_dice_one | explore80 | 78.6264 | 81.4339 | -0.0601 | -0.0497 | 6/20 | 예 | 4.82 |
| 78 | m081_blur | explore80 | 78.6263 | 81.4144 | -0.0602 | -0.0335 | 7/20 | 예 | 4.80 |
| 79 | m051_tversky_0p3 | explore80 | 78.6254 | 81.3788 | -0.0611 | -0.0518 | 7/20 | 예 | 4.93 |
| 80 | m045_ce_half | explore80 | 78.6246 | 81.4214 | -0.0619 | -0.0510 | 7/20 | 예 | 4.81 |
| 81 | m004_ff_lr5e6 | local19 | 78.6246 | 81.4344 | -0.0619 | -0.0468 | 6/20 | 예 | 2.46 |
| 82 | m029_bottleneck_v1 | explore80 | 78.6231 | 81.4371 | -0.0634 | -0.0521 | 6/20 | 예 | 4.83 |
| 83 | m030_bottleneck_v2 | explore80 | 78.6200 | 81.4339 | -0.0664 | -0.0564 | 5/20 | 예 | 4.84 |
| 84 | m084_sharpen | explore80 | 78.6121 | 81.4381 | -0.0744 | -0.0749 | 6/20 | 예 | 4.79 |
| 85 | m053_generalized_dice | explore80 | 78.6102 | 81.4695 | -0.0763 | -0.0652 | 8/20 | 예 | 4.87 |
| 86 | m083_drop_cdf | explore80 | 78.5799 | 81.5521 | -0.1066 | -0.1334 | 9/20 | 예 | 4.77 |
| 87 | m082_drop_zscore | explore80 | 78.5327 | 81.4144 | -0.1538 | -0.0964 | 4/20 | 예 | 4.78 |
| 88 | m100_scratch_attention_w32 | explore80 | 77.7360 | 81.0911 | -0.9504 | -0.9424 | 6/20 | 아니오 | 10.49 |
| 89 | m090_scratch_residual_w24 | explore80 | 77.4791 | 80.6193 | -1.2074 | -1.2029 | 4/20 | 아니오 | 7.29 |
| 90 | m099_scratch_attention_w24 | explore80 | 77.1538 | 80.4992 | -1.5327 | -1.5288 | 2/20 | 아니오 | 8.04 |
| 91 | m091_scratch_residual_w32 | explore80 | 76.6582 | 79.7897 | -2.0282 | -2.0415 | 0/20 | 아니오 | 9.63 |
| 92 | m098_scratch_attention_w16 | explore80 | 76.1958 | 79.6541 | -2.4907 | -2.5459 | 0/20 | 아니오 | 7.08 |
| 93 | m094_scratch_separable_w32 | explore80 | 76.0858 | 79.2897 | -2.6006 | -2.5809 | 0/20 | 아니오 | 11.66 |
| 94 | m089_scratch_residual_w16 | explore80 | 75.7622 | 79.1405 | -2.9243 | -2.9316 | 0/20 | 아니오 | 6.35 |
| 95 | m093_scratch_separable_w24 | explore80 | 75.6901 | 79.2564 | -2.9964 | -2.9853 | 0/20 | 아니오 | 7.47 |
| 96 | m092_scratch_separable_w16 | explore80 | 74.3293 | 77.9233 | -4.3572 | -4.3556 | 0/20 | 아니오 | 7.10 |
| 97 | m097_scratch_convnext_w32 | explore80 | 73.9620 | 77.7200 | -4.7245 | -4.7357 | 0/20 | 아니오 | 16.20 |
| 98 | m096_scratch_convnext_w24 | explore80 | 72.4240 | 76.1330 | -6.2625 | -6.2587 | 0/20 | 아니오 | 8.55 |
| 99 | m095_scratch_convnext_w16 | explore80 | 70.9950 | 74.8773 | -7.6915 | -7.8554 | 0/20 | 아니오 | 7.57 |

## 상위 후보의 클래스별 IoU

| 실험 | 초정 Si | Al3Ni | 공정 Si | Al |
|---|---:|---:|---:|---:|
| m011_seed2028_parent | 91.3248 | 72.7311 | 59.1479 | 93.9070 |
| m013_fullframe_parent | 90.8132 | 73.2013 | 59.1071 | 94.0176 |
| m063_aux_02 | 86.3737 | 74.5711 | 59.6790 | 94.1475 |
| m085_local_contrast | 86.3100 | 74.7333 | 59.6097 | 94.1061 |
| m009_head_only | 86.3963 | 74.4108 | 59.8490 | 94.0990 |
| m012_photo_parent | 86.5151 | 74.3587 | 59.8300 | 94.0490 |
| m008_decoder_only | 86.3961 | 74.3865 | 59.8622 | 94.0989 |
| m006_ema999 | 86.3854 | 74.3946 | 59.8416 | 94.1000 |
| m062_aux_01 | 86.3478 | 74.5103 | 59.7101 | 94.1515 |
| m016_aspp_w64_staged | 86.3754 | 74.3903 | 59.8419 | 94.1006 |

## 평가 해석

- 기준 모델보다 높은 D4 후보: 6/99개.
- 초기 가중치를 plain 기준으로 넘지 못한 후보: 83/99개.
- 완료 모델의 학습 시간 합: 485.28분. 프로세스 시작·최종 평가 시간 등은 별도입니다.
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
| 2 | m002_ff_short | 완료 | baseline | 전체 영상 300 updates, LR 1e-6 |
| 3 | m003_ff_gentle_long | 실패 | baseline | 전체 영상 1000 updates, LR 1e-6 |
| 4 | m004_ff_lr5e6 | 완료 | baseline | 전체 영상 600 updates, LR 5e-6 |
| 5 | m005_ff_lr1e5 | 완료 | baseline | 전체 영상 1000 updates, LR 1e-5 |
| 6 | m006_ema999 | 완료 | baseline | 추천 일정 + EMA 0.999 |
| 7 | m007_ema95 | 완료 | baseline | 추천 일정 + EMA 0.95 |
| 8 | m008_decoder_only | 완료 | baseline | encoder 고정, decoder와 classifier 미세조정 |
| 9 | m009_head_only | 완료 | baseline | classifier만 미세조정 |
| 10 | m010_encoder_slow | 완료 | baseline | encoder LR은 decoder의 0.1배 |
| 11 | m011_seed2028_parent | 완료 | seed2028 | 과거 seed2028 최고점에서 전체 영상 미세조정 |
| 12 | m012_photo_parent | 완료 | photo | 과거 photo_mild 모델에서 전체 영상 미세조정 |
| 13 | m013_fullframe_parent | 완료 | fullframe | 과거 fullframe 모델에서 전체 영상 미세조정 |
| 14 | m014_mild_photo | 완료 | baseline | 전체 영상에서 약한 광도 증강 확률 0.3 |
| 15 | m015_aspp_w32_staged | 완료 | baseline | ASPP width 32: 새 블록 선행 학습 후 분리 LR |
| 16 | m016_aspp_w64_staged | 완료 | baseline | ASPP width 64: 새 블록 선행 학습 후 분리 LR |
| 17 | m017_light_boundary_ce | 완료 | baseline | 경계 픽셀 CE 가중치 +0.2 |
| 18 | m018_light_class_weights | 완료 | baseline | 강한 가중치 대신 CE [1,1.05,1.1,1] |
| 19 | m019_short_hard_patch | 완료 | baseline | 낮은 LR 768 hard patch 200 후 전체 영상 600 |
| 20 | m020_aux_light | 완료 | baseline | 낮은 비중 0.05의 다중 해상도 보조 학습 |
| 21 | m021_aspp_v1 | 완료 | baseline | aspp 경량 변형 1, 전체 영상 학습 |
| 22 | m022_aspp_v2 | 완료 | baseline | aspp 경량 변형 2, 전체 영상 학습 |
| 23 | m023_ppm_v1 | 완료 | baseline | ppm 경량 변형 1, 전체 영상 학습 |
| 24 | m024_ppm_v2 | 완료 | baseline | ppm 경량 변형 2, 전체 영상 학습 |
| 25 | m025_large_kernel_v1 | 완료 | baseline | large_kernel 경량 변형 1, 전체 영상 학습 |
| 26 | m026_large_kernel_v2 | 완료 | baseline | large_kernel 경량 변형 2, 전체 영상 학습 |
| 27 | m027_axial_v1 | 완료 | baseline | axial 경량 변형 1, 전체 영상 학습 |
| 28 | m028_axial_v2 | 완료 | baseline | axial 경량 변형 2, 전체 영상 학습 |
| 29 | m029_bottleneck_v1 | 완료 | baseline | bottleneck 경량 변형 1, 전체 영상 학습 |
| 30 | m030_bottleneck_v2 | 완료 | baseline | bottleneck 경량 변형 2, 전체 영상 학습 |
| 31 | m031_se_v1 | 완료 | baseline | se 경량 변형 1, 전체 영상 학습 |
| 32 | m032_se_v2 | 완료 | baseline | se 경량 변형 2, 전체 영상 학습 |
| 33 | m033_eca_v1 | 완료 | baseline | eca 경량 변형 1, 전체 영상 학습 |
| 34 | m034_eca_v2 | 완료 | baseline | eca 경량 변형 2, 전체 영상 학습 |
| 35 | m035_cbam_v1 | 완료 | baseline | cbam 경량 변형 1, 전체 영상 학습 |
| 36 | m036_cbam_v2 | 완료 | baseline | cbam 경량 변형 2, 전체 영상 학습 |
| 37 | m037_skip_gate_v1 | 완료 | baseline | skip_gate 경량 변형 1, 전체 영상 학습 |
| 38 | m038_skip_gate_v2 | 완료 | baseline | skip_gate 경량 변형 2, 전체 영상 학습 |
| 39 | m039_decoder_refine_v1 | 완료 | baseline | decoder_refine 경량 변형 1, 전체 영상 학습 |
| 40 | m040_decoder_refine_v2 | 완료 | baseline | decoder_refine 경량 변형 2, 전체 영상 학습 |
| 41 | m041_highres_v1 | 완료 | baseline | highres 경량 변형 1, 전체 영상 학습 |
| 42 | m042_highres_v2 | 완료 | baseline | highres 경량 변형 2, 전체 영상 학습 |
| 43 | m043_channel_adapter_v1 | 완료 | baseline | channel_adapter 경량 변형 1, 전체 영상 학습 |
| 44 | m044_channel_adapter_v2 | 완료 | baseline | channel_adapter 경량 변형 2, 전체 영상 학습 |
| 45 | m045_ce_half | 완료 | baseline | 손실 탐색: ce_half |
| 46 | m046_dice_one | 완료 | baseline | 손실 탐색: dice_one |
| 47 | m047_lovasz_quarter | 완료 | baseline | 손실 탐색: lovasz_quarter |
| 48 | m048_focal_0p5 | 완료 | baseline | 손실 탐색: focal_0.5 |
| 49 | m049_focal_1p0 | 완료 | baseline | 손실 탐색: focal_1.0 |
| 50 | m050_focal_2p0 | 완료 | baseline | 손실 탐색: focal_2.0 |
| 51 | m051_tversky_0p3 | 완료 | baseline | 손실 탐색: tversky_0.3 |
| 52 | m052_tversky_0p7 | 완료 | baseline | 손실 탐색: tversky_0.7 |
| 53 | m053_generalized_dice | 완료 | baseline | 손실 탐색: generalized_dice |
| 54 | m054_smooth_0p01 | 완료 | baseline | 손실 탐색: smooth_0.01 |
| 55 | m055_smooth_0p03 | 완료 | baseline | 손실 탐색: smooth_0.03 |
| 56 | m056_smooth_0p05 | 완료 | baseline | 손실 탐색: smooth_0.05 |
| 57 | m057_hard_pixels_0p75 | 완료 | baseline | 손실 탐색: hard_pixels_0.75 |
| 58 | m058_hard_pixels_0p5 | 완료 | baseline | 손실 탐색: hard_pixels_0.5 |
| 59 | m059_hard_pixels_0p25 | 완료 | baseline | 손실 탐색: hard_pixels_0.25 |
| 60 | m060_edge_ce_half | 완료 | baseline | 손실 탐색: edge_ce_half |
| 61 | m061_edge_ce_one | 완료 | baseline | 손실 탐색: edge_ce_one |
| 62 | m062_aux_01 | 완료 | baseline | 손실 탐색: aux_01 |
| 63 | m063_aux_02 | 완료 | baseline | 손실 탐색: aux_02 |
| 64 | m064_boundary_005 | 완료 | baseline | 손실 탐색: boundary_005 |
| 65 | m065_wd_zero | 완료 | baseline | 최적화 탐색: wd_zero |
| 66 | m066_wd_0001 | 완료 | baseline | 최적화 탐색: wd_0001 |
| 67 | m067_wd_001 | 완료 | baseline | 최적화 탐색: wd_001 |
| 68 | m068_wd_05 | 완료 | baseline | 최적화 탐색: wd_05 |
| 69 | m069_beta2_99 | 완료 | baseline | 최적화 탐색: beta2_99 |
| 70 | m070_beta1_95 | 완료 | baseline | 최적화 탐색: beta1_95 |
| 71 | m071_sgd_001 | 완료 | baseline | 최적화 탐색: sgd_001 |
| 72 | m072_sgd_003 | 완료 | baseline | 최적화 탐색: sgd_003 |
| 73 | m073_adam | 완료 | baseline | 최적화 탐색: adam |
| 74 | m074_amsgrad | 완료 | baseline | 최적화 탐색: amsgrad |
| 75 | m075_constant_lr | 완료 | baseline | 최적화 탐색: constant_lr |
| 76 | m076_poly_lr | 완료 | baseline | 최적화 탐색: poly_lr |
| 77 | m077_brightness | 완료 | baseline | 데이터 탐색: brightness |
| 78 | m078_contrast | 완료 | baseline | 데이터 탐색: contrast |
| 79 | m079_gamma | 완료 | baseline | 데이터 탐색: gamma |
| 80 | m080_noise | 완료 | baseline | 데이터 탐색: noise |
| 81 | m081_blur | 완료 | baseline | 데이터 탐색: blur |
| 82 | m082_drop_zscore | 완료 | baseline | 데이터 탐색: drop_zscore |
| 83 | m083_drop_cdf | 완료 | baseline | 데이터 탐색: drop_cdf |
| 84 | m084_sharpen | 완료 | baseline | 데이터 탐색: sharpen |
| 85 | m085_local_contrast | 완료 | baseline | 데이터 탐색: local_contrast |
| 86 | m086_erase | 완료 | baseline | 데이터 탐색: erase |
| 87 | m087_sample_al3ni | 완료 | baseline | 데이터 탐색: sample_al3ni |
| 88 | m088_sample_eutectic | 완료 | baseline | 데이터 탐색: sample_eutectic |
| 89 | m089_scratch_residual_w16 | 완료 | scratch | 처음부터 학습: residual U-Net, base width 16 |
| 90 | m090_scratch_residual_w24 | 완료 | scratch | 처음부터 학습: residual U-Net, base width 24 |
| 91 | m091_scratch_residual_w32 | 완료 | scratch | 처음부터 학습: residual U-Net, base width 32 |
| 92 | m092_scratch_separable_w16 | 완료 | scratch | 처음부터 학습: separable U-Net, base width 16 |
| 93 | m093_scratch_separable_w24 | 완료 | scratch | 처음부터 학습: separable U-Net, base width 24 |
| 94 | m094_scratch_separable_w32 | 완료 | scratch | 처음부터 학습: separable U-Net, base width 32 |
| 95 | m095_scratch_convnext_w16 | 완료 | scratch | 처음부터 학습: convnext U-Net, base width 16 |
| 96 | m096_scratch_convnext_w24 | 완료 | scratch | 처음부터 학습: convnext U-Net, base width 24 |
| 97 | m097_scratch_convnext_w32 | 완료 | scratch | 처음부터 학습: convnext U-Net, base width 32 |
| 98 | m098_scratch_attention_w16 | 완료 | scratch | 처음부터 학습: attention U-Net, base width 16 |
| 99 | m099_scratch_attention_w24 | 완료 | scratch | 처음부터 학습: attention U-Net, base width 24 |
| 100 | m100_scratch_attention_w32 | 완료 | scratch | 처음부터 학습: attention U-Net, base width 32 |

## 실패와 재실행

같은 전체 실행 명령을 다시 실행하면 완료 결과는 검사 후 재사용하고 미완료 모델을 재개합니다.

- `m003_ff_gentle_long`: CalledProcessError: Command '['/home/sppark/miniconda3/envs/komap/bin/python', '-B', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8', '/Data3/sppark/komap/experiments/hundred_v1/run.py', 'train', 'm003_ff_gentle_long', '--worker', '--gpus', '8']' returned non-zero exit status 1.

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
