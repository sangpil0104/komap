**HRNet_taeho는 48위, 새 기준 image mIoU는 78.664330%입니다.**

빈 클래스 IoU=0, valid 20장 평균 기준입니다. 동일 모델의 중복 저장 기록을 합쳤습니다. B4 U-Net은 EfficientNet-B4 U-Net을 뜻합니다. HRNet 두 모델은 four_flip TTA, 나머지는 D4 TTA 결과입니다.

| 순위 | 모델 | image mIoU |
|---:|---|---:|
| 1 | B4 U-Net seed 32 | 78.976546% |
| 2 | B4 U-Net seed 63 | 78.959910% |
| 3 | B4 U-Net seed 11 | 78.941877% |
| 4 | HRNet_taeho_no_gaussian | 78.917288% |
| 5 | B4 U-Net seed 80 | 78.828614% |
| 6 | B4 U-Net seed 37 | 78.811772% |
| 7 | B4 U-Net seed 61 | 78.747686% |
| 8 | photo_mild | 78.720305% |
| 9 | e022_m063_v2_crop_mix | 78.709924% |
| 10 | B4 U-Net seed 46 | 78.707995% |
| 11 | e059_m012_v9_boundary_combo | 78.705745% |
| 12 | e029_m063_v9_boundary_combo | 78.698951% |
| 13 | e060_m012_v10_texture_combo | 78.694346% |
| 14 | m063_aux_02 | 78.692784% |
| 15 | e030_m063_v10_texture_combo | 78.690486% |
| 16 | m085_local_contrast | 78.689761% |
| 17 | m009_head_only | 78.688783% |
| 18 | m012_photo_parent | 78.688187% |
| 19 | e057_m012_v7_texture_branch | 78.687478% |
| 20 | e055_m012_v5_fullres_refine | 78.687436% |
| 21 | e052_m012_v2_crop_mix | 78.687215% |
| 22 | e058_m012_v8_complex_expert | 78.686812% |
| 23 | e053_m012_v3_hires_branch | 78.686792% |
| 24 | CNN U-Net baseline | 78.686480% |
| 25 | m008_decoder_only | 78.685911% |
| 26 | e051_m012_v1_hires_decoder | 78.685651% |
| 27 | e082_m062_v2_crop_mix | 78.683826% |
| 28 | m006_ema999 | 78.680404% |
| 29 | m062_aux_01 | 78.679953% |
| 30 | e034_m085_v4_boundary_head | 78.677321% |
| 31 | m016_aspp_w64_staged | 78.677032% |
| 32 | m010_encoder_slow | 78.676000% |
| 33 | m015_aspp_w32_staged | 78.675895% |
| 34 | m059_hard_pixels_0p25 | 78.675780% |
| 35 | e031_m085_v1_hires_decoder | 78.671635% |
| 36 | e037_m085_v7_texture_branch | 78.671615% |
| 37 | e035_m085_v5_fullres_refine | 78.671454% |
| 38 | e033_m085_v3_hires_branch | 78.671287% |
| 39 | e038_m085_v8_complex_expert | 78.668885% |
| 40 | e089_m062_v9_boundary_combo | 78.668532% |
| 41 | m054_smooth_0p01 | 78.667729% |
| 42 | m078_contrast | 78.667299% |
| 43 | m073_adam | 78.666630% |
| 44 | e090_m062_v10_texture_combo | 78.666528% |
| 45 | m064_boundary_005 | 78.666104% |
| 46 | e054_m012_v4_boundary_head | 78.665813% |
| 47 | m020_aux_light | 78.665731% |
| 48 | HRNet_taeho | 78.664330% |

순위 범위는 이미지별 지표 또는 예측 마스크가 남아 있어 재계산 가능한 저장 평가 결과입니다. 과거 요약 문서만 남은 모델은 포함하지 않았습니다.

추론 표기 정정: 앞서 작성한 model_results_0929_1.md의 최상위 표에서 HRNet_taeho_no_gaussian에 붙인 D4 TTA 표기는 four_flip TTA가 정확합니다. 원본 summary.json의 inference 설정으로 확인했으며 점수와 순위에는 영향이 없습니다. 기존 보고서는 보존했습니다.
