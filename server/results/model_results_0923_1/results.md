# model_results_0923_1 — evolve_v1: hundred_v1 top-10 parent x 8기법+2조합 (100개, 이미지 mIoU 기준)

**상태**: 완료 (100개 실험 학습·Plain/D4 검증 평가 완료, 실패 0건)

## 1. 실험 개요

- **부모 모델 10개**: hundred_v1 실험에서 이미지 mIoU 상위 10위였던 parent 체크포인트
  (`m011_seed2028_parent`, `m013_fullframe_parent`, `m063_aux_02`, `m085_local_contrast`,
  `m009_head_only`, `m012_photo_parent`, `m008_decoder_only`, `m006_ema999`, `m062_aux_01`,
  `m016_aspp_w64_staged`).
- **적용 기법**: 8가지 evolve 기법을 각각 단독으로(light 레벨) 적용한 v1~v8, 그리고 경계 품질
  목적의 heavy 조합(v9, 기법1+4+5+6)과 복잡 텍스처 목적의 heavy 조합(v10, 기법2+3+7+8) 총 10개
  variant. 10 parent × 10 variant = **100개 모델**. 코드/설계는
  [experiments/evolve_v1/evolvelab/catalog.py](../experiments/evolve_v1/evolvelab/catalog.py)
  참고.
- **본 리포트는 요청에 따라 image mIoU만 기준으로 정리했다 (global mIoU는 표에서 제외).**
  전체 지표(global mIoU 포함)는
  [experiments/evolve_v1/outputs/reports/results.md](../experiments/evolve_v1/outputs/reports/results.md)
  에 그대로 보존되어 있다.

## 2. 결론 요약

- **최고 성능**: `e014_m013_v4_boundary_head` (m013_fullframe_parent + boundary_aux_plus 단독,
  light) — **image mIoU 78.8291%**. accuracy_v1 baseline(78.6865%, D4)보다 **+0.1426%p** 높다.
  상위 10개 전부 baseline보다 높거나(1~9위) 근소하게 낮지 않은 수준(10위 78.8056%도 baseline보다
  +0.12%p 높음) — 즉 **상위권은 baseline 대비 실질적으로(비록 작지만) 개선**됐다.
- **부모별 최고 평균은 `m013_fullframe_parent`** (10개 variant 평균 78.7894%, 최고 78.8291%),
  다음이 `m011_seed2028_parent`(평균 78.6732%, 최고 78.8066%). 이 두 parent에서 top10의 9개가
  나왔다 (10위 중 8개가 m013, 나머지 2개가 m011).
- **기법별로는 `v6_composition_head`가 명확히 최악**이다 (10개 parent 평균 78.4100%, 최저
  77.8694% — accuracy_v1 baseline보다 오히려 -0.81%p 낮은 사례도 있음). 나머지 9개 기법
  (v1~v5, v7~v10)은 평균 78.65~78.69% 범위로 서로 비슷하고 baseline과 큰 차이가 없다.
- **heavy 조합(v9/v10)이 light 단독 기법보다 낫지 않았다** (heavy_combo 평균 78.6618% vs
  light_single 평균 78.6509% — 거의 동일, 그룹 내 분산은 heavy_combo가 더 큼). 여러 기법을
  동시에 강하게(heavy) 켜는 것 자체가 추가 이득을 주지는 않았다는 뜻이다.
- **권장**: `composition_head` 단독 적용은 피할 것. 나머지 기법은 `m013_fullframe_parent` /
  `m011_seed2028_parent` 위에서는 baseline 대비 소폭 개선 여지가 있으나, 다른 8개 parent
  위에서는 baseline과 비슷하거나 소폭 낮아 일반화된 이득이라 보기 어렵다.

## 3. 기법(variant)별 평균 (10개 parent 평균, image mIoU)

| 기법 | 평균 | 최저 | 최고 |
|---|---:|---:|---:|
| v2_crop_mix | 78.6909% | 78.6517% | 78.7681% |
| v7_texture_branch | 78.6866% | 78.6351% | 78.8066% |
| v1_hires_decoder | 78.6850% | 78.6362% | 78.8073% |
| v8_complex_expert | 78.6844% | 78.6441% | 78.8076% |
| v4_boundary_head | 78.6834% | 78.6460% | **78.8291%** |
| v3_hires_branch | 78.6834% | 78.6345% | 78.8073% |
| v5_fullres_refine | 78.6833% | 78.6348% | 78.8058% |
| v10_texture_combo (heavy) | 78.6748% | 78.6196% | 78.7842% |
| v9_boundary_combo (heavy) | 78.6488% | 78.5928% | 78.7827% |
| v6_composition_head | **78.4100%** | **77.8694%** | 78.6952% |

## 4. 부모(parent)별 평균 (10개 기법 평균, image mIoU)

| 부모 | 평균 | 최고 |
|---|---:|---:|
| m013_fullframe_parent | **78.7894%** | **78.8291%** |
| m012_photo_parent | 78.6752% | 78.7057% |
| m011_seed2028_parent | 78.6732% | 78.8066% |
| m063_aux_02 | 78.6500% | 78.7099% |
| m085_local_contrast | 78.6342% | 78.6773% |
| m062_aux_01 | 78.6337% | 78.6838% |
| m006_ema999 | 78.6209% | 78.6606% |
| m009_head_only | 78.6201% | 78.6616% |
| m016_aspp_w64_staged | 78.6186% | 78.6614% |
| m008_decoder_only | 78.6152% | 78.6641% |

참고: accuracy_v1 baseline (D4) image mIoU = **78.6865%**.

## 5. 검증셋 마스크 비교 이미지 (상위 10개)

전체 100개 중 이미지 mIoU 상위 10개에 대해서만 검증셋 20장 전부의 4패널(원본/오차/예측 마스크/
실제 마스크, D4 예측 기준) 비교 이미지를 생성했다 (100개 전부 생성 시 2,000장이 되어 상위권만
선별). `valid_mask_comparison/<experiment>/<image_id>.png`, 브라우저에서
`valid_mask_comparison/index.html`을 열면 모델별로 훑어볼 수 있다.

## 6. 전체 100개 결과표 (image mIoU 기준 순위)

| 순위 | 실험 | 부모 | 그룹 | 이미지 mIoU |
|---:|---|---|---|---:|
| 1 | e014_m013_v4_boundary_head | m013_fullframe_parent | light_single | 78.8291% **(top10, 비교 이미지 있음)** |
| 2 | e018_m013_v8_complex_expert | m013_fullframe_parent | light_single | 78.8076% **(top10, 비교 이미지 있음)** |
| 3 | e013_m013_v3_hires_branch | m013_fullframe_parent | light_single | 78.8073% **(top10, 비교 이미지 있음)** |
| 4 | e011_m013_v1_hires_decoder | m013_fullframe_parent | light_single | 78.8073% **(top10, 비교 이미지 있음)** |
| 5 | e007_m011_v7_texture_branch | m011_seed2028_parent | light_single | 78.8066% **(top10, 비교 이미지 있음)** |
| 6 | e017_m013_v7_texture_branch | m013_fullframe_parent | light_single | 78.8062% **(top10, 비교 이미지 있음)** |
| 7 | e001_m011_v1_hires_decoder | m011_seed2028_parent | light_single | 78.8061% **(top10, 비교 이미지 있음)** |
| 8 | e015_m013_v5_fullres_refine | m013_fullframe_parent | light_single | 78.8058% **(top10, 비교 이미지 있음)** |
| 9 | e005_m011_v5_fullres_refine | m011_seed2028_parent | light_single | 78.8057% **(top10, 비교 이미지 있음)** |
| 10 | e003_m011_v3_hires_branch | m011_seed2028_parent | light_single | 78.8056% **(top10, 비교 이미지 있음)** |
| 11 | e008_m011_v8_complex_expert | m011_seed2028_parent | light_single | 78.8008% |
| 12 | e020_m013_v10_texture_combo | m013_fullframe_parent | heavy_combo | 78.7842% |
| 13 | e019_m013_v9_boundary_combo | m013_fullframe_parent | heavy_combo | 78.7827% |
| 14 | e012_m013_v2_crop_mix | m013_fullframe_parent | light_single | 78.7681% |
| 15 | e002_m011_v2_crop_mix | m011_seed2028_parent | light_single | 78.7671% |
| 16 | e010_m011_v10_texture_combo | m011_seed2028_parent | heavy_combo | 78.7575% |
| 17 | e004_m011_v4_boundary_head | m011_seed2028_parent | light_single | 78.7209% |
| 18 | e022_m063_v2_crop_mix | m063_aux_02 | light_single | 78.7099% |
| 19 | e059_m012_v9_boundary_combo | m012_photo_parent | heavy_combo | 78.7057% |
| 20 | e029_m063_v9_boundary_combo | m063_aux_02 | heavy_combo | 78.6990% |
| 21 | e016_m013_v6_composition_head | m013_fullframe_parent | light_single | 78.6952% |
| 22 | e060_m012_v10_texture_combo | m012_photo_parent | heavy_combo | 78.6943% |
| 23 | e030_m063_v10_texture_combo | m063_aux_02 | heavy_combo | 78.6905% |
| 24 | e057_m012_v7_texture_branch | m012_photo_parent | light_single | 78.6875% |
| 25 | e055_m012_v5_fullres_refine | m012_photo_parent | light_single | 78.6874% |
| 26 | e052_m012_v2_crop_mix | m012_photo_parent | light_single | 78.6872% |
| 27 | e058_m012_v8_complex_expert | m012_photo_parent | light_single | 78.6868% |
| 28 | e053_m012_v3_hires_branch | m012_photo_parent | light_single | 78.6868% |
| 29 | e051_m012_v1_hires_decoder | m012_photo_parent | light_single | 78.6857% |
| 30 | e082_m062_v2_crop_mix | m062_aux_01 | light_single | 78.6838% |
| 31 | e034_m085_v4_boundary_head | m085_local_contrast | light_single | 78.6773% |
| 32 | e031_m085_v1_hires_decoder | m085_local_contrast | light_single | 78.6716% |
| 33 | e037_m085_v7_texture_branch | m085_local_contrast | light_single | 78.6716% |
| 34 | e035_m085_v5_fullres_refine | m085_local_contrast | light_single | 78.6715% |
| 35 | e033_m085_v3_hires_branch | m085_local_contrast | light_single | 78.6713% |
| 36 | e038_m085_v8_complex_expert | m085_local_contrast | light_single | 78.6689% |
| 37 | e089_m062_v9_boundary_combo | m062_aux_01 | heavy_combo | 78.6685% |
| 38 | e090_m062_v10_texture_combo | m062_aux_01 | heavy_combo | 78.6665% |
| 39 | e054_m012_v4_boundary_head | m012_photo_parent | light_single | 78.6658% |
| 40 | e064_m008_v4_boundary_head | m008_decoder_only | light_single | 78.6641% |
| 41 | e032_m085_v2_crop_mix | m085_local_contrast | light_single | 78.6618% |
| 42 | e042_m009_v2_crop_mix | m009_head_only | light_single | 78.6616% |
| 43 | e044_m009_v4_boundary_head | m009_head_only | light_single | 78.6615% |
| 44 | e092_m016_v2_crop_mix | m016_aspp_w64_staged | light_single | 78.6614% |
| 45 | e094_m016_v4_boundary_head | m016_aspp_w64_staged | light_single | 78.6613% |
| 46 | e074_m006_v4_boundary_head | m006_ema999 | light_single | 78.6606% |
| 47 | e062_m008_v2_crop_mix | m008_decoder_only | light_single | 78.6563% |
| 48 | e023_m063_v3_hires_branch | m063_aux_02 | light_single | 78.6563% |
| 49 | e027_m063_v7_texture_branch | m063_aux_02 | light_single | 78.6561% |
| 50 | e021_m063_v1_hires_decoder | m063_aux_02 | light_single | 78.6558% |
| 51 | e025_m063_v5_fullres_refine | m063_aux_02 | light_single | 78.6558% |
| 52 | e097_m016_v7_texture_branch | m016_aspp_w64_staged | light_single | 78.6528% |
| 53 | e041_m009_v1_hires_decoder | m009_head_only | light_single | 78.6527% |
| 54 | e047_m009_v7_texture_branch | m009_head_only | light_single | 78.6521% |
| 55 | e028_m063_v8_complex_expert | m063_aux_02 | light_single | 78.6521% |
| 56 | e072_m006_v2_crop_mix | m006_ema999 | light_single | 78.6517% |
| 57 | e073_m006_v3_hires_branch | m006_ema999 | light_single | 78.6517% |
| 58 | e077_m006_v7_texture_branch | m006_ema999 | light_single | 78.6509% |
| 59 | e071_m006_v1_hires_decoder | m006_ema999 | light_single | 78.6505% |
| 60 | e075_m006_v5_fullres_refine | m006_ema999 | light_single | 78.6504% |
| 61 | e081_m062_v1_hires_decoder | m062_aux_01 | light_single | 78.6476% |
| 62 | e024_m063_v4_boundary_head | m063_aux_02 | light_single | 78.6471% |
| 63 | e083_m062_v3_hires_branch | m062_aux_01 | light_single | 78.6471% |
| 64 | e048_m009_v8_complex_expert | m009_head_only | light_single | 78.6469% |
| 65 | e087_m062_v7_texture_branch | m062_aux_01 | light_single | 78.6467% |
| 66 | e085_m062_v5_fullres_refine | m062_aux_01 | light_single | 78.6466% |
| 67 | e098_m016_v8_complex_expert | m016_aspp_w64_staged | light_single | 78.6466% |
| 68 | e078_m006_v8_complex_expert | m006_ema999 | light_single | 78.6461% |
| 69 | e084_m062_v4_boundary_head | m062_aux_01 | light_single | 78.6460% |
| 70 | e088_m062_v8_complex_expert | m062_aux_01 | light_single | 78.6444% |
| 71 | e068_m008_v8_complex_expert | m008_decoder_only | light_single | 78.6441% |
| 72 | e045_m009_v5_fullres_refine | m009_head_only | light_single | 78.6375% |
| 73 | e095_m016_v5_fullres_refine | m016_aspp_w64_staged | light_single | 78.6373% |
| 74 | e093_m016_v3_hires_branch | m016_aspp_w64_staged | light_single | 78.6372% |
| 75 | e091_m016_v1_hires_decoder | m016_aspp_w64_staged | light_single | 78.6367% |
| 76 | e061_m008_v1_hires_decoder | m008_decoder_only | light_single | 78.6362% |
| 77 | e043_m009_v3_hires_branch | m009_head_only | light_single | 78.6359% |
| 78 | e067_m008_v7_texture_branch | m008_decoder_only | light_single | 78.6351% |
| 79 | e065_m008_v5_fullres_refine | m008_decoder_only | light_single | 78.6348% |
| 80 | e063_m008_v3_hires_branch | m008_decoder_only | light_single | 78.6345% |
| 81 | e100_m016_v10_texture_combo | m016_aspp_w64_staged | heavy_combo | 78.6341% |
| 82 | e050_m009_v10_texture_combo | m009_head_only | heavy_combo | 78.6337% |
| 83 | e080_m006_v10_texture_combo | m006_ema999 | heavy_combo | 78.6336% |
| 84 | e070_m008_v10_texture_combo | m008_decoder_only | heavy_combo | 78.6335% |
| 85 | e040_m085_v10_texture_combo | m085_local_contrast | heavy_combo | 78.6196% |
| 86 | e039_m085_v9_boundary_combo | m085_local_contrast | heavy_combo | 78.6122% |
| 87 | e099_m016_v9_boundary_combo | m016_aspp_w64_staged | heavy_combo | 78.6090% |
| 88 | e049_m009_v9_boundary_combo | m009_head_only | heavy_combo | 78.6089% |
| 89 | e069_m008_v9_boundary_combo | m008_decoder_only | heavy_combo | 78.6055% |
| 90 | e079_m006_v9_boundary_combo | m006_ema999 | heavy_combo | 78.6037% |
| 91 | e009_m011_v9_boundary_combo | m011_seed2028_parent | heavy_combo | 78.5928% |
| 92 | e056_m012_v6_composition_head | m012_photo_parent | light_single | 78.5646% |
| 93 | e026_m063_v6_composition_head | m063_aux_02 | light_single | 78.4776% |
| 94 | e086_m062_v6_composition_head | m062_aux_01 | light_single | 78.4401% |
| 95 | e036_m085_v6_composition_head | m085_local_contrast | light_single | 78.4164% |
| 96 | e046_m009_v6_composition_head | m009_head_only | light_single | 78.4098% |
| 97 | e076_m006_v6_composition_head | m006_ema999 | light_single | 78.4096% |
| 98 | e096_m016_v6_composition_head | m016_aspp_w64_staged | light_single | 78.4091% |
| 99 | e066_m008_v6_composition_head | m008_decoder_only | light_single | 78.4083% |
| 100 | e006_m011_v6_composition_head | m011_seed2028_parent | light_single | 77.8694% |

## 7. 실행 방법 (재현)

```bash
cd /Data3/sppark/komap
conda activate komap
python -B experiments/evolve_v1/run.py preflight
python -B experiments/evolve_v1/run_parallel.py   # 여러 GPU에 자동 분산 실행
# 또는 GPU 1장:  python -B experiments/evolve_v1/run.py suite
```

산출물 경로:

- 체크포인트: `experiments/evolve_v1/outputs/runs/<experiment>/best.pt`
- 검증 지표: `experiments/evolve_v1/outputs/evaluations/<experiment>/{plain,d4}/metrics_summary.json`
- 리포트(global mIoU 포함 전체): `experiments/evolve_v1/outputs/reports/results.md`, `results.csv`

---
*본 실험은 [spec.md](../../spec.md) 및 [AGENTS.md](../../AGENTS.md) 지침에 따라 기존 모델,
소스코드 및 이전 결과(accuracy_v1, hundred_v1)를 일절 수정하지 않고 독립 경로
(`experiments/evolve_v1/`)에서 구축되었다.*
