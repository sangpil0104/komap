# Model349 모델 전달 — lite13 retuned와 lite20 시드 탐색

100회 학습 시드 탐색에서 생성한 **HRNet347 12개, B4-326 8개, B4-322 4개, B4 warmstart 24개, HRNet392 8개**, 총 **56개 가중치**를 담았습니다. 여기에 최종 모델의 고정 **B4-329 3개**를 추가해 lite20 전달분은 **59개 가중치 / 29개 모델 ZIP**입니다. 새 lite13 완성 모델 13개를 포함해 저장소 전체는 **72개 가중치 / 30개 모델 ZIP**입니다.
시드 탐색 모델은 평가 단위에 맞춘 **28개 ZIP**이며, 각 ZIP은 원본 `last.pt`, `args.json`, 학습 명령, 점수/seed/fold/적용 위치, SHA256을 포함합니다.
B4 warmstart는 BN 통계 고정 확인 기록도 포함합니다. 기존 full-train70 r70 실험은 이번 전달 범위와 별도입니다.

## 새 모델: lite13 retuned — 2026-10-14

**[13개 가중치와 추론 코드 ZIP 다운로드](models/Model349-lite13-retuned/Model349_lite13_retuned_retrained_20261014.zip)** — 1.813 GB.
[사용법·구성](reports/model_results_20261014_1/README.md) · [전체 결과서](reports/model_results_20261014_1/model_results_20261014_1.md).

전체 R7+E5+F1 앙상블의 valid20 image mIoU는 **80.471931%**입니다.
선행/OOF 포함 33개 학습 작업에서 얻은 최종 13개 가중치이며, 이 ZIP 하나에 추론에 필요한 모델이 모두 있습니다.

| 데이터 | 실측 전체 추론시간 | 장당 평균 |
|---|---:|---:|
| valid 20장 | 407.561초 | 20.378초 |
| test 10장 | 223.297초 | 22.330초 |

RTX 4090, R은 GPU 3~6에 분배, E는 GPU 3, F는 CPU입니다. 모델 로딩·전처리·TTA/TENT·후처리·저장까지 포함한 전체 경과시간입니다.
학습·채점·보고서 생성은 제외하며 장당 값은 전체 시간/이미지 수입니다. test 정답이 없어 test 점수는 없습니다.
일부 checkpoint 선택과 원본 비율 조정에 valid20을 사용했습니다. 공개 원래 가중치의 80.779378%와 이번 재학습 점수는 구분합니다.

```bash
git lfs pull --include="models/Model349-lite13-retuned/Model349_lite13_retuned_retrained_20261014.zip"
python verify_download.py --family Lite13-retuned --deep
unzip models/Model349-lite13-retuned/Model349_lite13_retuned_retrained_20261014.zip
cd Model349_lite13_retuned_retrained_20261014
conda activate komap
python predict_portable.py --verify-only
python predict_portable.py --images /path/to/images --output /path/to/new_output --gpus 0
```

아래 lite20 후보별 점수와는 별도 실험입니다. 기존 59개 가중치 파일은 그대로 보존했습니다.

## lite20 계열별 최고 후보

**아래 점수는 해당 후보를 적용한 전체 R+E+F 앙상블의 valid20 image mIoU입니다. 개별 모델 단독 성능이 아닙니다.**
기존 lite20 기준은 **80.255846%**입니다. HRNet347은 명시된 HR336 부모와 함께 교체했으며, fold 모델들은 묶음으로 평가했습니다.
최고 후보는 각 계열에서 이번에 새로 학습한 후보 중 선정했습니다. 기존 가중치가 더 높은 경우도 아래에 표시합니다.

| 계열 | 가중치 수 / 후보 수 | 최고 후보 | 학습 seed (fold/slot) | image mIoU | 기준 대비 %p |
|---|---:|---|---|---:|---:|
| HRNet347 | 12 / 12 | [347_05](models/HRNet347/347_05_seed3424.zip) | 3424 | **80.373932%** | +0.118087 |
| B4-326 | 8 / 4 | [326_set1](models/B4-326/326_set1.zip) | fold1: 3457, fold2: 3458 | **80.303757%** | +0.047911 |
| B4-322 | 4 / 4 | [322_set2](models/B4-322/322_set2.zip) | fold3: 3465 | **80.258059%** | +0.002214 |
| B4-warmstart | 24 / 6 | [E_b4_set3](models/B4-warmstart/E_b4_set3.zip) | fold1: 3479, fold2: 3480, fold3: 3481, fold4: 3482 | **80.253095%** | -0.002750 |
| HRNet392 | 8 / 2 | [E_hr_set1](models/HRNet392/E_hr_set1.zip) | slot1/fold2: 3495, slot1/fold4: 3496, slot2/fold1: 3497, slot2/fold4: 3498 | **80.236614%** | -0.019232 |

- **HRNet347 최고**: `347_05`, 학습 seed **3424**, 부모 HR336=`336_05` (seed **3412**). 부모와 함께 사용한 평가 점수입니다.
- **B4-326 최고**: `326_set1`의 fold1 seed **3457**, fold2 seed **3458**을 함께 사용합니다.
- **B4-322 최고**: `322_set2`, fold3 seed **3465**입니다.
- **B4 warmstart 최고**: `E_b4_set3`, fold1~4 seed **3479, 3480, 3481, 3482**입니다.
- **HRNet392 최고**: `E_hr_set1`, slot1/fold2=**3495**, slot1/fold4=**3496**, slot2/fold1=**3497**, slot2/fold4=**3498**입니다.
- `slot1/slot2`는 기존 모델 경로의 `seed1/seed2` 슬롯을 뜻합니다. 실제 학습 난수 seed는 위의 34xx 값입니다.
- 새 B4 warmstart 및 HRNet392 후보는 모두 기존 기준보다 낮아 최종 선택에서는 **기존 E 가중치를 유지**했습니다.
- 최종 조합은 HRNet347 `347_07`(seed3426), B4-326 `326_set1`, B4-322 `322_set2` 등을 사용해 **80.479068%**입니다. 조합 선택 때문에 HRNet347의 계열별 최고(`347_05`)와 최종 선택(`347_07`)이 다릅니다.
- 이 저장소의 6개 계열만으로 최종 20개 모델 전체가 구성되지는 않습니다. R의 다른 구성 모델, HR336, F 및 기준 E 가중치는 별도입니다.

## 추가된 B4-329 — 고정 seed0, fold 1·3·4

[B4-329 가중치 3개 ZIP 다운로드](models/B4-329/329_fixed_seed0_folds134.zip) — 약 227.5 MB.

| 모델 | 가중치 수 | fold | 학습 seed | 학습 길이 | 역할 |
|---|---:|---|---:|---|---|
| R B4-329 | 3 | 1, 3, 4 | 0 | 각 16,000 iterations | 100회 시드 탐색에서 고정 사용 |

`model_results_20261012_1/selected_model`의 실제 배포 가중치와 `args.json`을 그대로 담았으며 SHA256을 검증했습니다.
전처리는 zcdf3 / image_global / target_width4000 / uniform이고, 모델은 UNet / efficientnet-b4입니다.
적용 경로는 `Model349_exh14/weights/model329_fullstats_width4000_uniform_group_oof/fold{1,3,4}/last.pt`입니다.

**이 B4-329에 대해 별도의 단독 성능 또는 시드별 최고 점수는 측정하지 않았습니다.**
기준 전체 R+E+F의 80.255846%와 최종 조합의 80.479068%는 이 가중치를 고정 사용한 앙상블 점수입니다.
이번 파일은 기존 R의 fold 가중치이며, E B4 warmstart 및 별도 r70 full-train70 B4-329와 구분됩니다.
100회 탐색의 후보 수는 28개로 그대로이고, 고정 모델 ZIP 1개가 추가됐습니다.

B4-329만 받으려면 저장소에서 다음을 실행하세요.

```bash
git lfs pull --include="models/B4-329/329_fixed_seed0_folds134.zip"
python verify_download.py --family B4-329 --deep
```

## 다운로드와 검증

가중치 ZIP 전체 크기는 소스 ZIP을 포함해 약 **9.79 GiB**입니다. 모델 ZIP은 Git LFS에 저장됩니다.
위 표의 최고 후보 링크 또는 아래 전체 목록에서 파일 페이지의 **Download raw file**로 ZIP을 받을 수 있습니다.
전체 clone은 Git LFS를 설치한 환경에서 다음과 같이 합니다.

```bash
git lfs install
git clone https://github.com/CNUKoMaP/giveItToHyeonSeok.git
cd giveItToHyeonSeok
python verify_download.py --deep
```

최고 후보 5개만 받으려면:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/CNUKoMaP/giveItToHyeonSeok.git
cd giveItToHyeonSeok
git lfs pull --include="models/HRNet347/347_05_seed3424.zip,models/B4-326/326_set1.zip,models/B4-322/322_set2.zip,models/B4-warmstart/E_b4_set3.zip,models/HRNet392/E_hr_set1.zip"
python verify_download.py --best --deep
```

일반 ZIP(Deflate) 형식이며 폴더 구조를 유지해 압축을 해제하세요. `last.pt` 옆에 해당 `args.json`을 둡니다.
`candidate.json`과 `metadata/checkpoints.csv`의 `deployed_weight_key`로 원래 앙상블의 적용 위치를 확인할 수 있습니다.
HRNet347은 완전한 모델 가중치이므로 자체 로딩에 부모 파일은 필요하지 않지만, 표의 앙상블 점수를 재현하려면 함께 평가한 HR336 부모 가중치도 필요합니다. 부모 정보/해시는 `metadata/parent_dependencies.json`에 있습니다.
HR336 seed3411/3412는 이전 [8개 HR336 가중치 ZIP](https://github.com/CNUKoMaP/sppark/blob/main/model349_lite20_seeds3407-3414_weights.zip)에도 포함되어 있습니다. 해당 저장소 접근 권한이 필요합니다.

## 지표와 학습 조건

- valid 이미지마다 네 클래스 IoU를 평균한 뒤 valid20의 평균을 계산합니다. 정답과 예측에 모두 없는 클래스는 IoU=0입니다. global mIoU는 계산하지 않았습니다.
- 동일한 valid20에서 반복 비교/선택한 개발용 점수이며, 독립 test 성능이 아닙니다.
- HRNet347: HR336 체크포인트에서 12 epochs 미세조정, 역빈도 CE 가중치.
- B4-326: zcdf3 / image_global / scale3 / area, fold1·2 각각 16,000 iterations.
- B4-322: gray3 / imagenet / scale3 / area, fold3 16,000 iterations.
- B4 warmstart: zcdf3 / image_global / target_width4000 / uniform, 기존 B4에서 BN 통계 고정, 각 fold 2,000 iterations.
- HRNet392: UnetPlusPlus / tu-hrnet_w48, patch512 / input_scale2 / width 정규화, 각 모델 70 epochs.
- 실행 환경은 conda `komap`, 사용 GPU는 물리 4~7입니다. 학습 인자의 상세값은 각 원본 `args.json` 및 `train_command.json`에 있습니다.

## 전체 후보 목록

### HRNet347

| 후보 / ZIP | 학습 seed (fold/slot) | HR336 부모 | 가중치 수 | image mIoU |
|---|---|---|---:|---:|
| [347_05](models/HRNet347/347_05_seed3424.zip) | 3424 | 336_05 / seed3412 | 1 | 80.373932% |
| [347_07](models/HRNet347/347_07_seed3426.zip) | 3426 | 336_05 / seed3412 | 1 | 80.371974% |
| [347_06](models/HRNet347/347_06_seed3425.zip) | 3425 | 336_05 / seed3412 | 1 | 80.371506% |
| [347_04](models/HRNet347/347_04_seed3423.zip) | 3423 | 336_05 / seed3412 | 1 | 80.369329% |
| [347_09](models/HRNet347/347_09_seed3428.zip) | 3428 | 336_04 / seed3411 | 1 | 80.330144% |
| [347_11](models/HRNet347/347_11_seed3430.zip) | 3430 | 336_04 / seed3411 | 1 | 80.329757% |
| [347_10](models/HRNet347/347_10_seed3429.zip) | 3429 | 336_04 / seed3411 | 1 | 80.329649% |
| [347_08](models/HRNet347/347_08_seed3427.zip) | 3427 | 336_04 / seed3411 | 1 | 80.329309% |
| [347_01](models/HRNet347/347_01_seed3420.zip) | 3420 | baseline_336 / seed0 | 1 | 80.256734% |
| [347_02](models/HRNet347/347_02_seed3421.zip) | 3421 | baseline_336 / seed0 | 1 | 80.256434% |
| [347_03](models/HRNet347/347_03_seed3422.zip) | 3422 | baseline_336 / seed0 | 1 | 80.255416% |
| [347_00](models/HRNet347/347_00_seed3419.zip) | 3419 | baseline_336 / seed0 | 1 | 80.255310% |

### B4-326

| 후보 / ZIP | 학습 seed (fold/slot) | HR336 부모 | 가중치 수 | image mIoU |
|---|---|---|---:|---:|
| [326_set1](models/B4-326/326_set1.zip) | fold1: 3457, fold2: 3458 | — | 2 | 80.303757% |
| [326_set2](models/B4-326/326_set2.zip) | fold1: 3459, fold2: 3460 | — | 2 | 80.262804% |
| [326_set0](models/B4-326/326_set0.zip) | fold1: 3455, fold2: 3456 | — | 2 | 80.260848% |
| [326_set3](models/B4-326/326_set3.zip) | fold1: 3461, fold2: 3462 | — | 2 | 80.253731% |

### B4-322

| 후보 / ZIP | 학습 seed (fold/slot) | HR336 부모 | 가중치 수 | image mIoU |
|---|---|---|---:|---:|
| [322_set2](models/B4-322/322_set2.zip) | fold3: 3465 | — | 1 | 80.258059% |
| [322_set1](models/B4-322/322_set1.zip) | fold3: 3464 | — | 1 | 80.256984% |
| [322_set0](models/B4-322/322_set0.zip) | fold3: 3463 | — | 1 | 80.254756% |
| [322_set3](models/B4-322/322_set3.zip) | fold3: 3466 | — | 1 | 80.253280% |

### B4-warmstart

| 후보 / ZIP | 학습 seed (fold/slot) | HR336 부모 | 가중치 수 | image mIoU |
|---|---|---|---:|---:|
| [E_b4_set3](models/B4-warmstart/E_b4_set3.zip) | fold1: 3479, fold2: 3480, fold3: 3481, fold4: 3482 | — | 4 | 80.253095% |
| [E_b4_set1](models/B4-warmstart/E_b4_set1.zip) | fold1: 3471, fold2: 3472, fold3: 3473, fold4: 3474 | — | 4 | 80.250952% |
| [E_b4_set5](models/B4-warmstart/E_b4_set5.zip) | fold1: 3487, fold2: 3488, fold3: 3489, fold4: 3490 | — | 4 | 80.248696% |
| [E_b4_set4](models/B4-warmstart/E_b4_set4.zip) | fold1: 3483, fold2: 3484, fold3: 3485, fold4: 3486 | — | 4 | 80.247329% |
| [E_b4_set2](models/B4-warmstart/E_b4_set2.zip) | fold1: 3475, fold2: 3476, fold3: 3477, fold4: 3478 | — | 4 | 80.246791% |
| [E_b4_set0](models/B4-warmstart/E_b4_set0.zip) | fold1: 3467, fold2: 3468, fold3: 3469, fold4: 3470 | — | 4 | 80.245746% |

### HRNet392

| 후보 / ZIP | 학습 seed (fold/slot) | HR336 부모 | 가중치 수 | image mIoU |
|---|---|---|---:|---:|
| [E_hr_set1](models/HRNet392/E_hr_set1.zip) | slot1/fold2: 3495, slot1/fold4: 3496, slot2/fold1: 3497, slot2/fold4: 3498 | — | 4 | 80.236614% |
| [E_hr_set0](models/HRNet392/E_hr_set0.zip) | slot1/fold2: 3491, slot1/fold4: 3492, slot2/fold1: 3493, slot2/fold4: 3494 | — | 4 | 80.231845% |

## 함께 제공한 기록

- `metadata/best_per_family.json`: 계열별 최고 후보와 구성원.
- `metadata/candidate_scores.csv`, `metadata/all_candidates.json`: 28개 후보의 전체 점수.
- `metadata/checkpoints.csv/json`: 72개 가중치의 seed/fold/slot/적용 경로/SHA256.
- `metadata/candidates/`: 후보별 valid20 이미지 점수와 출처.
- `metadata/additional_models.json`: 추가한 고정 B4-329 가중치 3개와 앙상블 문맥 정보.
- `source_snapshot.zip`: 해당 탐색에서 동결한 모델/학습/추론 소스. 원본 소스 해시와 대조했습니다.
- `environment.freeze.txt`: 실제 conda komap의 패키지 버전.
- `metadata/original_plan.json`, `metadata/original_final_report.md`: 원래 100회 계획과 최종 결과.
- `manifest.json`, `SHA256SUMS.txt`, `verify_download.py`: ZIP 및 압축 내부 가중치 검증.
- 학습 데이터 및 나머지 앙상블 가중치는 포함하지 않았습니다. 원본 코드와 명령의 서버 절대경로는 수신 환경에 맞게 지정해야 합니다.

- `metadata/complete_models.json`: lite13 retuned 전체 모델 13개, 구성과 실측 점수/추론시간.
