# lite19a R329(fold) 시드 탐색 결과

`CNUKoMaP/model349-lite19a-r70-swa` 릴리즈 v1의 **lite19a** 구성에서, R의 B4-329 fold 모델만 시드를 바꿔 다시 학습하고 valid20으로 채점했습니다.

**결론: 새 시드 6개 모두 릴리즈 가중치(orig, 80.5236%)보다 낮았습니다(0/6).** 가장 높은 시드는 1562(80.5067%, −0.017%p)였습니다. 시드만 바꿨을 때 valid20은 표준편차 약 0.1%p로 흔들립니다.

## 실험 설정

| 항목 | 내용 |
|---|---|
| 기준 구성 | lite19a (`selected_config_lite19a.json`), 가중치 19개 |
| 바꾼 것 | R의 `model329_fullstats_width4000_uniform_group_oof` **fold 1·3·4** (lite19a가 쓰는 fold만) |
| 고정한 것 | 나머지 R 가중치, E(`model329_ws`, HRNet392), F(refiner), 결합 비율·gamma — 전부 릴리즈 v1 그대로 |
| 학습 레시피 | `retrain/retrain_r.sh`의 B4-329와 동일: UNet / efficientnet-b4, input zcdf3, norm image_global, scale 1, target_width 4000, sample uniform, crop 1536, bs 2, **16,000 iters**, lr 3e-4, wd 1e-2, dice_w 1, `--no_valid_eval`(valid 미사용, last.pt) |
| fold 분할 | 번들의 `prepare_group_cv.py` + `model303_progress.json`으로 생성한 grouped CV (fold1 52장, fold3 53장, fold4 53장) |
| 시드 | `train.py --seed`에 **1557, 1558, …, 1562**를 그대로 넣음 (`retrain_r.sh`의 `0 + 1000·S` 규칙이 아님) |
| 추론·채점 | 시드마다 lite19a로 valid20, test10 전체 추론(R+E+F) → `score.sh`(image-wise 4-class mIoU) |
| orig | 릴리즈 329 fold 1·3·4 그대로 같은 경로로 추론. **80.523642%** = 번들 README의 lite19a 값(80.5236%)과 일치해 재현 확인 |
| 환경 | RTX 3090 (1장에 시드 1개, GPU 3장 병렬), torch 2.x + smp 0.5.0 + timm 1.0.29 |
| 소요 시간 | fold 1개 119~147분, 시드 1개 약 6.5~7.5시간 + 추론 약 45분 |

## 결과

| 시드 | valid20 | orig 대비 | 개선/악화 장수 | 최악 이미지 (차이) | test10 마스크 차이 |
|---|---:|---:|:--:|---|---:|
| orig | **80.523642%** | — | | | |
| 1557 | 80.310724% | −0.2129%p | 8 / 12 | A71_step2_sample1 (−1.50) | 0.069% |
| 1558 | 80.462612% | −0.0610%p | 11 / 9 | A186_step2_sample2 (−1.13) | 0.066% |
| 1559 | 80.250768% | −0.2729%p | 5 / 15 | A71_step2_sample1 (−3.24) | 0.065% |
| 1560 | 80.406448% | −0.1172%p | 10 / 10 | A186_step2_sample2 (−1.86) | 0.074% |
| 1561 | 80.333914% | −0.1897%p | 13 / 7 | A186_step2_sample2 (−1.91) | 0.071% |
| 1562 | 80.506727% | −0.0169%p | 12 / 8 | A42_step2_sample4 (−0.46) | 0.068% |

- 새 시드 6개: 평균 **80.3785%**, 표준편차 **0.097%p**, 범위 80.2508~80.5067 (폭 0.26%p)
- orig는 새 시드 평균보다 0.145%p(표준편차의 약 1.5배) 높고, orig를 넘은 시드는 0/6
- test10 마스크는 시드마다 orig 대비 0.065~0.074% 픽셀이 다름

### 시드 편차가 큰 이미지

| 이미지 | 시드 간 표준편차 | 시드 6개 범위 | orig |
|---|---:|---|---:|
| A71_step2_sample1 | 1.51%p | 51.20 ~ 55.02 | 54.44 |
| A186_step2_sample2 | 0.81%p | 80.65 ~ 82.68 | 82.55 |
| A11_step1_sample4 | 0.50%p | 78.32 ~ 79.87 | 79.31 |
| A181_step1_sample1 | 0.35%p | 79.32 ~ 80.31 | 80.09 |

이미지별 시드 편차의 중앙값은 약 0.1%p로, 점수 차이의 대부분이 위 몇 장에서 나옵니다.

## 해석

- orig의 329 fold 선택(1·3·4)과 결합 비율(gamma329)은 릴리즈 가중치를 놓고 valid20으로 정해졌습니다. 그래서 같은 레시피의 새 시드가 valid20에서 orig보다 약간 낮게 나오는 것은 예상 범위입니다(orig 쪽에 선택 이득이 들어 있음).
- 시드 편차(표준편차 약 0.1%p)가 orig와의 차이(약 0.15%p)와 비슷한 크기이므로, valid20으로 최고 시드를 골라 교체하는 것은 근거가 약합니다. 1562(−0.017%p)도 6개 중 최고를 고른 값이라 orig와 차이가 없다고 보는 것이 맞습니다.
- 팀원 시드 탐색의 +0.19%p 이득도 이 편차 범위와 비슷한 크기입니다.

## 첨부 파일

| 경로 | 내용 |
|---|---|
| `seed{1557..1562}/model329_fullstats_width4000_uniform_group_oof/fold{1,3,4}/` | 시드별 가중치 `last.pt`, `args.json`, `train.log` |
| `seed{1557..1562}/valid_metrics.json` | 시드별 valid20 이미지별 점수 |
| `SHA256SUMS.txt` | 가중치 체크섬 |
| `summary.txt`, `summary.json` | 위 결과표 원본 |
| `events.log` | 전체 실행 기록 (단계별 시작/종료 시각, 학습 마지막 줄, 결과) |
| `scripts/` | 실행 스크립트 (`run.sh`, `summarize.py`, `stop_now.sh`) |

### 사용법

각 `seedN/` 아래는 R 가중치 폴더 구조 그대로입니다. 릴리즈 `WEIGHTS/R`의 `model329_fullstats_width4000_uniform_group_oof`를 원하는 시드의 폴더로 바꾸고, 재학습 가중치라 `--allow-retrained`를 붙여 추론합니다.

```sh
sha256sum -c SHA256SUMS.txt
# WEIGHTS/R 의 나머지는 그대로 두고 model329_... 만 seed1562/model329_... 로 교체(링크)한 뒤
bash run_infer_lite19a.sh WEIGHTS IMAGES OUT GPU --allow-retrained
```
