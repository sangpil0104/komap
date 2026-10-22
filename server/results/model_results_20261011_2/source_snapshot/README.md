# Model349 lite20_r70 실험

GitHub `Model349_lite20_r70_bundle_code.zip`을 고정 커밋에서 내려받아 별도 실험으로 준비했습니다. **GPU 물리4~7 / conda komap / nohup**을 사용합니다. 학습 시작은 사용자가 합니다.

```bash
cd /Data3/sppark/komap/experiments/model349_lite20_r70_retrain_gpu4567_v1
./start_nohup.sh
tail -F logs/nohup_latest.log
```

현재 같은 GPU의 seed100 탐색이 실행 중이면 종료될 때까지 기다립니다. 기존 작업을 중단하지 않습니다. GPU도 메모리64MiB 이하/사용률0일 때만 배정합니다. 큐 대기 상태, 단계 진행률, 학습 진행률, 경과 시간과 예상 남은 작업 시간이 전체 로그에 나옵니다. ETA에는 다른 작업의 종료를 기다리는 시간이 포함되지 않습니다. 각 작업에는 상세 `tqdm` 로그가 있습니다.

## 어떤 실험인가

번들의 r70 recipe대로 **R의 B4 세 모델만 새로 학습**합니다. 기준은 `results/model_results_20261011_1`의 완료된 lite20 S=0 모델입니다. 진행 중인 시드 탐색 모델은 사용하지 않습니다.

| R 계열 | 입력/정규화 | 해상도/샘플링 | 학습 |
|---|---|---|---|
| 322 | gray3 / imagenet | scale3 / area | train70, 21,000 iterations |
| 326 | zcdf3 / image_global | scale3 / area | train70, 21,000 iterations |
| 329 | zcdf3 / image_global | target width4000 / uniform | train70, 21,000 iterations |

모두 ImageNet encoder 초기화, seed0, Unet/EfficientNet-B4, crop1536, batch2, AdamW lr3e-4/wd1e-2, CE+Dice, 마지막 checkpoint입니다. 배치별 scheduler도 원래 recipe를 유지합니다. 프로젝트에서 검증한 worker/Albumentations 시드 고정과 터미널 진행률 조정을 적용했습니다.

`train_data/`에는 train70만 보입니다. `--no_valid_eval`을 고정하여 valid/test를 새 B4 학습 및 checkpoint 선택에 쓰지 않습니다. 21,000회는 번들에서 미리 정한 값이며 valid 점수로 바꾸지 않습니다. R의 나머지5개, E8개와 F1개는 그대로 유지합니다. 따라서 **기존 R11+E8+F1=20개 → r70 R8+E8+F1=17개**입니다. r70의 원래 manifest에서 검사를 생략하던 새 B4도 이 실행기는 학습 후 실제 SHA256을 생성해 엄격하게 검증합니다. `--allow-retrained`는 사용하지 않습니다.

학습은3개라 최대3개 GPU에서 동시에 진행합니다. 이후 **r70/lite20 × valid/test = 추론4개 작업**을 GPU4~7에 분산합니다. 두 구성 모두 원본 이미지부터 R·E·F를 다시 계산합니다. 14개 공통 가중치와 E 예측의 동일성을 확인한 뒤 다음을 비교합니다.

- valid20의 R 단독, R+E, R+E+F image mIoU
- 이미지별 변화와 시편 bootstrap 95% 차이 구간(20,000회, seed81795)
- test10 예측 마스크의 달라진 픽셀 수(정답/점수 사용 없음)

대표 결과는 요청한 r70입니다. 평가 뒤 더 높은 구성을 재선택하지 않습니다. 단일 시드의 개발 실험이며, 기존 lite20의 다른 규칙에는 valid 기반 선택이 남아 있어 독립 일반화 검증으로 해석하지 않습니다.

## 점검, 로그, 중단과 재개

```bash
./run.sh check  # 학습 없는 환경/소스/입력/recipe 검사
./run.sh plan   # 3개 학습, 전체9개 단계 확인
```

`./run.sh train`도 `--execute`가 없으면 점검만 합니다. 새로 실행할 때마다 소스 스냅샷과 새 출력 디렉터리를 만듭니다. 중복 실행은 잠금으로 막습니다.

```bash
# 시작 시 표시된 PID로 중단
kill -TERM <PID>

# RUN_DIRECTORY로 재개
./start_nohup.sh --resume /Data3/sppark/komap/experiments/model349_lite20_r70_retrain_gpu4567_v1/outputs/runs/r70_<실행시각>
```

완료된 단계는 SHA256 확인 후 건너뜁니다. 중단된 단계는 별도 `attempt_2`에서 처음부터 다시 실행하며, 이전 시도와 모델은 보존합니다. 중간 iteration의 optimizer 상태를 이어서 학습하는 기능은 아닙니다. 전체 로그는 `logs/nohup_latest.log`, 상세 로그는 `outputs/runs/<실행>/logs/`, 상태는 해당 실행의 `status.json`입니다. 모델이 있는 기존 실험이나 진행 중인 탐색 소스는 수정하지 않습니다.

## 결과 저장

완료 시 KST 날짜로 새 `results/model_results_YYYYMMDD_다음번호/`를 만듭니다.

- 동일 이름의 `.md` 결과서, `metrics.json`, `per_image.csv`, `paired_comparison.json`
- `valid_mask_comparison/`: r70 최종 valid20의 2×2 비교
- `r_only/`, `stage1/`: r70의 R 단독 및 R+E 결과/비교 이미지
- `baseline_lite20/`: 기준 모델의 같은 평가 결과
- `test_predictions/`: r70 test10 마스크
- `selected_model/`: r70 가중치17개, 추론 코드/설정/엄격한 검증기
- `source_snapshot/`, 출처·환경·실행 기록

패널은 좌상 원본, 우상 오류, 좌하 예측, 우하 정답입니다. 맞은 픽셀은 흰색, 오류는 정답 색이며 image mIoU와100%-image mIoU를 표시합니다. 지표는 이미지별4개 클래스 IoU 평균을20장 평균하며, union=0이면 그 클래스 IoU=0입니다. global mIoU는 계산하지 않습니다.

기준 실측값은 R **80.010135%**, R+E+F **80.255846%**입니다. r70 점수는 아직 없으며 학습 후 측정됩니다. 기존 기준20개 모델과 새17개 모델은 모두 실행 폴더에 보존합니다. 입력 가중치는 기존 완료 실험을 읽기만 하므로 그 경로를 이동/삭제하면 검사가 실패합니다.

## 출처

- 저장소: https://github.com/CNUKoMaP/lhs
- 커밋: `34c46604a699a9067605540295f647d0d5cca443`
- ZIP: `Model349_lite20_r70_bundle_code.zip`
- SHA256: `d65995562b6505d09e6429f8f535c2f11b74558bb14074753d39567c7ab0cb72`

`downloads/`에 원본 ZIP, `upstream/`에 원본 소스를 보존했습니다. 새 upstream의 변경은 R B4 full70 경로 지원과 r70 recipe/config입니다. 공통 학습/추론 소스가 직전 ZIP과 동일함을 비교한 뒤, 프로젝트의 기존 metric/진행률 조정을 유지했습니다. 차이는 `source_changes.diff`, 검증 내용은 `BUILD_REPORT.md`에 기록합니다.
