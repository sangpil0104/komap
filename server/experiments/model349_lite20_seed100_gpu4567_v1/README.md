# Model349 lite20 B4ws — 100개 학습 시드 탐색

현재 `model_results_20261011_1`의 lite20 모델을 기준으로 탐색합니다. 기준 valid20 image mIoU는 **80.25584552762022%**입니다. 기존 모델/소스/결과는 수정하지 않습니다.

## 시작과 로그

학습 시작은 사용자가 합니다. 아래 실행기는 conda `komap`을 활성화하고 GPU **물리 4,5,6,7**만 사용하여 `nohup`으로 실행합니다.

```bash
cd /Data3/sppark/komap/experiments/model349_lite20_seed100_gpu4567_v1
./start_nohup.sh
tail -F logs/nohup_latest.log
```

실행 전 검사만 하려면 `./run.sh check`, 배분만 보려면 `./run.sh plan`입니다. `train --execute`가 있어야 학습을 시작합니다. `check`는 GPU에 모델을 올리거나 학습하지 않습니다.

전체 로그에는 학습 완료 수, 경과 시간, 예상 남은 시간과 각 GPU 작업의 최근 로그가 20초마다 나옵니다. 각 학습의 tqdm도 `outputs/runs/<실행>/logs/`에 기록됩니다. GUI/웹 서버는 사용하지 않습니다.

```bash
# 시작 시 출력된 PID로 중단: 이 탐색이 시작한 자식 프로세스만 정리
kill -TERM <PID>

# 출력된 RUN_DIRECTORY를 사용하여 동일 탐색 재개
./start_nohup.sh --resume /Data3/sppark/komap/experiments/model349_lite20_seed100_gpu4567_v1/outputs/runs/seed100_<실행시각>
```

완료 작업은 파일 SHA256을 검증한 뒤 건너뜁니다. 중간에 중단된 작업은 `attempt_2` 등 새 디렉터리에서 해당 작업만 처음부터 다시 시작합니다. 완료 체크포인트와 이전 시도는 보존합니다. 동일 실험의 동시 실행은 잠금으로 막습니다. 다른 작업이 쓰는 GPU는 비워질 때까지 기다립니다. 시간 제한 없이 100개를 모두 마칩니다.

## 100개 학습 배분

여기서 100개는 전체 20개 모델 묶음을 100번 학습한다는 뜻이 아니라 **구성 모델 학습 총 100회**, 서로 다른 seed **3407~3506**입니다. fold 묶음을 합치면 R/E 후보 64개와 F 후보 8개를 평가합니다. 조합 재평가는 별도이며 추가 학습 횟수에 포함되지 않습니다.

| 구성 | 학습 수 | 조건 |
|---|---:|---|
| R HRNet336 | 12 | ImageNet 초기화, 70 epochs |
| R HRNet347 | 12 | 기존336에서4개 + 상위 새336 두 부모에서 각4개 |
| R Native269 | 8 | 현재 Model257 고정 |
| R restart51 | 12 | 현재 Model109 고정 |
| R restart61 | 4 | 현재 restart51 고정 |
| R B4-326 | 8 | fold1,2 × 4개 묶음 |
| R B4-322 | 4 | fold3 × 4개 |
| E B4 BN-frozen warmstart | 24 | 기존 pre-warmstart B4, 4 folds × 6개 묶음, 2000 iterations |
| E HRNet392 | 8 | 활성 네 슬롯 × 2개 묶음, ImageNet 초기화 |
| F residual refiner | 8 | R/E 선택 이후 OOF feature로 학습 |
| 합계 | **100** | R/E 92 → F 8 |

데이터/fold/epoch/iteration/optimizer/loss는 기존 lite20 recipe를 고정하고 학습 시드만 바꿉니다. Model347의 부모 선택과 F의 선택 HR OOF 갱신이 명시적인 의존 관계입니다. 347 자식은 해당 336 부모와 함께 평가하고 조합 선택에서도 둘을 묶어 유지합니다. 기본336에서 학습한347 후보는 기본336을 유지하는 조합으로 평가합니다.

각 후보는 **전체 최종 R+E+F의 valid20 image mIoU**로 평가합니다. 각 그룹 상위2개/기존 모델을 포함하여 최대2회 조합 개선을 수행하고, 기준 및 모든 단일 교체 후보 중 최고값을 하한으로 보존합니다. R/E 선택 뒤 변경된 활성 HR 슬롯의 train OOF만 다시 예측합니다. 다른 HR OOF와 **warmstart 이전 B4 OOF**는 고정합니다. train70에 대해 fold 및 시편 그룹 분리를 확인한 뒤 F feature를 만들고 F를 8개 학습합니다. 마지막으로 최상위 F를 사용한 조합을 한 차례 재확인하며 이때 E HR은 고정합니다. 최종 결과가 기준 또는 이미 측정한 최고 후보보다 낮으면 그 후보를 유지합니다.

## 지표와 결과

각 이미지에서 4개 클래스 IoU를 평균하고, 그 값을 valid20에서 평균합니다. union=0 클래스의 IoU는 **0**입니다. global mIoU는 계산하지 않습니다. seed 선택에 valid20을 반복 사용하므로 결과는 독립적인 일반화 성능이 아닌 개발/선택 결과입니다. test10은 최종 모델이 정해진 뒤에만 추론하고 점수를 구하지 않습니다.

최종 패키지는 **20개 체크포인트(R11+E8+F1)**이며, 원래 lite20의 subset, gamma, 밝기 분기, TTA 실행 코드, FP32 weights와 FP16 family 확률 저장을 유지합니다. 새 패키지의 원본 이미지부터 시작하는 valid 추론이 탐색 예측과 픽셀 단위로 일치해야 결과를 확정합니다.

완료 시 다음을 새 `results/model_results_날짜_번호/`에 저장합니다(날짜는 기존 프로젝트와 같이 KST).

- 동일 이름의 Markdown 결과서, `metrics.json`, `leaderboard.csv/json`
- `valid_mask_comparison/`: 좌상 원본, 우상 오류(정답색/맞은 위치 흰색), 좌하 예측, 우하 정답; image mIoU와 100%-image mIoU 표시
- `selected_model/`: 새 SHA256 manifest와 검증기/추론 코드를 포함한 최종 20개 모델
- `test_predictions/`, `source_snapshot/`, 실행 plan/환경 기록

후보 전체의 체크포인트/명령/로그/검증 결과는 `outputs/runs/<실행>/`에 보존합니다. 초기 계획은 `plan.json`, 원래 학습 인자는 `original_recipe.json`입니다. baseline은 원래 실험을 읽기만 하므로 이전 실험 경로를 이동/삭제하면 시작/재개 검사가 실패합니다. 신규 소스/데이터/fold 구성은 이 디렉터리에 따로 복사되어 있습니다.

GPU 4개에서 학습 계산만 약7시간 규모이며, 후보 추론·CPU 조합 평가·최종 검증이 추가됩니다. 정확한 소요 시간은 실행 중 관측값으로 보정합니다. 충분한 디스크 공간을 확인하고 시작하며, 모든 시도와 모델을 보존하므로 수십~수백 GB가 추가될 수 있습니다.
