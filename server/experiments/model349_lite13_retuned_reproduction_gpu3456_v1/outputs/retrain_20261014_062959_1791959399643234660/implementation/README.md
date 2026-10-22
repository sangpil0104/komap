# Model349 lite13 retuned 전체 재학습 — GPU 3~6

GitHub `CNUKoMaP/lhs/Model349_lite13_retuned_reproduction_code.zip`을 별도 실험으로 준비했습니다.
**물리 GPU 3·4·5·6 / conda komap / nohup**을 사용합니다. 학습 시작은 사용자가 합니다.

```bash
cd /Data3/sppark/komap/experiments/model349_lite13_retuned_reproduction_gpu3456_v1
./start_nohup.sh
```

실시간 전체 로그:

```bash
tail -f /Data3/sppark/komap/experiments/model349_lite13_retuned_reproduction_gpu3456_v1/logs/latest.log
```

실행 전 검사와 학습 계획:

```bash
./run.sh check
./run.sh plan
```

`./run.sh train`은 점검만 합니다. 직접 포그라운드에서 시작하려면 `./run.sh train --execute`입니다.
시작 시 GPU 점유를 확인하고, 요청 GPU에 다른 작업이 있으면 학습을 시작하지 않습니다.
GPU 번호는 실행 시 UUID로 변환하여 지정합니다. 이 실행기의 중복 실행은 lock으로 막습니다.

전체 로그에는 단계 진행률, 경과 시간, 추정 ETA, 진행 중인 작업의 최신 로그가 나옵니다.
작업별 상세 로그는 `outputs/retrain_실행ID/logs/작업명.log`입니다.
`latest_run.json`에 현재/최근 실행 폴더가 기록됩니다. 학습기에도 tqdm 진행률을 추가했습니다.
ETA는 완료 단계 속도에 따른 추정치이며 서로 다른 학습 단계의 시간을 정확히 예측하는 값은 아닙니다.

중단할 때는 `logs/latest.pid`의 PID에 TERM을 보냅니다. 실행기가 자식 프로세스도 종료합니다.

```bash
kill -TERM "$(cat logs/latest.pid)"
```

매번 실행할 때 새 출력 폴더와 소스 스냅샷을 만듭니다. 중단된 출력은 보존됩니다.
이 실행기는 중단 지점 자동 재개를 제공하지 않으며 다시 시작하면 새 실험을 처음부터 학습합니다.
학습 중 저장된 `best.pt`만 보고 완료로 간주하지 않고, 프로세스 성공 종료와 산출물을 확인해 완료 기록을 남깁니다.

## 구성

최종 배포 가중치는 **R7 + E5 + F1 = 13개**입니다. 선행 모델과 F 학습용 OOF 모델을 포함하면 학습 작업은 **33개**입니다.

| 묶음 | 학습 작업 수 | 내용 |
|---|---:|---|
| ResNet34 | 13 | model6 → 62 → 97 → 98 → 109, restart51 → 61, 194 → 195 → 239 → 242 → 257 → 269 |
| HRNet336/347/N3 | 3 | 336 기본 학습, 347 추가 학습, 414 N3 입력 추가 학습 |
| B4-329 및 warmstart | 6 | 329 fold1~4, BN 통계 고정 추가 학습 fold1·3 |
| B4-326 | 1 | fold2, seed3456 |
| HRNet392 OOF | 8 | seed1·2 각각 fold1~4 |
| HRNet392 train70 | 1 | seed5, 70장 전체 학습 |
| F | 1 | train70의 OOF 특징으로 8 epochs / 1,120 updates |
| 합계 | **33** | 추론·데이터 준비·채점을 포함한 전체 51단계 |

기존에 학습한 segmentation 가중치를 시작점으로 재사용하지 않습니다.
ImageNet 초기 가중치에서 출발해 선행 모델도 이번 실행 안에서 학습합니다.
원본의 최종 13개 구성, 시드, 학습량, fold, 추론 정밀도, TTA/TENT, 결합 비율을 유지합니다.

- R: restart61, model269, B4-326 fold2, B4-329 fold1·3·4, HRNet347.
- E: B4 warmstart fold1·3, HRNet414 N3, HRNet392 seed1 fold4, HRNet392 train70 seed5.
- F: 새 train70 OOF로 학습한 보정기.
- retuned R: `al_gamma=1.0`, `al3ni_gamma=1.0`, `eut_edge_blend=0.125`.
- retuned E: B4 / HR 묶음1 / HR 묶음2 = `0.4 / 0.42 / 0.18`.

GPU마다 독립 학습 레인을 순서대로 처리합니다. 모든 학습이 같은 시각에 끝나지는 않으므로 마지막 구간에는 일부 GPU가 먼저 쉬게 됩니다.
F 학습은 모든 OOF 생성 이후 GPU 3에서 진행합니다.

## 지표와 데이터

기존 실험과 동일한 train70 / valid20 / test10을 새 실험의 `data/`에 복사했습니다.
190개 데이터 파일의 SHA-256을 `data_manifest.json`에 보존합니다.
OOF는 train70의 59개 시편 그룹을 고정 4-fold로 나누며, 각 held-out 크기는 18·18·17·17장입니다.
valid/test는 OOF 학습에 들어가지 않습니다.

프로젝트 지침대로 **이미지별 네 클래스 IoU 평균 → valid20 이미지 평균**을 사용합니다.
정답과 예측에 모두 없는 클래스의 IoU는 0이며 global mIoU는 계산하지 않습니다.
일부 과거 R 학습기에 있던 global/빈 클래스 제외 checkpoint 선택도 이 기준으로 통일했습니다.
따라서 원본과 선택 checkpoint가 달라질 수 있으며 이를 결과 보고서에 명시합니다.
나머지 모델은 원본 레시피대로 마지막 가중치를 사용합니다.

공개 **80.779378%**는 원래 가중치의 참조 점수입니다. 이번 재학습 결과나 소요시간을 보장하는 값이 아닙니다.
원본 결합 비율과 일부 R checkpoint 선택에 valid20이 사용되므로 독립 test 성능으로 해석하지 않습니다.

## 추론시간 기록

학습 후 **원본 이미지부터 새로 valid20, test10 추론**을 실행하면서 각각 측정합니다.

- 전체 실제 경과시간(초), 장당 평균시간(초/장), 처리량(장/초).
- R·E·F 단계별 경과시간과 실제 GPU UUID·모델명.
- R은 GPU 3~6에 구성요소를 분배하며, E는 GPU 3, F는 CPU에서 실행합니다.
- 코드/가중치 링크 준비, Python 시작, 모델 로딩, 이미지 입출력·전처리, R의 TTA/TENT, E/F와 후처리, 확률·마스크 저장, 임시 폴더 정리를 포함합니다.
- 학습, 정답 채점, 결과 비교 그림 생성은 추론시간에서 제외합니다.
- `time.perf_counter()`로 하위 추론 프로세스가 종료될 때까지 측정합니다. GPU 출력의 CPU 변환·마스크 저장이 완료된 후이므로 비동기 CUDA 실행 완료도 포함합니다.
- 명시적 warm-up이나 이전 예측 재사용은 하지 않으며 split마다 한 번 측정합니다. 장당 값은 전체 시간/이미지 수이며 단일 이미지 지연시간이 아닙니다.

결과를 보고할 때는 정확도와 함께 이 실측 시간을 사용합니다. 준비 단계에서 추론시간을 추정하여 결과로 적지 않습니다.

## 완료 시 결과

한국 날짜 기준 새 `results/model_results_YYYYMMDD_번호/`에 자동 저장합니다.

- `model_results_YYYYMMDD_번호.md`: valid20 image mIoU, 추론시간 표, 구성과 재현 범위.
- `metrics.json`, `per_image.csv`: 독립 채점한 이미지별 지표.
- `inference_timing.json`, `inference_timing.csv`: valid/test별 실측 추론시간과 측정 범위.
- `valid_mask_comparison/`: valid20의 2×2 비교 그림. 좌상 원본 / 우상 오류(정답색, 맞은 픽셀 흰색) / 좌하 예측 / 우하 정답. image mIoU와 100%−image mIoU를 표시합니다.
- `selected_model/`: 새 가중치 13개, 코드, 설정, SHA-256 manifest, `predict.sh`.
- `test_predictions/`: test10 예측. 정답이 없으므로 test 점수는 계산하지 않습니다.
- `source_snapshot/`, `stage_receipts/`, 환경 및 실행 기록.

## 출처·검증

- 저장소: https://github.com/CNUKoMaP/lhs
- 고정 커밋: `b6fa9f84eb6415a1a4258b717467350355237e6c`
- ZIP SHA-256: `23a7a085eb075fb4303fb5505dc99c1003abe5804af635b20a292f857184fb69`
- `downloads/`: GitHub 원본 ZIP과 제공된 해시.
- `upstream/`: 수정하지 않은 원본 코드.
- `source/`: 이 실험에서만 사용하는 실행 코드. 변경 기록은 `source_changes.diff`입니다.
- `BUILD_REPORT.md`, `checks/`: CPU 검사·데이터 검사·보고서 생성 검증 기록.

기존 실험, 모델, 결과와 이전 `gpu4567` 다운로드 폴더는 수정하지 않았습니다.
