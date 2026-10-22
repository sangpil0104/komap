# Model349 lite12 FP16 — 시드 학습 100개

GitHub `CNUKoMaP/lhs`의 `Model349_lite12_base`를 기준으로 한 새 실험입니다.
원본 commit은 `f6456f54e5fc6517a89a3fdb679645a0520bbf61`입니다.
기존 모델·소스·결과는 보존하며, **사용자가 실행해야 학습이 시작됩니다.**

## 실행

```bash
cd /Data3/sppark/komap/experiments/model349_lite12_seed100_gpu4567_v1
./start_nohup.sh
```

Conda `komap`을 자동 활성화하고, 물리 GPU **4,5,6,7**을 UUID로 지정합니다.
GPU 하나당 작업 하나, 최대 4개 작업을 병렬 실행합니다. 작업이 끝나면 다음 작업을 배정합니다.
다른 작업이 GPU 메모리를 1 GiB 이상 사용 중이면 해당 GPU가 비기를 기다립니다.
기존 프로세스를 종료하지 않습니다.

실행 시 출력되는 `tail -f ...` 명령으로 터미널에서 진행률·경과 시간·ETA를 확인합니다.
학습/검증 tqdm과 작업별 로그가 함께 남습니다. 웹 서버는 사용하지 않습니다.
터미널에서 직접 실행하려면 `./run.sh train --execute`입니다.

**기본은 100개 완료까지 실행**합니다(`--budget-mode finish-all`).
기존 실행기의 `--hours 10` 값은 이 모드에서는 종료 제한이 아닙니다.
ETA는 초기 추정과 완료 작업 실측에서 갱신하며, GPU 대기·최종 조합 평가·복사 시간에 따라 달라집니다.

학습 없이 확인하려면:

```bash
./run.sh plan
./run.sh check
./run.sh train  # --execute 없이는 CPU 사전 점검만 수행
```

## 100개 학습 배분

이전과 동일하게 전체 모델을 100번 처음부터 학습하는 방식이 아니라,
최종 모델의 구성 가중치를 시드별로 학습하고 최종 조합으로 평가하는 방식입니다.
시드는 이전과 같은 범위 `3407–3506`입니다. 간소화판에 남은 구성으로 작업을 재배분했으므로
기존 23개판과 구성별 시드가 일대일로 대응하는 paired 비교는 아닙니다.

| 구성 | 학습 수 | 시드 | 시작점 |
|---|---:|---|---|
| Native269 | 16 | 3407–3422 | 기존 Model257 |
| HRNet336 | 12 | 3423–3434 | 동일 ImageNet encoder |
| Model109 | 16 | 3435–3450 | 기존 Model98 |
| restart51 | 16 | 3451–3466 | 기존 Model109 |
| restart59 | 8 | 3467–3474 | 기존 Model109 |
| restart61 | 8 | 3475–3482 | 기존 restart51 |
| B4 326 | 8 | 3483–3490 | fold 1·3 × 4세트, ImageNet encoder |
| B4 329 | 16 | 3491–3506 | fold 1·2·3·4 × 4세트, ImageNet encoder |
| 합계 | **100** | **3407–3506** | |

각 fine-tune은 기존 부모 가중치에서 시작합니다. 새 부모를 후속 후보의 시작점으로 바꾸지 않아
부모 변경과 학습 시드 변경을 섞지 않습니다. 학습 횟수·데이터·손실·checkpoint 선택 레시피는
기존 실행기의 설정을 복사했습니다. restart61의 기존 학습 내부 calibration 기반 epoch 선택은
유지하지만, 그 calibration을 최종 lite12의 고정 결합/후처리에 적용하지 않습니다.
불필요한 restart61 test 추론만 새 소스 사본에서 생략합니다.

B4는 필요한 fold가 모두 끝난 세트만 최종 후보로 평가합니다.
따라서 **100개 학습 → 단일 구성 교체 76개 + B4 세트 8개 = 후보 84개**입니다.
간소화판에서 제거된 Model123 EMA·Model347·Model322는 학습하지 않습니다.

## 기준 모델과 평가

GitHub 배포에는 가중치가 없습니다. 로컬의 보존된
`results/model349_inference_bundle_20261007_1/weights/`에서 필요한 12개만 읽어
이 실험의 `weights/`에 **FP16 저장 사본**을 만들었습니다. 원본은 수정하지 않았습니다.
이 가중치는 GitHub manifest의 파일과 다르며, 공식 기록 점수 **80.1255610%를 재현했다고 주장하지 않습니다**.
이전 시드 탐색에서 선택된 80.255658% 모델을 기준으로 사용한 것도 아닙니다.

사용자가 시작하면 먼저 이 로컬 기준 모델을 실제 valid20에서 추론하고 점수를 측정합니다.
모든 후보는 이 점수와 같은 환경에서 비교합니다. `upstream/`에는 GitHub 원본,
`frozen/`에는 동일 추론 코드·설정, `provenance.json`에는 commit과 해시가 있습니다.

평가 조건은 GitHub lite12와 동일합니다.

- 가중치 12개: 326 fold1·3, 329 전체 4-fold, HRNet336, Native269, 고해상도 4개.
- 저장 가중치 FP16, 로드 후 파라미터 FP32. 원본 코드의 autocast 호출은 그대로 유지합니다.
- TENT 5단계, OpenCV/torch thread 2, 고해상도 D4. B4/269/HRNet 실제 추론 TTA는 off입니다.
- 결합 계수, fold, calibration, 밝기 라우팅과 후처리를 고정합니다.
- 후보도 FP16 사본을 만든 다음 GitHub 추론 함수로 다시 추론합니다. 학습 중 점수를 최종 점수로 대신 쓰지 않습니다.
- 고해상도 후보는 해당 경로를 쓰는 검증 이미지만 재추론합니다. 나머지는 동일 기준 확률을 재사용합니다.
  restart51 15장 / Model109 2장 / restart59 2장 / restart61 1장입니다.

지표는 **이미지별 네 클래스 IoU 평균 → 검증 20장 평균**입니다.
정답·예측 모두 없는 클래스 IoU는 0이며 global mIoU는 계산하지 않습니다.
각 구성의 상위 2개와 기존 기준 가중치를 최대 2회 순차 조합 비교하고,
기준 모델과 최고 단일 교체 후보보다 낮은 점수를 선택하지 않습니다.
동일 validation으로 시드를 선택하므로 결과는 독립 test 성능이 아닙니다.

## 재개와 보존

```bash
./start_nohup.sh --resume /Data3/sppark/komap/experiments/model349_lite12_seed100_gpu4567_v1/outputs/runs/seed100_실행ID
```

완료 작업은 해시를 검사하고 건너뜁니다. 중단된 작업은 `attempt_2` 등 새 폴더에서
처음부터 다시 학습합니다. optimizer 상태부터 이어가는 재개는 아닙니다.
출력된 메인 PID에 `kill -TERM PID`를 보내면 이 실행기가 만든 자식 작업도 종료합니다.
같은 실행기를 중복 시작하면 lock으로 거부합니다.

선택적으로 `--budget-mode pause --hours 10 --reserve-hours 1.25`를 주면 예산 안에 완료할 것으로
예상되는 작업만 새로 시작하고, 완료 후보로 중간 결과를 저장합니다. 이 경우 100개가 다 끝나지
않을 수 있으며 위 명령으로 재개해야 합니다.

`outputs/runs/seed100_실행ID/`에는 소스 스냅샷, 계획, 실행 명령, 체크포인트 원본과 FP16 사본,
추론 확률, 후보별 보고서/비교 이미지, 완료 기록, GPU/진행 로그를 보존합니다.

최종 결과는 한국 날짜 기준 **`results/model_results_YYYYMMDD_다음번호/`**에 새로 생성합니다.

- 같은 이름의 `.md` 보고서, `leaderboard.csv`, `search_summary.json`.
- `valid_mask_comparison/`의 검증 20장: 좌상 원본 / 우상 오류 / 좌하 예측 / 우하 정답.
  맞은 픽셀은 흰색, 틀린 픽셀은 정답 색이며 상단에 image mIoU와 100%−image mIoU 표시.
- `candidates/`: 완료 후보 각각의 보고서와 20장 비교 이미지.
- `selected_model/`: 최종 12개 FP16 가중치와 추론 코드·설정.
- `source_snapshot/`: 실행 당시 학습·평가·실행기 코드.

빌드 단계에서는 학습·GPU 추론·TENT를 실행하지 않습니다. CPU 검증 내역은 `BUILD_REPORT.md`에 있습니다.
