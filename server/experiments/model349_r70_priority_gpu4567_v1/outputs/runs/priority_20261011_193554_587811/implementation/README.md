# Model349 우선순위 통합 실행 — GPU 4,5,6,7

사용자가 아래 명령을 실행하면 네 단계를 순서대로 처리합니다. 빌드·check 명령은 실행 중인 학습을 중단하거나 새 학습을 시작하지 않습니다.

```bash
cd /Data3/sppark/komap
bash experiments/model349_r70_priority_gpu4567_v1/start_nohup.sh
```

1. `model349_lite20_seed100_gpu4567_v1`의 현재 run에서 새 작업 배정을 멈춥니다. 이미 실행 중인 최대 4개 작업은 학습·체크포인트 저장·평가를 끝내도록 기다립니다. 그 뒤 기존 실행기를 정상 종료하고 완료 기록과 체크포인트 SHA256을 보존합니다. 따라서 중단까지는 현재 작업이 끝나는 시간이 추가로 필요합니다.
2. `model349_lite20_r70_retrain_gpu4567_v1`을 실행합니다. B4 322/326/329를 train70 전체, seed0, 각 21,000 iterations로 학습하고, 기존 r70 실행기의 valid/test 추론 및 결과 정리를 완료합니다.
3. 앞서 확인한 HR336 시드 **3407~3414 (`336_00`~`336_07`)** 각각에 새 B4 3개를 적용합니다. R의 B4 fold 가중치 6개를 full70 3개로 교체합니다. **E의 B4 fold 모델은 유지합니다.** 각 조합은 R8+E8+F1=17개이며, r70 기준에서 HR336만 해당 시드 가중치로 바꿉니다. 원본 valid20부터 8개 조합을 GPU 4~7에 4개씩 분배해 새로 추론하고 R/R+E/R+E+F를 평가합니다. 시드별 B4 재학습은 없습니다.
4. 원래 seed100 run을 `--resume`으로 실행해 100/100까지 완료하고 기존 실행기의 최종 결과 정리까지 기다립니다. **원래 seed100 계획·fold 기준 모델은 유지합니다.** 3단계의 r70 비교는 별도 실험입니다.

## 로그와 상태

```bash
tail -f /Data3/sppark/komap/experiments/model349_r70_priority_gpu4567_v1/logs/nohup_latest.log
bash /Data3/sppark/komap/experiments/model349_r70_priority_gpu4567_v1/run.sh status
```

통합 로그에는 단계, 완료 수, 경과 시간, 예상 남은 시간과 개별 로그 경로가 표시됩니다. 개별 학습 로그에는 기존 tqdm 진행률을 유지합니다. GPU 대기 시간은 작업 ETA에 포함되지 않습니다. localhost 서버나 웹 대시보드는 사용하지 않습니다.

`outputs/latest_run.json`의 run 아래에서 `state.json`, `logs/`, `pause_complete.json`, `r70_complete.json`, `evaluated/`, `evaluation_export.json`, `seed_resume_complete.json`을 확인할 수 있습니다. 전체 완료 후 `results_index.json`과 `summary.md`가 세 결과서 위치를 모읍니다.

## 결과 저장

- r70 학습 결과: 기존 r70 실행기가 `results/model_results_날짜_번호/`를 새로 생성합니다.
- 8개 비교 결과: 별도 `results/model_results_날짜_번호/`에 동명 MD, 비교 표, 전체 지표 JSON, 최고 R+E+F 조합의 `selected_model/`을 저장합니다. R 단독 최고 시드도 별도로 기록합니다.
- `valid_mask_comparison/`: R+E+F 최고 시드의 valid20 비교 그림. `seeds/seed_번호/{R,stage1,stage2}/valid_mask_comparison/`: 8개 전체의 단계별 비교 그림. 총 500장(480장+대표 20장)입니다.
- 각 2×2 그림은 좌상 원본/우상 정답색 오류(정답 픽셀 흰색)/좌하 예측/우하 정답이며, image mIoU와 100%-image mIoU를 표시합니다.
- 이미지별 네 클래스 IoU를 먼저 평균한 뒤 valid20 평균을 사용합니다. union=0은 IoU=0이며 global mIoU는 계산하지 않습니다.
- 기존 8개 fold R 성적은 SHA256으로 고정한 기존 R 평가 JSON, R+E+F 성적은 해당 시드의 완료 기록을 사용합니다. 새 결과와 %p 차이를 계산합니다. valid에서 비교한 순위이며 독립 test 점수는 아닙니다.

## 중단·재개

동일한 `start_nohup.sh` 명령은 이 통합 실행기의 최근 미완료 run을 이어갑니다. 중복 실행은 파일 잠금으로 차단합니다. 완료된 단계를 재학습하지 않으며, 완료 기록의 파일 해시를 검증합니다. 이미 전체 완료된 run은 다시 실행하지 않습니다.

실행 중 통합 로그 시작 부분에 출력된 PID에 `kill -TERM PID`를 보내면 현재 하위 작업을 정리하고 중단합니다. 강제 `kill -9`는 피하세요. 기존 실행기의 resume은 **완료된 작업 단위**입니다. 비정상 중단된 미완료 작업의 체크포인트와 로그는 보존하지만, 해당 작업은 새 attempt에서 처음부터 다시 실행합니다. optimizer의 중간 iteration을 이어가는 방식은 아닙니다.

저장 대상 원래 run은 다음으로 고정했습니다. 다른 탐색 run을 찾아 임의로 중단하지 않습니다.

```text
/Data3/sppark/komap/experiments/model349_lite20_seed100_gpu4567_v1/outputs/runs/seed100_20261011_170427_031422
```

1단계는 기존 실행기만 SIGSTOP으로 잠시 멈춰 새 작업 배정을 차단합니다. 워커는 계속 실행해 결과를 저장합니다. 워커가 끝나면 기존 실행기에 SIGTERM/SIGCONT를 보내 정상 정리합니다. 완료 체크포인트는 원래 위치에 두고 경로·해시·완료 기록을 별도 보존합니다. 이전 소스·가중치·결과를 덮어쓰지 않습니다.

통합 실행 중에는 별도 r70/seed100 시작 명령을 함께 실행하지 마세요. 다른 GPU 작업을 임의 종료하지 않으며, 사용할 GPU가 비어야 작업을 시작합니다. 이 실행기는 conda `komap`만 사용하고 물리 GPU 4~7을 UUID로 지정합니다.

## 학습 없이 사전 점검

```bash
bash experiments/model349_r70_priority_gpu4567_v1/run.sh check
bash experiments/model349_r70_priority_gpu4567_v1/run.sh plan
```

`checks/` 아래 검증용 패키지·보고서는 기존 가중치/예측을 사용한 CPU fixture입니다. 새 r70 학습 결과가 아니며 실제 결과 폴더와 분리됩니다.
