# Model349 — GPU 8개 병렬 실행 v2

**기본 GPU pool: 물리 GPU 0,1,2,3,4,5,6,7. 학습은 사용자가 시작합니다.**

```bash
cd /Data3/sppark/komap/experiments/model349_train_route_v2
./start_nohup.sh
```

`conda komap`에서 실행하며 PID와 `tail -f` 로그 명령을 출력합니다. 터미널 직접 실행은 `./run.sh train --execute`입니다.

학습 없이 확인:

```bash
./run.sh plan     # GPU 배치와 작업 수만 표시
./run.sh check    # 환경·데이터·소스 무결성·4개 fold 점검
./run.sh train    # 실행 전 점검만; --execute가 있어야 실제 학습
```

GPU를 일부만 쓰려면 `MODEL349_GPUS=0,1,2,3 ./start_nohup.sh` 또는 `./start_nohup.sh --gpus 0,1,2,3`을 사용합니다. CPU 조합 탐색 worker는 `KOMAP_WORKERS=4` 또는 `--workers 4`로 지정합니다.

## GPU 배치

각 모델·fold에 GPU 한 개씩 배정하고, 독립 작업을 최대 8개 동시에 실행합니다. 최초 배치는 다음과 같습니다.

| GPU | 첫 작업 |
|---|---|
| 0 | ResNet Model6, 이후 의존 관계를 따라 후속 모델 |
| 1 | HRNet Model336 |
| 2–5 | B4 Model322 fold 1–4 |
| 6–7 | B4 Model326 fold 1–2 |

작업이 끝난 GPU에는 준비된 후속 모델·OOF 추론·남은 B4 fold(Model326/329)를 자동 배정합니다. GPU마다 `CUDA_VISIBLE_DEVICES`를 해당 물리 GPU의 UUID로 제한합니다. 한 GPU에 이 실행기의 GPU 작업 두 개를 동시에 배정하지 않습니다.

모델의 batch size, learning rate, seed, fold 분할, iteration/epoch 수는 v1과 동일합니다. B4는 각 fold batch 2 × 16,000 step을 유지합니다. 독립 모델을 병렬 실행하므로 모델 자체를 DDP로 변경하지 않습니다. 데이터 증강의 원본 시간 기반 worker seed 때문에 수치 결과는 실행마다 달라질 수 있습니다.

총 59단계 중 50개가 GPU 작업, 9개가 CPU 작업입니다. B4 12개 fold, ResNet 계보와 HRNet을 의존 관계에 따라 실행합니다. 부모 체크포인트가 필요한 작업은 부모의 정상 종료·출력 확인 후 시작합니다. 최종 조합은 필요한 모든 확률이 준비된 뒤 수행합니다.

**의존 작업만 남은 구간이나 마지막 CPU 조합 탐색에서는 GPU 8개가 항상 동시에 바쁘지는 않습니다.**

## 보존·진행 확인

기존 `model349_train_route_v1` 전체를 변경하지 않았습니다. v2의 `source/`, `upstream/`, ZIP, 데이터는 별도 사본입니다. 모델 구현·지표·결과 저장 코드는 v1과 바이트 단위로 같습니다. 이번 변경은 실행 순서와 GPU 배정입니다.

매 학습은 `outputs/runs/model349_train_UTC시간/`에 새로 저장합니다.

- `source_snapshot/`, `launcher_snapshot/`: 실제 모델 코드와 병렬 실행기 스냅샷.
- `logs/작업이름.log`: 각 모델·fold의 tqdm, 경과 시간, ETA.
- 메인 로그: 전체 완료 단계 수, 현재 GPU별 작업, 15초마다 각 작업의 마지막 진행 정보.
- `events.jsonl`: 시작·완료 시각, 실제 GPU UUID와 명령.
- `status.json`: 현재 상태와 작업별 경과 시간.
- `outputs/`: 이번 실행의 체크포인트와 확률 캐시.

전체 단계 ETA는 단계별 시간 차이가 커 대략적인 값이며, 각 학습 로그의 tqdm에서 해당 작업의 ETA를 확인할 수 있습니다. localhost 서버는 사용하지 않습니다.

한 작업이 실패하거나 실행기를 중단하면 해당 실행기가 띄운 나머지 작업 및 자식 프로세스를 종료하고 로그·체크포인트를 보존합니다. 실행기 중복 실행은 파일 잠금으로 차단합니다. 자동 중간 재개 기능은 없습니다.

## 지표·결과

v1과 동일하게 이미지별 4개 클래스 IoU 평균 후 valid 20장 평균을 계산하며, 빈 클래스 IoU는 0입니다. 실행 경로에서 global mIoU는 계산하지 않습니다. 원본 중간 모델의 일부 선택 정책을 image mIoU로 통일한 v1 변경 사항을 그대로 사용합니다.

학습·조합 완료 후 `results/model_results_MMDD_다음번호/`에 Markdown 결과 보고서, `metrics.json`, 예측 마스크와 `valid_mask_comparison/` 20장을 저장합니다. 비교 이미지는 좌상 Original / 우상 오류 / 좌하 Prediction / 우하 Ground truth이며, 맞은 픽셀은 흰색, 틀린 픽셀은 정답 색입니다. 위에 image mIoU와 `100%−image mIoU`를 표시합니다.

세부 원본 재현 설명은 [보존된 v1 설명](../model349_train_route_v1/README.md)을 참조하세요. v2 검증 기록은 `reports/`에 저장합니다. `inherited_v1/`는 기존 지표·forward·조합 검증 결과이며, 병렬 실행기는 별도의 합성 subprocess 검사로 확인합니다. 실제 학습은 시작하지 않습니다.
