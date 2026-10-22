# Model349 클래스 가중치 제거 실험

완료된 `model349_train_route_v2/outputs/runs/model349_train_20260930T143930_234475Z`의 소스 스냅샷을 복사한 독립 실험입니다. 이전 결과는 `results/model_results_0930_1`이며, 기존 소스·체크포인트·결과를 수정하거나 덮어쓰지 않습니다. 이전 학습 체크포인트를 재사용하지 않고 선행 모델부터 새로 학습합니다. ImageNet 사전학습 초기화는 기존과 같습니다.

## 변경 내용

- 모든 학습 단계의 semantic CE 클래스 가중치를 제거했습니다 (`weight=None`, 네 클래스에 동일한 가중치).
- Model62는 `--loss-mode standard`, HRNet Model347은 `--ce-class-weights` 없이 실행합니다. ResNet 계열의 코드 내부 가중치도 제거했습니다. 원래 가중치가 없던 Model6, B4, Model336의 CE는 그대로입니다.
- CE/Dice 비율, 클래스 평균 Dice, 경계 손실과 경계/비경계 BCE 보정(`pos_weight`), patch sampling, seed, batch size, epoch, fold, 추론 gamma 탐색은 유지합니다. 클래스 가중치 제거는 희소 클래스 patch sampling 제거를 뜻하지 않습니다.
- 물리 GPU **6, 7만** 사용합니다. UUID로 GPU를 지정하며 독립 모델·fold를 GPU당 하나씩, 최대 두 작업까지 동시에 실행합니다. 단일 모델을 DDP로 나누는 방식은 아닙니다. 의존성이 있는 단계는 선행 단계 완료 후 실행합니다.
- 총 59단계(50 GPU / 9 CPU)입니다. 내부 출력 폴더의 `weighted`, `inverse_ce` 등 이름은 기존 경로 호환성을 위해 남아 있지만, 이 실험의 실제 CE에는 클래스 가중치가 없습니다.

## 실행

학습 시작은 사용자가 합니다. 실행기는 자동으로 conda `komap`을 활성화합니다.

```bash
cd /Data3/sppark/komap
bash experiments/model349_no_class_weights_v1/start_nohup.sh
```

위 스크립트가 내부에서 `nohup ... run.sh train --execute`를 실행하고 PID와 로그 경로를 출력합니다. SSH 연결이 끊겨도 계속 진행합니다. 기본 GPU는 6,7로 고정되어 있으며 별도 `CUDA_VISIBLE_DEVICES` 설정은 필요하지 않습니다.

출력된 명령으로 진행 상황을 확인합니다.

```bash
tail -f /출력된/로그/경로
```

전체 단계 진행률과 각 학습의 `tqdm` 진행률·경과 시간·예상 남은 시간을 터미널 로그에서 확인할 수 있습니다. 단계별 로그는 실행별 `outputs/runs/.../logs/`에 저장됩니다. 전체 단계 ETA는 단계별 작업량 차이로 부정확할 수 있습니다.

학습 없는 점검:

```bash
bash experiments/model349_no_class_weights_v1/run.sh plan
bash experiments/model349_no_class_weights_v1/run.sh check
```

`run.sh train`도 `--execute`가 없으면 사전 점검 후 종료합니다. 같은 실험의 중복 학습은 lock으로 차단합니다. 학습 중단은 출력된 PID에 `kill -TERM PID`를 보내면 됩니다. 재실행은 새 경로에서 처음부터 시작하며 기존 실행을 덮어쓰지 않습니다.

## 지표와 결과

각 이미지에서 네 클래스 IoU를 동일하게 평균하고, valid 20장의 image mIoU를 평균합니다. 정답과 예측에 모두 없는 클래스는 IoU=0이며 global mIoU는 계산하지 않습니다. valid는 체크포인트·조합 선택에도 사용됩니다.

전체 파이프라인 완료 후 `results/model_results_MMDD_번호/`에 Markdown 보고서, 지표 JSON, 예측 mask, `valid_mask_comparison/` 20장을 저장합니다. 2×2 배치는 좌상 Original / 우상 Error / 좌하 Prediction / 우하 Ground truth입니다. Error는 맞은 픽셀을 흰색, 틀린 픽셀을 정답 색으로 표시하며 image mIoU와 100%−image mIoU를 함께 표기합니다. 폴더 날짜는 UTC 기준입니다.

실행별 소스·명령·환경·데이터 해시·`ablation.json`을 함께 보존합니다. 변경 내역은 `reports/local_patches.diff`, 검증 결과는 `reports/*validation.json`에 있습니다.
