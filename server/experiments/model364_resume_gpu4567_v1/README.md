# Model364 중단 지점 재개 — GPU 4,5,6,7

```bash
cd /Data3/sppark/komap/experiments/model364_resume_gpu4567_v1
./start_nohup.sh
```

conda `komap`에서 nohup으로 실행하며 PID, 로그 경로, `tail -f` 명령을 출력합니다. GPU 4,5,6,7만 UUID로 배정하고 독립 작업 최대 4개를 실행합니다. 진행률, 경과 시간, ETA를 터미널 로그에 제공합니다. 학습은 사용자가 시작합니다.

실패 실행 `model364_train_20261006T075724_689269Z`의 완료된 26단계를 재사용하고 남은 31단계만 실행합니다. Model336/347/361 등 완료된 학습을 다시 하지 않습니다. Model361 검증을 가장 먼저 배정합니다.

중단된 Model257 학습과 Model322 fold3 학습은 optimizer/iteration 재개 상태가 없습니다. Model257은 완료된 Model242 checkpoint에서 해당 4-epoch 단계만 다시 시작하고, Model322 fold3은 해당 16000-iteration fold만 다시 시작합니다. 완료된 fold1/2는 재사용합니다. 중단된 Model347 검증과 실패한 Model361 검증은 첫 이미지부터 다시 실행합니다.

Model361 검증의 native overlap을 170에서 **114**로 바꿨습니다. scale3에서 native tile171, stride57이므로 정상 타일링이 가능합니다. 원래 scale2의 overlap 비율(170/256)을 비슷하게 유지합니다. 학습 코드·학습 가중치·gate 설정은 그대로입니다. 이 추론 overlap은 번들 원본 인자와 다른 로컬 수정이며 점수에 영향을 줄 수 있습니다.

실행 시 기존 outputs 약 12GB를 새 실행 경로에 독립 복사합니다. 기존 체크포인트와 소스·로그·결과를 덮어쓰지 않습니다. 복사 진행률을 표시하며, 새 복사본의 미완료 출력만 정리합니다. data와 소스도 복사하고 CV 경로를 새로 생성합니다. 새 결과는 `outputs/runs/model364_resume_.../`에 보존하고 완료 시 프로젝트 `results/model_results_MMDD_다음번호/`에 보고서 및 비교 이미지20장을 만듭니다.

학습 없이 경로/계획을 검사하려면 `./run.sh check`를 실행합니다. `./run.sh train`도 `--execute` 없이는 검사만 합니다. 실행 중 중단은 출력된 PID에 `kill PID`를 보냅니다. 이 실행기를 다시 실행하면 원래 중단 지점에서 새 복사본으로 재개합니다. 새 재개 실행 자체의 중간 지점을 자동 선택하지 않습니다.

## 완료한 검사

- `reports/preflight.log`: 완료 작업26개 및 필수 출력 존재, 데이터 수, source SHA256, GPU UUID, 잔여31단계/의존 그래프와 native 타일 조건 검사.
- `reports/scale3_validation.json`: 기존 Model361 `last.pt`와 실제 첫 valid 이미지 A117_step4_sample2(875×1230)를 GPU4에서 scale3/overlap114/batch2/flip4로 추론해 shape, 유한 확률, 확률 합 검사 통과. 학습/backward 없음.
- `reports/scheduler_validation.json`: 합성 subprocess31단계 GPU4–7 독점 배정, 의존 순서, 실패 시 자식 종료, 필수 출력 검사 통과.
- `reports/preservation_validation.txt`: 독립 복사본 수정이 원본 bytes에 영향을 주지 않는지, 새 복사본의 미완료 디렉터리만 제거하는지 검사 통과. Python 및 shell 구문 검사 통과.

추론 overlap 수정은 `pipeline_tasks.json`과 각 실행의 provenance 및 결과 보고서에 기록됩니다. 이번 준비 과정에서는 학습을 시작하지 않았습니다.
