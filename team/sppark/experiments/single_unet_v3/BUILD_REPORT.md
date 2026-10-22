# single_unet_v3 — 6 GPU 실행 인자 오류 수정

v2 통합 로그에서 `run.py: error: ambiguous option: --run could match --run-path, --run_path`를 확인했다. torchrun이 worker의 `--run`을 자신의 축약 옵션으로 해석해 worker 생성 전에 종료됐다. 새 v3에서 torchrun 옵션 뒤에 `--`를 추가했다.

- 이전 실패 로그: `../single_unet_v2/outputs/logs/suite_20260924T053536_GpL2HW.log` (보존).
- 대상 GPU: **0,1,2,5,6,7**.
- 재개 run: `/Data3/sppark/komap/experiments/single_unet_v3/outputs/runs/S0_x1_seed2026_migrate1to6_20260924T053853_286049Z`.
- 체크포인트: S0 **2,000 / 8,000 updates**, 남은 S0 6,000 → S1 8,000 → S2 8,000.
- checkpoint SHA256: `efc387d8fa787219effa1c01532fe4ee43c48c967bc9932f2d3c83fe644acc6e`.
- v1 소스 13개, v2 소스 16개 및 각 run의 snapshot, 원본/v2/v3 체크포인트 동일 hash 확인.
- 모델·loss·배치·학습 일정은 v2와 동일하다. v3의 소스 차이는 실행 인자 구분, nohup에서 tqdm 출력 끄기, 보고서 버전명이다.
- 통합 로그 및 60초 주기 전체 ETA/KST·UTC 완료 시각 예측을 유지한다. 초기 약 6시간 48분은 미측정 근사이며 실행 후 실측 보정된다.

## 실행

```bash
cd /Data3/sppark/komap
bash experiments/single_unet_v3/start_nohup_6gpu.sh
tail -F experiments/single_unet_v3/outputs/logs/latest.log
```

## 검증

- pytest **35개 통과**. 보고서 제목 수정 후 해당 보고서 생성 검사도 별도로 통과.
- 실제 launch 함수가 만드는 명령을 검증하고, 실제 torchrun 6개 worker에서 `_worker --run RUN --resume --help` 정상 종료 확인. `--help`는 학습 함수 진입 전에 종료된다.
- torchrun 옵션 구분자가 없을 때 원래 오류가 재현되고, 추가하면 worker 옵션이 정확하게 보존됨을 확인.
- 원본 데이터 100개 해시/분할 및 checkpoint 모델 상태 strict load 검사 통과.
- Bash 문법 검사, plan-only 예상 시간/완료 시각 출력 통과.
- 학습/optimizer update는 실행하지 않았다. 새 run의 worker.json/training.jsonl은 생성되지 않았다.
- 검증 기록: `outputs/verification/launcher_fix_20260924T053926_620459Z.json`.
