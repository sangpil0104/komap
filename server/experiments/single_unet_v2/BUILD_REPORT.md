# single_unet_v2 — 6 GPU 전환 기록

- 대상 GPU: **0,1,2,5,6,7**. GPU 3/4는 이 실행에서 사용하지 않는다.
- 기존 S0는 **2,000 / 8,000 updates** 저장을 확인한 뒤 중지했다.
- 다음 학습은 S0의 **남은 6,000 updates**부터 시작한다.
- 원본 run: `/Data3/sppark/komap/experiments/single_unet_v1/outputs/runs/S0_x1_seed2026_official_20260924T045652_436262Z`
- 원본 checkpoint: `/Data3/sppark/komap/experiments/single_unet_v1/outputs/runs/S0_x1_seed2026_official_20260924T045652_436262Z/checkpoints/update_002000.pt`
- 재개 run: `/Data3/sppark/komap/experiments/single_unet_v2/outputs/runs/S0_x1_seed2026_migrate1to6_20260924T052545_946201Z`
- checkpoint SHA256: `efc387d8fa787219effa1c01532fe4ee43c48c967bc9932f2d3c83fe644acc6e`
- 원본과 복사본 hash 일치, model state strict load, 데이터/split 및 원본 소스 snapshot 검증 완료.
- 기존 single_unet_v1의 소스 hash가 모두 유지됨을 확인했다. 이전 checkpoint와 로그도 보존했다.

## 배치와 재개 조건

유효 배치는 기존 **8**이다. 6 GPU에 실제 샘플을 `2,2,1,1,1,1`로 분배한다. 모든 rank가 같은 횟수의 DDP forward/backward를 수행하고, 채움 샘플은 loss/gradient 가중치 0으로 처리한다. 실제 샘플에는 6/8 가중치를 적용해 DDP 평균 이후 실제 8개 샘플 평균 gradient가 되도록 한다.

모델, AdamW state, update, 학습률 위치와 다음 샘플 인덱스를 보존한다. rank0의 RNG는 이어받고 추가 rank는 재현 가능한 새 stream으로 초기화한다. GPU 수 변경으로 stochastic depth/BN 실행 순서가 달라져 1 GPU의 이후 경로와 bitwise 동일하지는 않다.

## 검증

- pytest **24개 통과**.
- 실제 지정 6 GPU에서 유효 배치 8 gradient와 기준값 일치 확인.
- 같은 6 GPU에서 모델의 finite gradient 및 rank 간 동기화 확인.
- 실제 checkpoint의 6 GPU 복원 검사는 `outputs/migration/resume_validation.json`에 기록한다.
- 검증 과정에서 새로운 optimizer step/실제 데이터 학습은 실행하지 않는다.

## 재개

```bash
cd /Data3/sppark/komap
bash experiments/single_unet_v2/run_6gpu.sh
```

기본 순서: S0 이어 학습 → 검증/새 results 보고서 → S1 → S2. 학습 시작은 사용자에게 맡긴다.

S0만 재개하려면 `bash experiments/single_unet_v2/run_6gpu.sh --remaining`을 실행한다.

v2에는 `run.py pause RUN` 명령을 추가했다. 다음 optimizer update 완료 후 저장하고 중지하며 suite도 다음 모델을 시작하지 않는다.

## nohup 통합 로그 추가 (2026-09-24)

- 실행: `bash experiments/single_unet_v2/start_nohup_6gpu.sh`.
- 확인: `tail -F experiments/single_unet_v2/outputs/logs/latest.log`.
- 기존 학습 소스 hash에 포함되지 않는 별도 launcher 두 파일을 추가했다. 기존 v1 소스 13개, v2 소스 16개 및 각 run의 source snapshot hash가 그대로임을 검증했다.
- 원본 및 이관 checkpoint SHA256이 위 값과 일치하고, 재개 위치가 여전히 update 2,000임을 확인했다.
- stdout/stderr, 10 updates마다 학습 지표, 검증 이미지 진행률, 평가 summary, 결과 경로를 하나의 새 로그에 기록한다. 과거 로그는 보존한다.
- 60초마다 전체 경과/남은 시간/총 예상 소요와 KST·UTC 완료 시각을 갱신한다. 시작 전 근사는 약 6시간 48분이며, 새 지표 4개부터 실측 속도로 보정한다. 실제 6 GPU 학습 시간을 측정한 결과는 아니다.
- 추가 검증 **9개 통과**: 초기/실측 ETA, JSONL 부분 쓰기 처리, 로그 중복 방지, plan-only 비실행, 학습 없는 대체 CLI의 정상·중지·실패 상태 및 3단계 출력 통합, 실제 nohup의 stdout/stderr 통합과 이전 로그 보존.
- `bash -n` 통과. 검증에는 fixture 파일만 사용했으며 실제 학습/optimizer update는 실행하지 않았다.
