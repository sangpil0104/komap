# smoke 검증 상태 (2026-10-09, 팀원 인계)

smoke는 학습량을 최소로 줄여 코드 경로만 확인하는 실행이며, 점수에는 의미가 없습니다. 사용자 지시로 중간에 멈추고 이후 검증을 팀원에게 인계했습니다.

| 단계 | 상태 (학과 P100, fold 1·시드 0, `--smoke`, 90장 B 방식 데이터) |
|---|---|
| `prepare_data.py` (시편 단위 4-fold, fold별 사전확률·클래스 가중치) | 통과 |
| 결정성 런타임 (`cudnn.benchmark=True`를 넣어도 False 유지, 결정적 알고리즘 켜짐) | 통과 |
| `train_fold_seed.py` 19단계 | model6 → 62 → 97 → 98 → 109 → restart51 → restart59 → restart61 → 194 → 195 → 239 통과, model242 실행 중 중지 |
| 나머지 단계 (269 체인, B4 3종, HRNet 2종), `predict_oof.py`, `select_oof.py`, `infer.py` | 미검증 (팀원 인계) |
| `oof_expert/`, A 방식 valid 1회 채점 | 미검증 (팀원 인계) |

전체 규모의 학습과 점수는 없습니다.
