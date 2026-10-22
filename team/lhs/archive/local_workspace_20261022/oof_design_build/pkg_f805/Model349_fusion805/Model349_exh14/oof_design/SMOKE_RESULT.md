# smoke 검증 상태 (2026-10-09, 진행 중)

이 ZIP은 smoke 검증이 끝나기 전에 올렸습니다. smoke는 학습량을 최소로 줄여 코드 경로만 확인하는 실행이며, 점수에는 의미가 없습니다.

| 단계 | 상태 (학과 P100, fold 1·시드 0, `--smoke`) |
|---|---|
| `prepare_data.py` (90장, 시편 단위 4-fold, fold별 사전확률·클래스 가중치) | 통과 |
| 결정성 런타임 (`cudnn.benchmark=True`를 넣어도 False 유지, 결정적 알고리즘 켜짐) | 통과 |
| `train_fold_seed.py` 19단계 | model6 → 62 → 97 → 98 → 109 → restart51 → restart59 통과, 나머지 진행 중 |
| `predict_oof.py`, `select_oof.py`, `infer.py` | 미실행 |
| `oof_expert/` (E 학습·예측·규칙 선택·결합) | 미실행 |
| A 방식 (train 70장 준비, valid 1회 채점) | 미실행 |

검증이 끝나면 결과를 이 파일에 반영해 ZIP을 다시 올립니다. 전체 규모의 학습과 점수는 아직 없습니다.
