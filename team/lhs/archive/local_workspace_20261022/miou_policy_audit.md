# Validation image-wise mIoU: empty-class policy audit

2026-09-26 기준으로 저장된 validation 결과를 두 결측 정책에 맞춰 다시 정리했다. 두 값 모두
이미지별 4개 클래스 IoU를 평균한 뒤 20장 평균을 내며, 차이는 `TP+FP+FN=0`일 때뿐이다.

| 기준/상태 | empty union = 0 | empty class 제외 (NaN skip) | 판정 |
|---|---:|---:|---|
| 최고 raw validation route: Model270 | **78.7209%** | **79.4978%** | valid 최고점; CV 일반화 근거 부족 |
| 현재 채택 기준 route: Model225 | 78.6921% | 79.4693% | 현재 보수적으로 유지 |
| 최고 단독 probability archive: Model289 | 77.4181% | 78.0528% | route calibration/ensemble 전 단일 archive |

Model270은 Model225보다 raw valid에서 `+0.0289%p` (`empty=0`), `+0.0285%p` (skip) 높다.
그러나 Model225 대비 19개 specimen-group leave-one-out 변화는 `-0.1320%p`였고, paired
group-bootstrap 95% 구간 `[-0.3548, +0.0574]%p`가 0을 포함했다. 개선 이미지는 `9/20`이라
현재 최고 route로 바꾸지는 않았다. 80% 목표까지의 간격은 raw 최고 기준 `1.2791%p`이며,
skip 기준 `0.5022%p`다. 공식 evaluator의 결측 정책이 확인되기 전까지 두 수치를 함께 본다.

재채점 범위:

- P100의 191개 `results.json`에서 20장 per-image 점수 벡터 5,002개를 다시 산술 평균했다.
  결과 JSON 안의 `best_*`와 `records` 중복이 포함되며, 저장된 값과 `1e-9` 이상 차이는 없었다.
- 188개 NPZ를 검사했고, validation 20장 전부의 class probability가 있는 111개 archive를
  정답 mask에 직접 대조해 다시 점수화했다. 오류는 0건이었다.
- 기존 모델과 팀 B4는 class channel 순서가 다르므로 별도로 매핑했다. 기존 순서는
  `[Al3Ni, Primary Si, Al, Eutectic Si]`, 팀 B4 순서는 `[Al, Al3Ni, Primary Si, Eutectic Si]`다.
- 전수 결과(이번 재실행): [111개 확률 아카이브 재채점 JSON](miou_policy_audit/rerun_20260926/archive_rescore.json),
  [5,002개 저장 점수 벡터 재평균 JSON](miou_policy_audit/rerun_20260926/score_vectors_recomputed.json).
  기존 감사 산출물은 별도 보존했다. Model321 결과까지 포함해 재검사했으며, 최고 route는 변하지 않았다.
  원본은 P100의 `/workspace/komap_model1/outputs/miou_policy_audit/`에도 보존했다.
- 사용자 요청에 따라 P100에서 같은 범위를 다시 전수 계산해 별도 보관했다:
  [확률 아카이브 재채점 JSON](miou_policy_audit/rerun_20260926_user/archive_rescore.json),
  [저장 점수 벡터 재평균 JSON](miou_policy_audit/rerun_20260926_user/score_vectors_recomputed.json).
  재검사도 191개 결과 JSON / 5,002개 점수 벡터, 188개 NPZ / 111개 valid 아카이브였고,
  파싱·재계산 오류는 0건이었다. 두 정책 모두 최고 route는 Model270으로 동일했다.
  당시 진행 중이던 Model322는 완성된 valid 예측이 없어 포함되지 않았다.

로컬 재현 도구는 [recompute_miou_policy.py](recompute_miou_policy.py)와
[recompute_miou_recorded_results.py](recompute_miou_recorded_results.py)다. 이 감사는
기존 validation 결과의 산출값을 통일해 읽기 위한 것이며, 20장 validation 선택으로 생길 수 있는
과적합을 제거하거나 독립 test 성능을 보증하지는 않는다.
