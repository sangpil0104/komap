# Model388 사전 해석·후속 채택 safety addendum

## 상태와 범위

이 addendum은 Model388의 training-GT-only E1 anchor-dose 결과를 확인한 뒤, held-out OOF 점수를 열람하기 전에 작성했다. 2026-10-02 13:17 UTC 확인 시 seed1/fold3의 `valid_probabilities.npz`, manifest, summary 파일은 P100에 생성되어 있었지만, 이 addendum 작성 전까지 해당 score/prediction 값을 읽지 않았다. Model388 원 사전등록의 sampler 처치와 E1/E2 정의는 변경하지 않는다. 이는 점수 결과를 보기 전에 E2 해석 및 향후 학습 채택 기준을 명시하는 보충 문서다.

E1에서 확인된 작은 component의 anchor 재분배는 크다. `<64px` component의 현재 positive anchor 비중은 fold3/4 training pool에서 Al3Ni `10.2–11.4%`, Eutectic Si `30.3–32.4%`였고, Model388 후보는 eligible component가 있는 image/class에서 그 bin의 positive anchor를 0으로 둔다. 따라서 64–255px component dose 증가와 동시에 기존 tiny-component exposure가 감소하는 **재분배 처치**로 표현한다. 단일 목표 bin의 순수한 증량 효과라고 부르지 않는다. Candidate는 Model388 lock에 정의된 그대로 유지하고 OOF를 보고 임계값·fallback·혼합 비율을 바꾸지 않는다.

## E2 결과 해석

- Eligible size bins의 oracle gain은 “FN을 완벽히 정정했을 때”의 upper bound라서 필요한 조건으로만 사용한다. sampler의 실제 causal 효과를 추정하거나 성능 향상을 예측하지 않는다.
- `<64px` 및 목표가 아닌 component-size bins의 E2 headroom은 descriptive-only다. 큰 값은 후보가 해를 낼 증거가 아니고, 작은 값도 안전 증거가 아니다. 이 값들은 GO와 veto 어느 쪽에도 쓰지 않는다.
- GO bin은 원 사전등록의 area `>=64px`인 8 class×bin cells뿐이다. `<64px`는 candidate에서 선택하지 않으므로 GO bin으로 재해석하지 않는다.
- OOF subset, seed, bootstrap, 99.5% CI, group support, largest-group share, dose gate는 원 사전등록 그대로 유지한다. 결과를 본 뒤 bin을 합치거나 gate를 낮추지 않는다.

## 추후 paired sampler 학습 결과의 고정 safety/adoption 기준

Model388 E2 gate를 통과하더라도 학습은 자동 시작하지 않는다. 별도 training preregistration에서 candidate sampler 하나만 Model382 대비 비교하고, 다른 모든 recipe·fold·metric은 고정한다. Candidate OOF는 같은 seed/fold의 Model382 OOF에 image 단위로 짝짓는다. Seed1 candidate를 먼저 평가하고, 사전 조건을 통과하면 seed2에서 같은 처치를 그대로 반복한다. Seed2 실행 여부가 seed1 결과에 조건부임을 보고하고, 결과를 보고 sampler를 다시 조정하지 않는다.

최종 채택에는 아래를 모두 요구한다.

1. Official train-OOF image-wise 4-class mIoU delta가 최소 `+0.50%p`이고, paired specimen-group bootstrap 95% CI 하한이 0보다 크며, 4개 outer fold 중 최소 3개에서 양수다.
2. 네 class별 IoU delta를 모두 보고한다. Al3Ni 또는 Eutectic Si class-IoU의 paired group-bootstrap 95% CI 하한이 `−1.0%p`보다 낮으면 채택하지 않는다.
3. `<64px` GT component hit-rate(성분 내 해당 class 예측 픽셀이 1개 이상인 성분 비율) delta와 그 group-bootstrap CI를 보고한다. Candidate−baseline의 95% CI 하한이 `−5%p`보다 낮으면 채택하지 않는다. 같은 bin의 GT 밖 class false-positive 픽셀 변화도 함께 보고한다.
4. 64–255, 256–1023, 1024–4095, >=4096 bins의 component hit-rate/FN-pixel 변화를 모두 기술한다. 큰 성분 오류가 악화된 경우를 숨기지 않는다. `<64`와 non-target bin oracle 값은 이 guardrail의 대체물이 아니다.
5. Any guardrail CI가 margin과 겹쳐 harm을 배제하지 못하면 `inconclusive / not adopted`로 둔다. CI 전체가 harm margin을 넘으면 `harm signal / not adopted`; 실패는 harm의 인과 증명이 아니다.

이 후보의 비교는 OOF headroom을 보고 gate를 통과해 시행하는 조건부 development experiment다. 독립 test 성능의 증거로 주장하지 않는다. Official validation/test를 새 candidate의 반복 선택·threshold 조정에 사용하지 않는다.
