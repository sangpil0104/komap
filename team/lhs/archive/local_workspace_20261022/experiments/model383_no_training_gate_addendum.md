# Model383 addendum: no-training gates on unseen outer folds

작성: 2026-10-02, before any Model383 training. This addendum is stricter than the original Model383 CPU gate. It does not change Model382 or Model383's fixed training recipe; it can only cancel Model383.

**수정 상태:** 첫 버전의 `>3px`·top-2 저마진·단순 band/full attribution ratio 임계값은 아래의 2026-10-02 Opus 수리 보완으로 폐기했다. 실제 계산/판정은 [v2 코드](model383_no_training_gate_v2.py)와 [v2 smoke](model383_no_training_gate_v2_smoke.py)만 사용한다. 구버전 스크립트는 검토 이력일 뿐 실데이터 gate로 사용하지 않는다.

## 왜 추가했는가

Model383의 exploratory CPU 계산은 Model374 folds 1–2의 36장/30 groups만 포함했다. GT-assisted oracle `+15.3682%p`와 3px 초과 오류 `5.02%`는 학습 가능한 이득이 아니라 부분 OOF 상한/거리 통계다. 이 부분 결과를 근거로 바로 4-fold 학습하면 후보 선택 편향이 더 커진다. Claude Code Opus는 fold3–4의 no-training gate와 기존 Model376과의 메커니즘 비교를 먼저 하라고 권고했다.

## 고정된 추가 게이트

판정 데이터는 **오직 folds 3–4** (34 images)다. 이미 본 folds1–2는 참고 표에만 두고 gate 계산에 합치지 않는다. 모든 확률맵은 공식 train-only grouped OOF여야 하며, 이미지 ID와 fold assignment를 GT mask와 대조한다.

1. **기하/대표성:** Model374의 동일 3×3 relaxation band에서 fold3–4의 GT oracle 이득이 `≥ +5.00%p`이고, band로부터 Euclidean distance가 3px 초과인 오분류 비율이 `<15%`여야 한다.
2. **학습 가능한 불확실성:** Model374의 band 내부 오분류 중 GT 클래스가 top-2에 있으며 argmax 확률과 GT 확률의 차이가 `<0.20`인 pixel 비율이 `≥30%`여야 한다.
3. **기존 Model376 계열의 정보:** 같은 fold3–4에서 Model374 prediction을 band 안에서만 Model376 prediction으로 바꾼 image-mIoU 변화, band 밖만 바꾼 변화, 그리고 전체 Model376 변화 및 band oracle headroom을 계산한다. Model376의 전체 이득이 양수이고 band-only delta가 그 이득의 `≥50%`를 차지하면서 동시에 band oracle headroom의 `<3%`만 회수한다면, 같은 boundary family의 낮은 실현율로 보고 Model383을 취소한다.
4. **중복 구현 확인:** Model376의 실제 training/loss source와 기록된 args를 복구해 해당 loss가 이미 3×3 neighborhood-union likelihood인지 검토한다. 동일 기법이면 Model383을 취소한다. 소스를 복구하지 못하거나 fold3–4 Model376 probabilities가 없으면 이 게이트는 미통과다(대체 자료로 넘어가지 않음).

네 조건 모두 통과해야만 기존 Model383 preregistration의 seed0, 네 fold training 및 adoption gates로 넘어간다. 하나라도 실패하거나 필요한 artifact가 누락되면 이번 BLR 실험은 실행하지 않는다. 이 게이트도 기존 모델/손실 아이디어의 사후 선택 영향을 완전히 제거하지 않으므로 결과는 selection-conditional OOF로 보고한다.

## 2026-10-02 Opus 수리 보완: replacement delta 비율 폐기

아직 folds3–4 실데이터를 보지 않은 상태에서 gate 수학을 재검토했다. 앞 절의 단순 band/full delta 비율은 같은 이미지의 class-IoU 분모를 공유하므로 기여가 가산되지 않고, 작은 full delta에서 불안정하다. 이 비율 기반 취소 규칙과 top-2/0.20 near-miss 게이트는 사용하지 않는다. `>3px` 비율도 반경 1px의 BLR band와 맞지 않으므로 취소 임계값으로 쓰지 않는다. GT band oracle은 BLR이 얻을 수 있는 상한이 아니라 단순 headroom 진단이다.

대신 [수정 gate 코드](model383_no_training_gate_v2.py)가 GT-present 기준으로 고정한 4-class empty=0 image mIoU를 사용해 두 영역 교체의 2-player Shapley 값을 계산한다. `f(∅)=Model374`, `f(B)=374 outside + 376 inside`, `f(O)=374 inside + 376 outside`, `f(BO)=Model376 full`이라 할 때:

`φ_B = 0.5 × [(f(B)−f(∅)) + (f(BO)−f(O))]`, `Δ_full=f(BO)−f(∅)`.

fold3와 fold4 각각 안에서 specimen groups를 복원추출하는 paired bootstrap(20,000회, seed 383)으로 `Δ_full`, `φ_B`, band-oracle headroom `H` 및 `φ_B/H`의 95% percentile CI를 산출한다. **취소 조건은 단 하나:** `Δ_full`의 95% CI 하한이 0보다 크고 동시에 `φ_B/H`의 95% CI 상한이 `0.03` 미만일 때다. 그 외의 모든 경우는 이 proxy 결과만으로 BLR을 취소하지도, 학습 GO를 선언하지도 않는다.

Model376 소스 검토는 오직 정확한 구현 중복 여부를 확인한다. “다른 구현”이라는 사실만으로 Model383의 유효성을 입증하거나 Model376을 BLR의 유효한 proxy로 간주하지 않는다. 소스/archives 누락, split 불일치, 또는 bootstrap 결과 불안정은 자동 학습 허가가 아니며 미결 상태로 남겨 사용자 승인/새 증거를 기다린다. 이 attribution 또한 역사적인 방법 선택을 포함하는 selection-conditional 진단이다.

## 근거 범위

부분 folds1–2 분석은 [기계 판독 결과](results/model383_boundary_relaxation_gate_partial_folds12.json), 계산 코드는 [Model383 CPU gate](model383_boundary_relaxation_gate.py)다. untouched folds3–4의 Model374/376 Shapley attribution 및 group-bootstrap 취소 검사는 [수정 no-training gate v2](model383_no_training_gate_v2.py), [합성 smoke v2](model383_no_training_gate_v2_smoke.py)다. 논문 방법은 [Zhu et al., CVPR 2019](https://arxiv.org/abs/1812.01593)를 참고했으나, 그 벤치마크 성능을 KoMaP 결과의 근거로 사용하지 않는다.
