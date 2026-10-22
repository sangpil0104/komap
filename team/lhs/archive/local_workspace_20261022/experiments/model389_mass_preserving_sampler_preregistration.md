# Model389 (가칭): eligible-component 균등 anchor sampler

## 상태와 기존 결과와의 분리

Model389는 **별도 조건부 후보**이며, 이미 hash-lock된 Model388 정의를 수정·대체하지 않는다. Model388 E1은 training-GT-only지만 eligible component를 고르는 경우 `<64px` positive anchors를 0으로 재배치하는 큰 trade-off를 보였다. Model389 후보는 tiny-component anchor mass를 현재 방식과 정확히 같게 보존하고, `>=64px` component 내부의 기존 mass만 component 간 균등 재분배한다.

현재 seed1/fold3 OOF files는 P100에 생성됐지만 이 분석은 score/prediction 값을 열기 전에 작성한다. 따라서 이 문서는 **pre-score-inspection lock**이지 OOF artifact 생성 이전 preregistration이라고 주장하지 않는다. OOF 결과를 본 뒤 component threshold, candidate, gate를 고치지 않는다. 목표는 공식 train-OOF evidence가 `64–255px` Eutectic component를 놓치는 구조와 일치할 때만 순차 sampler 비교를 고려하는 것이다.

## 고정 sampler 처치

Image `i`에 minority class 중 존재하는 수를 `K_i`라고 하면 positive-anchor budget은 `B_i=70×12×0.5/K_i`다. Class 선택확률·positive crop 확률·epoch/crop 수·crop geometry·augmentation·loss·optimizer·model은 기존 recipe 그대로다.

Class `c`의 총 GT pixels를 `P_ic`, 8-connectivity GT component j 면적을 `A_ij`, `<64px` component pixels 합을 `S_ic`, `>=64px` components 집합을 `E_ic`, 그 수를 `n_ic`라 둔다.

- `<64px` component j expected anchors: `B_i × A_ij/P_ic` (현재 pixel sampler와 동일).
- `>=64px` component j expected anchors: `B_i × (P_ic−S_ic)/(P_ic×n_ic)`.
- `n_ic=0`이면 현재 pixel sampler로 fallback한다.
- 따라서 image/class별 positive-anchor 총량과 tiny-bin mass는 보존된다. 바뀌는 것은 eligible mass 안에서 pixel-area proportional 선택을 component-uniform 선택으로 바꾸는 것 하나다.
- `MAX_MINORITY_COORDS=4000` 제한은 tiny pool 및 각 eligible component 안의 uniform reservoir로 구현한다. 구현 전 CPU simulation에서 기대값과 일치하는지 검증한다.

## E1: 정확한 dose manipulation gate (train GT만, CPU)

Fold3·4의 training masks만 읽고 image/class별 닫힌식으로 계산한다. Monte Carlo 추정이 아니다.

- `<64px` bin dose ratio는 각 image/class에서 정확히 `1.0`이어야 하며 class 전체 expected anchor 변화도 0이어야 한다(절대 오차 `≤1e−9`). 어기면 구현 오류/NO-GO.
- 주 대상은 Eutectic Si `64–255px` bin이다. 각 주 fold에서 총 candidate/current expected anchor ratio `≥1.25`, component별 ratio 중앙값 `≥1.25`여야 한다. 해당 bin은 각 fold `≥30 components` 및 `≥5 specimen groups`에 분포해야 한다.
- Al3Ni `64–255px` dose, 모든 다른 bins, `>=1024px` dose, tiny/eligible fallback image 수·비율도 보고한다. 큰 bin dose가 `<0.5×`면 heavy-redistribution 경고를 붙인다.
- folds1–2는 탐색용, folds3–4를 주 분석으로 고정한다. Official valid/test 및 OOF는 읽지 않는다.

## E2: 고정 Model382 train-OOF feasibility gate (CPU, training effect 아님)

Model382 seed1/2 raw OOF probabilities에서 argmax를 구하고, native-grid held-out GT의 8-connected Eutectic components로 계산한다. 주 분석은 seed1/folds3–4, seed2는 별도 재현 확인, folds1–2는 탐색용이다.

1. **Component miss 정의:** GT component 내 해당 class로 예측된 pixel 비율(recall)이 `<0.5`이면 해당 component를 miss로 센다. Target miss rate는 `[64,256)` Eutectic components 중 missed/total이다. Comparator는 `>=256px` Eutectic components 전체 missed/total이다. Pixel이나 component를 독립 bootstrap 단위로 취급하지 않는다.
2. **Mechanism gate:** seed1 folds3–4 pooled target miss rate `≥0.30`, comparator 대비 miss-rate ratio `≥2.0`, absolute difference `≥0.15`여야 한다. folds3과 4 각각 target rate가 comparator보다 높아야 하며 group-stratified paired bootstrap의 95% difference CI 하한이 0보다 커야 한다. Seed2도 같은 방향이어야 한다. Comparator rate가 0이고 target rate가 `≥0.30`이면 ratio는 수학적으로 무한대로 취급하되 JSON에는 `null`로 기록하고, difference 조건도 반드시 통과해야 한다. 두 rate가 모두 0이면 NO-GO다.
3. **Official-score headroom:** `[64,256)` Eutectic component 안의 FN만 GT class로 oracle correction하고 공식 image-wise 4-class mIoU(empty union=0)를 다시 계산한다. seed1, folds3–4 pooled 평균 oracle gain point estimate `≥+0.005` absolute mIoU, 20,000회 specimen-group bootstrap 95% CI 하한 `≥+0.0025`여야 한다. fold별 gain 양수, 각 fold `≥30 target components/≥5 groups`, seed2 동일 방향 및 seed1 gain의 절반 이상을 요구한다.
4. Al3Ni 및 `<64px`·`>=256px` bin headroom/miss-rate는 별도로 보고한다. Oracle 결과는 upper-bound screen일 뿐 sampler의 causal effect나 `<64px` 개입 안전성을 말하지 않는다.

## 순차 비교 순서 및 채택

1. Model382의 여덟 train-OOF run 완료 뒤 Model387, Model388, Model389의 고정 CPU audits를 각각 실행한다. 기존 Model388 E2 gate와 Model388 safety addendum은 변경하지 않는다. 새 GPU 학습은 자동 시작하지 않는다.
2. Model388 E2가 GO이면 기존 Model388 처치를 먼저 seed1/four-fold P100 paired OOF로 시험한다. 기존 safety addendum의 adoption 기준을 그대로 적용한다. Model388이 채택되면 Model389는 실행하지 않는다.
3. Model389는 다음 두 조건을 모두 충족할 때만 별도 P100 candidate로 실행한다: Model389의 E1/E2 gate를 통과했고, Model388 training comparison이 `<64px` component hit-rate guardrail에서 harm/inconclusive이거나 해당 hit-rate delta point estimate가 음수다. Model388의 원인 무관한 overall gain 부족만으로 Model389를 자동 실행하지 않는다. Model388 E2 target headroom 자체가 실패하면 Model389는 target dose가 더 작고 같은 component errors를 겨냥하므로 NO-GO다.
4. **Model389 paired OOF adoption gate:** fixed four-fold train OOF에서 Model382 same-seed baseline 대비 official image-wise mIoU delta point estimate `≥+0.25%p`, specimen-group paired bootstrap 95% CI 하한 `>0`, 4 fold 중 `≥3` 양수. 네 class 모두 보고하고 Al3Ni/Eutectic class-IoU 95% CI 하한 `≥−1.0%p`, `<64px` component hit-rate delta 95% CI 하한 `≥−5%p`를 요구한다. Seed1을 먼저 평가하고 gate 통과 시 seed2를 똑같이 반복한다. Bootstrap CI는 fixed model의 group/image 표본 변동이지 학습 seed 분산이 아니다. 두 seed의 방향 일치만 별도로 확인한다.
5. 두 Model388/389 비교는 OOF headroom을 이용해 선택한 조건부 development experiments다. Official validation/test로 candidate·threshold·sampler를 반복 선택하지 않고 독립 test 성능을 주장하지 않는다.

## 재현 및 잠금

E1/E2 CPU 분석 코드는 [Model389 audit script](model389_mass_preserving_sampler_audit.py), synthetic fixture는 [smoke](model389_mass_preserving_sampler_audit_smoke.py)다. Seed1/fold3 OOF files가 이미 생성된 사실과, 이 hash lock 시점까지 OOF values를 읽지 않았음을 lock receipt에 기록한다. Model382가 현재 쓰는 P100에서는 이 audit 때문에 추가 GPU job을 실행하지 않는다.
