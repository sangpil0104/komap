# Model388: minority-phase component-balanced crop sampler 감사

## 동기와 식별 한계

Model382의 현재 sampler는 image당 12 crops/epoch 중 기대상 절반을 positive로 정하고, 해당 image에 존재하는 minority classes `[Al3Ni, Eutectic Si]` 중 하나를 균등 선택한 뒤 그 class의 pixel coordinate를 균등 선택한다. coordinate는 `(image,class)`마다 최대 4,000개로 제한된다. crop native size는 `512 / input_scale 2 = 256 px`이며 선택 pixel은 ±64 px jitter 뒤에도 crop 안에 남는다.

처음 고려한 “held-out component의 sampler 노출량과 그 component OOF 오류의 상관” 분석은 **실행하지 않는다**. Grouped OOF에서 해당 held-out image/specimen group은 모델 학습에서 완전히 제외되므로 그 component 자체는 학습 crop에 노출된 적이 없다. Fold 간 training pools도 강하게 겹치고 fold가 4개뿐이라 exposure 차이와 fold 난이도를 분리할 수도 없다. 따라서 OOF와 training-fold exposure를 연결한 인과·상관 효과는 기존 자료로 식별되지 않는다.

대신 두 독립 정보를 계산한다: (E1) training-fold GT만 이용한 sampler manipulation/dose check, (E2) held-out OOF만 이용한 component-size-bin 오류와 perfect-correction mIoU 상한. 둘은 원인-결과 추정치가 아니다. E1은 후보 sampler가 component exposure를 실제로 바꾸는지만, E2는 그 bin 오류가 공식 점수에서 가질 수 있는 최대 headroom만 말해 준다.

## 고정 대상과 split

- 대상 GT class: class id `1=Al3Ni`, `3=Eutectic Si`.
- Connected components: native-grid 8-connectivity.
- Candidate가 실제로 균등 sampling하는 eligible components는 area `>=64 native pixels`; 작은 isolated pixels/label speckle (`<64`)은 새 방식으로 oversample하지 않는다. Eligible component가 없는 image/class는 기존 pixel sampler를 fallback으로 쓴다.
- Area bins (native pixels), class마다 고정: `[1,64)`, `[64,256)`, `[256,1024)`, `[1024,4096)`, `[4096,∞)`. 첫 bin은 노출·오류 기술통계에만 포함되고 candidate의 eligible 대상은 아니다. 총 10 class×bin cells.
- 주 subset은 outer folds3–4, 탐색 subset은 folds1–2. Seed1을 주 분석, seed2를 재현성 확인으로 별도 보고한다. Official valid/test 및 그 라벨·예측은 열지 않는다.
- 각 outer fold f의 exposure 계산은 fixed train split에서 f가 아닌 세 fold의 masks만 쓴다. OOF outcome은 오직 f의 held-out masks/probabilities를 쓴다. Image/group ID의 정확히 한 fold 소속을 검사한다.

## E1: training GT 전용 expected anchor-dose manipulation check

Model382의 정확한 positive-sampling 규칙을 기대값으로 계산한다. image당 70 epochs × 12 crops, positive 확률 `0.5`다. minority class가 K개 있는 image에서 각 class의 expected positive anchors는 `70×12×0.5/K`다.

- 기존 pixel-uniform sampler: component j(area A_j, class total pixels P_c)의 expected anchor count는 `positive_anchors × A_j/P_c`다. 4,000 coordinate reservoir는 uniform subsample이므로 그 reservoir randomness에 대한 기대값을 쓴다.
- 비교할 단일 변경 후보: 기존 positive 확률과 class 선택확률은 보존하고, eligible components(area≥64)가 있으면 class 안에서 component를 균등 선택한 뒤 해당 component의 pixel을 균등 선택한다. 해당 class의 총 expected anchors는 유지한다. eligible component가 없으면 기존 pixel sampler로 fallback한다. 실제 제안 학습 코드가 구현되기 전에는 이 비교를 분석상 후보로만 취급한다.
- 보고: fold·class·area bin별 component/group 수, 기존/후보의 기대 anchor 수, candidate/current dose ratio, class 전체 positive-anchor 총량 변화(0이어야 함). 이는 crop 전체의 50% component coverage가 아니라 crop에 보장 포함되는 선택 anchor 수다. 미세 구조 전체의 실제 시각 노출이나 학습 효과를 의미하지 않는다.

## E2: train-OOF 전용 component oracle headroom

Model382 seed별 고정 OOF 확률을 argmax하고 held-out GT에서 class별 8-connected components를 구한다. 각 class×area bin을 차례로 독립 처리해 해당 bin GT component 안에서만 FN pixels를 GT class로 바꾸고, 다른 모든 예측은 그대로 둔다. 공식 image-wise 4-class mIoU(empty union=0)를 baseline과 비교한다. 이 perfect-correction delta는 달성 가능한 성능 예측이 아닌 상한이다. 각 image에 그 bin component가 없어도 delta 0으로 공식 image 평균에 포함한다.

- 주 분석은 seed1, folds3–4. Seed2, folds3–4는 sign/magnitude replication; folds1–2는 exploratory only.
- 각 bin의 image-wise oracle gain, fold별 gain, FN pixels, component/group support, 최대 specimen-group headroom share를 보고한다.
- 불확실성은 fold 안에서 specimen group을 복원추출하는 paired 20,000회 bootstrap(seed `388`)으로 계산한다. mIoU는 반복마다 image mean으로 다시 계산하며 pixel/component를 독립 표본으로 취급하지 않는다.
- E2가 sampler 효과를 식별한다고 해석하지 않는다. Model382/388 후보를 비교하는 학습 실험은 이 자료를 보고 가설을 선택한 조건부 development test가 된다.

## 후속 P100 학습 gate (자동 실행 금지)

새 sampler 학습은 현재 Model382의 여덟 OOF run이 모두 끝나고 audit source/hash 검증이 통과한 뒤에만 검토한다. 아래 조건을 모두 충족하는 사전 지정 class×area bin이 하나 이상 있어야 별도 training preregistration을 작성할 수 있다.

1. **Headroom:** seed1, folds3–4 pooled oracle gain의 99.5% group-bootstrap CI 하한이 `+0.005` absolute mIoU 이상이다. (10개 bin family를 보수적으로 보정.)
2. **일관성:** folds3과 4가 모두 양수이고, 최고 group 하나가 전체 oracle gain의 50% 미만이며, seed2도 같은 부호이면서 seed1 gain의 절반 이상이다.
3. **Support:** 각 주 fold에 해당 bin component가 30개 이상, 5개 이상 specimen groups에 걸쳐야 한다. 미달은 underpowered NO-GO다.
4. **Dose:** E1에서 해당 bin의 candidate/current expected component-anchor ratio가 folds3·4 각각 `>=2.0`; class 전체 expected positive anchor count 변화는 `0`이어야 한다.

어느 하나라도 실패하면 component sampler 학습은 시작하지 않는다. Gate를 통과하더라도 자동 실행/채택은 하지 않는다. P100 추가 학습은 user 지시대로 Model382 종료 후 별도 순차 작업이며, 나머지 recipe는 변경하지 않는다. 비교 실험의 OOF 효과는 data-driven gate 선택에 조건부인 개발 evidence이지 independent test가 아니다.

## 재현

```bash
python experiments/model388_component_sampler_audit.py \
  --mode exposure --cv-root /path/to/fixed_group_cv --folds 3 4 \
  --output experiments/results/model388_exposure_folds3_4.json

python experiments/model388_component_sampler_audit.py \
  --mode headroom --cv-root /path/to/fixed_group_cv \
  --model382-root /path/to/model382_seed_count_control --folds 3 4 \
  --output experiments/results/model388_headroom_folds3_4.json
```

E1은 training masks만 읽고 CPU-only다. E2는 고정 train OOF masks/probabilities만 읽고 CPU-only다. 둘 다 valid/test 경로를 인자로 받지 않는다.
