# Model387: 해상도 병목 여부를 위한 train-OOF 경계 오류·oracle 상한 감사

## 목적과 범위

Model386의 folds1–2 Step4 분석에서 Eutectic Si `>=8 px` 내부 FN율이 높았지만, 내부 support의 87%가 A182 한 시편에 집중됐다. 그러므로 이를 일반적인 경계/feature-stride 병목으로 간주하지 않는다. Model382의 고정 HRNet-W48/U-Net++ seed1·2 train-OOF에서 모든 phase를 함께 분석해, 오류가 경계대에 일관되게 몰리는지와 그 구간을 완벽히 고친다는 비현실적 상한에서 official image-wise mIoU 개선 여지가 얼마나 되는지를 측정한다.

이것은 진단이지 새 모델·해상도 변경의 성능시험이 아니다. 이 분석만으로 고해상도 학습을 자동 시작하거나 성능 향상을 주장하지 않는다. official valid 20장과 test 10장, 그 라벨·예측은 읽지 않는다.

## 데이터와 사전 구분

- 입력은 Model382 seed1 및 seed2의 네 fold train-OOF 확률 아카이브와 해당 held-out train mask뿐이다. 지정된 fold의 모든 image가 exactly-once OOF이며 group은 fold 사이에 중복되지 않아야 한다.
- **주 진단 집합:** folds3–4. Model386 거리 가설의 생성에 사용된 folds1–2와 분리한다. 결과를 확인하기 전 이 문서와 분석 코드를 고정한다.
- **탐색 집합:** folds1–2. 앞선 Model385/386 탐색과 겹치므로 재현·비교만 하고 확증 근거로 합치지 않는다.
- seed1을 주 분석, seed2를 같은 고정 recipe의 재현성 확인으로 보고 각 결과를 분리한다. 두 seed를 독립 표본처럼 합산하지 않는다.
- 결과 파일은 전체 source archive·mask 해시, 선택한 fold와 seed, group/image 수를 기록한다. Model382 기존 preregistration lock 및 fold manifest·checkpoint·OOF hash 검사를 통과하지 않으면 중단한다.

## 고정 지표

1. 원본 native GT의 4-neighbor class transition 양쪽 pixel을 phase boundary로 표시하고 정확한 Euclidean distance로 가장 가까운 boundary까지 거리를 구한다. 사전 고정 구간은 `[0,2)`, `[2,4)`, `[4,8)`, `>=8` native pixel이다.
2. 각 image·seed·class·거리대에서 GT class support와 false-negative 수를 세고 `FN / GT support`를 계산한다. support가 0인 경우 rate는 null로 남기며 pooled pixel rate와 group별 분포를 구분해 보고한다.
3. 각 거리대에 속하는 모든 픽셀의 예측을 GT로 바꾼 oracle mask를 만들고, 공식 image-wise 4-class mIoU(empty union=0)의 baseline 대비 차이를 계산한다. 이는 해당 거리대의 **완벽한 correction 상한**이지 실제 개선 예측이 아니다.
4. 주 대비는 각 class별 `[0,4)`와 `>=8`의 FN율 차이다. 보조 지표는 전체 four-class oracle mIoU 상한과 거리대별 support/FN 절대량이다. Eutectic Si만 선택해 결론 내리지 않고 네 class를 모두 보고한다.
5. 불확실성은 specimen group을 outer fold 내에서 복원추출하는 20,000회 paired bootstrap(seed `387`)으로 계산한다. mIoU oracle gain은 공식 image 단위 평균을 유지하며, 픽셀 FN율은 매 반복의 group 표본에서 support와 miss를 합산해 다시 계산한다. fold별 결과도 별도 제시한다.

## 사전 고정 해석 규칙

- folds1–2 결과는 탐색으로만 표기한다. folds3–4에서 관측한 효과크기·방향·group별 일관성과 seed1/2 재현성을 함께 제시하고, pixel 수를 독립 표본 수처럼 취급하지 않는다.
- 거리대별 oracle gain은 달성 가능한 상한이다. 상한이 작으면 그 거리대 오류만 겨냥한 고해상도 실험을 우선하지 않는다. 상한이 커도 resolution/feature-stride가 원인이라는 증거는 아니다.
- 같은 위치의 높은 오류가 두 seed와 여러 specimen group에서 재현되어야 해상도 개입 후보로 논의할 수 있다. A182 단독 또는 한 fold 단독 신호는 GO 근거가 아니다.
- 이 audit을 보고 새 resolution recipe를 선택하면 그 후속 비교는 가설 선택에 조건부인 개발 실험으로 명시한다. 독립 test 성능을 주장하지 않으며, valid로 scale·stride·threshold를 튜닝하지 않는다. 새 학습 전에는 contrast, split, compute budget, adoption gate를 별도로 사전등록한다.
- 이 파일은 Model382 학습 시작 뒤 작성됐다. 주 분석 fold3–4 OOF probability/manifest가 생성되기 전이며, 분석 정의·코드 hash lock 후 해당 결과를 읽는다. train loss/epoch는 사전등록 지표나 선택 기준이 아니다.

## 재현 명령

Model382 여덟 fold가 모두 완료되고 archive/hash 검증이 끝난 뒤, P100 작업공간의 같은 mask·OOF 경로를 사용해 CPU에서 실행한다.

```bash
python experiments/model387_resolution_headroom_audit.py \
  --cv-root /path/to/fixed_group_cv \
  --model382-root /path/to/model382_seed_count_control \
  --folds 3 4 \
  --output experiments/results/model387_resolution_headroom_folds3_4.json
```

탐색 folds1–2 재현은 output 파일명을 별도로 지정한다. 이 분석은 GPU를 사용하지 않는다.
