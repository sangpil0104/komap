# Model381 사전등록: Al3Ni–Eutectic 경계 오류 기하 진단

## 목적과 실행 범위

- 목적: Model377에서 Eutectic→Al3Ni 오류는 감소했지만 역방향 오류가 증가한 원인이 경계 위치 불확실성인지, 한 방향의 체계적 변위인지, 모델별 오류 분산인지 구분한다.
- 입력은 70장 train의 exactly-once specimen-group OOF 예측(Model374/376/377)과 해당 train GT뿐이다. valid/test는 읽지 않는다. 모델 학습 없이 Colab CPU에서 실행하며 Model380 P100 CPU 감사와 독립적으로 진행한다.
- 클래스 순서는 `[Al, Al3Ni, Primary Si, Eutectic Si]`; 주 분석 쌍은 Al3Ni(id 1)–Eutectic Si(id 3)로 고정한다.
- 본 진단의 픽셀 거리·oracle은 metallurgical ground truth가 아니라 라벨 기준의 기하 통계다. 임계값은 실행 전에 고정하며 결과에 맞춰 바꾸지 않는다.

## 고정 측정

1. **오류 근접도:** GT Al3Ni/Eutectic 4-neighbor interface에서 exact L2 Euclidean distance transform 기준 ≤2px인 양방향 pair-swap 오류 비율을 각 이미지에서 계산한다.
2. **부호 있는 변위:** GT pair mask 안에서 signed distance `d(to Al3Ni) − d(to Eutectic)`을 계산한다. 예측된 Al3Ni-side boundary pixel에서 GT pair domain에 속하는 픽셀만 표본으로 삼는다. 음수는 GT Al3Ni 쪽에서 예측 경계가 멈춘 것(Al3Ni 과소/Eu 과대 방향), 양수는 GT Eutectic 쪽에서 멈춘 것(Al3Ni 과대/Eu 과소 방향)으로 정의한다. 이미지별 중앙값을 specimen group별 중앙값으로 요약하고, group bootstrap 중앙값 CI를 보고한다.
3. **오류 위치 일치도:** 거리 ≤2px pair-swap 오류 mask의 세 모델 간 intersection/union (Jaccard)을 계산한다. 이미지 합산 픽셀 Jaccard와 specimen-group별 Jaccard 평균/95% CI를 함께 보고한다.
4. **제한적 oracle ceiling:** Model374에서 GT interface 거리 ≤2px 이내의 Al3Ni↔Eutectic swap만 GT 정답으로 바꿔 official image-wise 4-class mIoU, empty-union=0을 다시 계산한다. 다른 경계 오류나 class pair는 고치지 않는다. 이는 실제 가능한 모델 점수가 아니다.

## Model378 SVLS 진행 판정 (결과 확인 전 고정)

다음 네 조건을 모두 만족할 때만 Model378을 실행 가능한 후보로 둔다:

- Model374 양방향 Al3Ni/Eutectic pair-swap 오류의 `≥60%`가 GT pair interface로부터 `≤2px`다.
- Model374 signed-offset group median의 절댓값이 `≤0.5px`이고, specimen-group bootstrap 95% CI가 0을 포함한다.
- Model374/376/377 세 모델의 near-interface 오류 Jaccard (group-equal mean)가 `≥0.50`이다.
- Model374의 제한적 oracle mIoU 상승폭이 `≥+0.50%p`다.

근접도 또는 oracle 조건 실패는 Model378 no-go다. signed-offset 조건만 실패하면 체계적 경계 편향 가능성이므로 Model378을 보류하고 원인을 재검토한다. 오류 Jaccard가 낮으면 SVLS보다 seed/representation variance 분석을 우선한다. 네 조건 통과도 Model378 채택을 뜻하지 않는다. Model378 자체의 기존 사전등록 OOF 성능·작동기전 gate를 모두 통과해야 하며, seed noise floor는 별도로 측정한다.

## 해석 가드

- 이 진단은 Model374/376/377 레시피 선택에 조건부인 탐색 분석이다. 본문 보고와 코드에 이 점을 명시한다.
- pairwise signed distance는 불규칙·얇은 상에서 국소 법선 변위를 직접 측정하는 물리량이 아니다. 거리, 방향, support를 함께 공개하고 금속학적 조성/상평형의 증거로 해석하지 않는다.
- 결과 확인 후 거리, pair, bootstrap 단위, gate를 바꾸지 않는다. 새 방법을 시도할 경우 별도 사전등록을 만든다.
