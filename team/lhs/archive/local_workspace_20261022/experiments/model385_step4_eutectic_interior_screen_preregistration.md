# Model385 Stage-0: Step4 Eutectic interior-error feasibility screen

작성: 2026-10-02, 결과 열람 전.

## 목적

Claude Code Opus가 제안한 다중 phase-texture prototype 보조손실 실험에 앞서, 추가 학습 없이 Step4의 Eutectic Si false negative가 GT phase boundary에만 몰리는지 내부까지 퍼지는지 확인한다. 내부 오류가 충분하면 표현학습 후보를 더 검토할 근거가 되고, 거의 전부 경계에 있으면 prototype 실험을 보류하고 이미 사전등록된 Model383 boundary-label-relaxation gate의 자료 복구를 우선한다.

## 고정 입력 및 범위

- Model374 train-only specimen-group OOF folds 1–2의 확률맵과 대응 train GT mask만 사용한다 (36장, 30 specimen groups를 예상).
- Official valid/test 이미지·mask·prediction 및 mIoU는 읽지 않는다.
- OOF 이미지 키는 같은 fold GT mask stem과 정확히 일치해야 하고, specimen group은 folds 간 분리되어야 한다.
- 대상은 파일명에서 `_step4_`가 확인되는 이미지의 GT Eutectic Si (class id 3) 픽셀만이다.

## 정의와 사전 판정

- GT phase boundary: 4-neighbor에서 서로 다른 GT class에 접한 양쪽 픽셀 모두를 boundary로 둔다.
- 거리: 원본 mask pixel 격자에서 Euclidean distance transform을 쓴다. 입력 해상도를 리사이즈하지 않는다.
- Eutectic false negative: `GT == Eutectic_Si`이고 argmax 예측이 Eutectic_Si가 아닌 pixel.
- Interior FN: 위 pixel 중 가장 가까운 GT phase boundary까지 거리가 `>= 8` native pixels인 경우. 그 미만은 near-boundary FN이다.
- Primary statistic: 전체 Step4 OOF 이미지에서 pooled interior Eutectic FN / pooled Eutectic FN. specimen group을 fold 안에서 재표집하는 paired cluster bootstrap 20,000회, seed 385로 95% percentile CI를 함께 보고한다.
- Feasibility screen은 point estimate `>= 0.40`일 때만 통과한다. 통과는 학습 승인이나 일반화 증거가 아니다. 실패하면 prototype-loss 학습을 진행하지 않고 Model383 자료 복구 쪽에 우선순위를 둔다.
- False-negative의 예측 목적지 class를 interior/near-boundary로 분리해 기술한다. 이 사후 기술표로 hyperparameter나 다음 후보를 sweep하지 않는다.

## 구현 및 검증

평가기: [model385_step4_eutectic_interior_screen.py](model385_step4_eutectic_interior_screen.py). 코드 검증은 synthetic smoke test와 원격 P100의 실제 OOF key/hash 대조를 수행한다. 기록할 결과는 JSON이며, raw per-image diagnostic 외에 classification/selection은 하지 않는다.

## 다음 단계 조건

- Stage-0가 실패하면 prototype experiment는 중단한다. 이는 Model383 자동 GO를 뜻하지 않는다. Model374 folds 3–4 및 Model376 OOF 복구와 기존 Model383 gate가 여전히 필요하다.
- Stage-0가 통과하면, 별도 사전등록 및 필요한 fold checkpoint 확보 뒤에만 multi-prototype 보조손실의 4-fold grouped-OOF paired control을 설계한다. 추가 Colab GPU 이용/학습은 기존 사용자 지시와 Colab 권한을 확인한다.
