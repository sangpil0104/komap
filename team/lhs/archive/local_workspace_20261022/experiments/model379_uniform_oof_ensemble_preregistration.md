# Model 379 사전등록: 고정 uniform OOF 확률 앙상블

## 상태와 목적

- 상태: **대기**. A100에서 진행 중인 Model377의 4-fold grouped OOF 완료 및 확률 아카이브 확인 전에는 평가하지 않는다. GPU 학습은 추가하지 않는다.
- 목적: 서로 다른 학습 목적(Model374 baseline, Model376 경계 가중 CE, Model377 directional loss)의 오차가 보완적인지, 하이퍼파라미터 탐색 없이 확인한다.
- 이 결과는 이미 진행한 모델·손실 선택 과정에 조건부인 train-OOF 진단이다. 별도 외부 표본의 독립 일반화 증거로 취급하지 않는다.

## 고정 비교

- 기준: Model374, 공식 train-only grouped 4-fold OOF.
- 유일한 후보: 같은 이미지와 같은 held-out fold에서 생성된 Model374, Model376, Model377의 softmax probability map을 각각 `1/3`씩 평균한 뒤 `argmax`한다.
- 내부 클래스 순서는 `[Al, Al3Ni, Primary Si, Eutectic Si]`로 고정한다. 확률 재보정, class gamma, TTA, morphology, 후처리, per-image routing 및 valid/test 기반 변경은 금지한다.
- 모델별 아카이브는 fold별 이미지 ID 집합·원본 mask ID·공간 크기·확률 유효성·정규화 합이 일치해야 한다. 각 OOF 이미지에 대해서는 그 이미지가 제외된 동일 fold의 세 예측만 결합한다. 다른 fold 모델 예측은 섞지 않는다.
- 70개 train image를 정확히 한 번만 포함하고, official image-wise 4-class mIoU와 empty-union=0으로 계산한다.

## 사전 고정 채택 기준

Model374 대비 아래 조건을 모두 만족할 때만 후속 후보로 남긴다.

1. paired OOF image-wise mIoU 향상 `≥ +0.5%p`.
2. 59개 specimen group paired-bootstrap 95% CI 하한 `> 0`.
3. 4개 fold 중 최소 3개에서 향상하고, 최악 fold 변화 `≥ −0.3%p`.
4. 어떤 class mean IoU도 Model374 대비 `−0.5%p`보다 더 악화하지 않는다.
5. Step4 subset의 image-wise mIoU가 Model374 대비 `−0.5%p`보다 더 악화하지 않는다.

후보는 하나뿐이며, 기준 실패 뒤에 가중치를 탐색하거나 구성 모델을 사후 제외하지 않는다. CI는 specimen-group 표본 변동만 반영하고 학습 seed 변동이나 기존 실험 선택 편향은 제거하지 않는다.

## 해석 및 다음 단계

- gate 통과는 추가 확인의 근거이지, valid/test 성능이나 최종 80% 달성을 입증하지 않는다. OOF에서 하이퍼파라미터를 다시 맞추지 않는다.
- gate 실패 시 이 ensemble은 미채택하고 Model374 기준을 유지한다.
- gate 통과 시에도 실제 배포 후보로 주장하기 전에, 동일한 평균 규칙으로 전체 train 70장에 각 고정 레시피를 학습한 확률원 생성 가능성·추론 정의를 확정해야 한다. 그 후 official valid 20장은 한 번만 최종 보고용으로 평가하며, 해당 점수로 가중치·모델 구성을 변경하지 않는다. Test label은 개발 과정에 쓰지 않는다.

## 고정 규칙의 근거

Claude Code Opus는 Model377 완료 후 비용 없는 우선 진단으로 동일 fold OOF 확률의 uniform 평균을 제안했다. 동일 outer fold의 세 모델만 쓰므로 각 이미지 예측은 세 모델 모두가 학습에서 제외한 이미지에 대한 예측이다. 단, 세 학습 recipe가 기존 OOF 오류 분석 후 설계·선택됐으므로 이 평가는 완전한 nested-CV가 아니며 selection-conditional로 보고한다.
