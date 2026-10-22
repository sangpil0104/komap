# Model382 보조 E5 OOF 분석 사전등록

초안 작성: 2026-10-02T07:41:19Z, Model382 첫 seed/fold의 OOF 확률맵 생성 전. 최종 문안·평가기 SHA-256 lock receipt는 [여기](results/model382_e5_preregistration_lock.json)에 둔다.

이 보완은 [Model382 primary 사전등록](model382_seed_count_control_preregistration.md)의 A/S1/S2/B/C arms, primary 판정 순서, 학습 레시피, 채택 기준을 바꾸지 않는다. 새 학습은 없고, 모든 Model382 seed/fold 결과가 끝난 뒤 train-only OOF 확률맵으로 한 번만 계산하는 고정 secondary arm을 추가한다.

## 고정 arm

`E5 = (Model374 seed0 + Model376 seed0 + Model377 seed0 + Model382 seed1 + Model382 seed2) / 5`.

각 이미지는 해당 이미지의 held-out fold에서 나온 확률맵만 포함한다. 다섯 softmax 맵을 동일 가중치로 평균한 뒤 argmax한다. 가중치·class bias·threshold·TTA·후처리 탐색은 하지 않는다. 이 arm은 기존 recipe-diversity C와 same-recipe seed arm B를 대체하지 않으며, 별도 Model 번호나 별도 checkpoint를 만들지 않는다.

## 평가 및 채택 규칙

- 입력은 70장/59 specimen group의 고정 train-only OOF masks/probabilities뿐이다. Official valid/test와 해당 라벨은 읽지 않는다.
- 네 fold 모두 다섯 멤버의 OOF ID 집합이 해당 fold mask ID와 정확히 일치해야 하고 specimen group이 fold를 넘지 않아야 한다. seed/fold archive 누락, Model382 manifest·checkpoint·OOF hash·recipe 불일치가 하나라도 있으면 primary/E5 평가 전체를 중단한다. Model382 probability archive와 completion manifest의 수정 시각도 lock 시각보다 뒤여야 한다. 부분 평균·대체 멤버는 금지한다.
- 공식 지표는 이미지별 4 class IoU의 산술평균, empty union=0이다. fold 안에서 specimen group을 복원추출하는 paired bootstrap 20,000회, seed 382를 쓴다. 비교 arm 간에는 같은 bootstrap 표본을 쓴다.
- `+0.20%p`는 70개 이미지별 mIoU delta의 산술평균(이미지 가중, pixel-pooled IoU 아님) 기준이다. fold 양수는 각 fold 내부 이미지 평균 delta가 엄격히 `>0`인 경우만 센다. CI는 two-sided 95% percentile interval `[2.5,97.5]`다. A71 제외 점추정은 남은 69개 이미지의 image-mean delta다.
- E5는 **보조 탐색 후보**로만 고려한다. 채택 검토 조건은 모두 만족해야 한다: (1) E5−A가 primary 사전등록의 기존 절대 adoption gate 전부 통과, (2) E5−B와 E5−C 각각 `≥+0.20%p`, 각 paired 95% CI 하한 `>0`, 각 4개 fold 중 최소 3개 양수, (3) E5−A/B/C 모두 A71 제외 점추정이 음수가 아님.
- 조건 미달이면 E5를 채택하지 않고 추가 ensemble 수/가중치 탐색도 하지 않는다. 이 보조 결과로 A/B/C primary 해석이나 adoption gate를 사후 변경하지 않는다.
- 모든 조건 통과는 E5를 단 한 번의 공식 valid 최종 보고 후보로 올릴 자격만 준다. 자동 최종 채택이나 독립 test 성능 주장이 아니다. Valid에서는 가중치·멤버·threshold 재선택을 하지 않는다. E5는 5개 예측을 요구하므로 기존 3-model arm보다 모델 forward 수가 2개 많다.
- 이 OOF 분석은 이전 architecture/loss 선택의 영향을 받는 selection-conditional 증거다. 위 gate를 통과하더라도 독립 test 성능을 증명하지 않으며, valid는 이미 고정된 최종 보고에만 사용한다.
