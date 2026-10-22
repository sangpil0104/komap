# Model386: Step4 Eutectic 거리구간별 FN율 파일럿

작성: 2026-10-02. Model385의 fold1–2 결과와 Opus의 denominator 검토 이후 정의했다. **이번 folds1–2 계산은 탐색용이며 가설 확증이나 학습 선택에 쓰지 않는다.**

## 목적

Model385는 Step4 Eutectic false-negative 중 경계에서 8 native px 이상 떨어진 비율만 측정했다. 이는 GT Eutectic support가 각 거리대에 얼마나 있는지 보정하지 않는다. Model386은 각 거리 구간의 GT Eutectic 픽셀을 분모로 두고 `FN rate = (GT Eutectic, predicted non-Eutectic pixels) / GT Eutectic pixels`를 계산한다.

## 고정 정의

- 입력: Model374 grouped train-OOF probability maps와 동일 fold의 train GT masks만 사용한다. 전송 크기를 줄인 실행에서는 probability archive의 checksum을 기록하고 정확히 `argmax`한 uint8 class map을 사용하며 resize/calibration/postprocess는 하지 않는다. official valid/test 파일·mask·예측은 읽지 않는다.
- 대상: filename에 `_step4_`가 있는 image의 Eutectic Si(class 3)만 본다.
- GT phase boundary: 4-neighbor에서 서로 다른 class에 접하는 양쪽 픽셀.
- 거리는 native GT mask 격자상의 exact L2 Euclidean distance다.
- 고정 구간: `[0,2)`, `[2,4)`, `[4,8)`, `[8,∞)` px.
- Primary summary: 구간별 pooled FN rate와 `<4px` 대 `≥8px` FN-rate difference/ratio. uncertainty는 fold 내부 specimen-group resampling 20,000회, seed 386의 percentile CI로 기술한다.
- OOF key는 해당 fold의 train GT mask stem과 정확히 일치해야 하며 specimen group은 folds 간 분리되어야 한다. 확률은 유한하고 class dimension/정규화 검사를 통과해야 한다.

## 해석·사용 제한

- Model386 folds1–2는 Model385와 앞선 조사에 사용한 데이터이므로 **exploratory pilot only**다. 이 결과로 경계 loss, label relaxation, SDF head를 GO/STOP하거나 임계값을 고르지 않는다.
- folds3–4 결과가 복구되면 동일한 고정 구간 지표를 별도 표로 보고한다. 그래도 방법 선택이 앞선 folds1–2 분석에 조건부임을 명시한다.
- 학습 GO는 이 분석으로 선언하지 않는다. BLR 검토는 기존 [Model383 v2 preregistered gate](model383_no_training_gate_addendum.md)의 원본 조건을 그대로 적용한다. gate 자료가 없으면 학습을 보류한다.
- 이 파일럿은 성능 점수나 금속학적 ground truth 주장이 아니다. 거리대별 miss rate가 비슷하면 “오류가 경계에 농축됐다”는 해석을 지지하지 않는다.

## 구현 및 합성 확인

계산 코드는 [model386_step4_eutectic_distance_rate_pilot.py](model386_step4_eutectic_distance_rate_pilot.py); 합성 거리·분모 smoke는 [model386_step4_eutectic_distance_rate_pilot_smoke.py](model386_step4_eutectic_distance_rate_pilot_smoke.py)다.
