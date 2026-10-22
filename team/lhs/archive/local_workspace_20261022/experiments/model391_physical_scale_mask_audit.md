# Model391: 물리 배율 정규화 가능성 train-mask 사전 진단

## 질문과 범위

HRNet Model374/382는 모든 입력에 고정 `input_scale=2.0`을 적용하지만 원본 이미지 해상도는 다르다. 같은 물리 시야를 서로 다른 픽셀 밀도로 촬영했다면, 고정 배율에서 상의 경계 두께(픽셀 단위)가 원본 이미지 폭에 비례할 수 있다. 이 mask-only 진단은 그 전제가 train 분할에서 관찰되는지 시험한다. pixel geometry의 상관은 촬영 시야/배율 metadata를 대신하지 않으며, 정규화의 mIoU 개선을 증명하지 않는다.

- 유일한 입력은 `train.zip`의 70개 `train/images/*_image.png`와 70개 `train/masks/*_mask.png` 쌍이다. valid/test, 예측, 체크포인트, OOF/score, outputs는 열지 않는다.
- 각 mask의 4-class RGB를 ID로 복호화한다. 각 class가 다른 phase와 맞닿는 수평·수직 interface 양쪽에서, interface 법선 방향으로 이어지는 같은 class의 axial pixel-run 길이를 측정한다. 이미지별·class별 요약치는 그 run 길이의 median이다. 이는 기존 Model374 boundary-thickness audit와 같은 pixel-space proxy이며 물리 두께가 아니다.
- 종속변수는 이미지별 median run length의 log, 핵심 설명변수는 native image width의 log다. `_step1_`을 기준으로 Step2–4 지시변수를 넣은 OLS slope를 class별로 계산한다. 표본 불확실성은 59 specimen group 단위 cluster bootstrap 20,000회(seed 391), percentile 95% CI로 계산한다. 결측 class-run은 해당 이미지의 해당 class에서만 제외한다.
- 입력 pair 70장/59 group, mask class palette, 이미지/mask native shape 일치, bootstrap 유효 반복 수를 검증한다. 여러 native width가 없거나 디자인 행렬이 rank-deficient이면 결과를 만들지 않고 중단한다.

## 사전 판정 규칙

- 비-Al 3개 phase 중 최소 2개(Al3Ni, Primary_Si, Eutectic_Si)에서 slope 점추정이 `[0.5, 1.5]`이고 bootstrap CI 하한이 `>0.25`이면, 상 pixel scale이 이미지 폭과 함께 변한다는 **정규화 후속 가설**을 지지한다. 이것만으로 GPU 학습은 자동 승인되지 않는다.
- 세 phase 모두 CI 상한이 `<0.5`면 width-based normalization 가설을 보류한다.
- 그 밖은 불확정으로 분류한다. 특정 phase/step을 보고 임계값을 바꾸지 않는다.
- 지지 결과여도 Model382/387/388/389의 기존 locked 순서와 gate를 먼저 마친다. 이후 별도의 HRNet paired 4-fold OOF 사전등록에서만 정규화 학습을 고려한다.

## 재현

합성 회귀 smoke는 실제 데이터 접근 없이 경계 run 길이 및 known-slope 복원을 시험한다: [smoke](model391_physical_scale_mask_audit_smoke.py). 분석 코드: [model391_physical_scale_mask_audit.py](model391_physical_scale_mask_audit.py). 결과 파일은 `experiments/results/model391_physical_scale_mask_audit.json`이며 train-only aggregate 통계만 저장한다. 사전등록, 분석/smoke 코드의 SHA-256, 고정 `train.zip` SHA-256은 [lock receipt](results/model391_physical_scale_mask_audit_lock.json)에 있다.
