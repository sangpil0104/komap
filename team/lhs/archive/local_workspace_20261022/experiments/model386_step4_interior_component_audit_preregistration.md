# Model386 follow-up: Step4 Eutectic interior component concentration

작성: 2026-10-02, Model386 folds1–2 distance-band rates와 group concentration을 확인한 뒤. 본 분석은 **post-hoc exploratory audit**로 명시하며 모델 선택·학습 gate가 아니다.

## 목적과 고정 측정

- 대상은 Model374 train-OOF folds1–2의 Step4 이미지와 같은 fold train GT만이다. official valid/test를 읽지 않는다.
- interior는 GT Eutectic(class 3)에서 모든 4-neighbor GT phase boundary까지 Euclidean native-pixel 거리가 `>=8px`인 픽셀이다.
- 해당 GT interior mask에서 8-connectivity connected component를 모두 찾고, 각 component의 pixel support, false-negative 수/율, argmax 도착 class와 bounding box를 기록한다. 크기 기준으로 작은 성분을 사후 제거하지 않는다.
- 시편별 합계·분포와 상위 group의 interior support/FN 점유율을 요약한다. 이 지표는 pixel/component의 독립성을 가정한 검정이나 일반화 추정치로 쓰지 않는다.
- 5, 40% 등 임계값을 추가하지 않으며, 어떤 결과도 Model385 prototype stop 또는 Model383 folds3–4 gate를 바꾸지 않는다.

## 입력·무결성·재현

Model374 probability maps에서 argmax로 lossless class-ID archive를 만들고 원본 probability/argmax archive checksum을 manifest에 보존한다. Fold별 key는 train mask stem과 정확히 맞아야 하고 specimen group은 fold 간 분리되어야 한다. 구현은 [component audit](model386_step4_interior_component_audit.py), Colab runner는 [runner](model386_component_colab_runner.py)다.
