# 일반화 한계와 다음 개선 경로

## 보장 가능한 것과 평가해야 하는 것
현재 100장 중 정답은 train70+valid20=90장이다. valid는 광범위한 선택에 노출되었고 train-valid 7개 시편이 중복된다. test10은 정답이 없다. 어떤 학습/경량화 방법도 임의의 새로운 시편에서 80% 유지/향상을 보장하지 못한다.

체크포인트 ZIP의 모든 entry/tensor가 정확히 같고 로더·추론 계산이 같다면, ZIP 저장 포맷 변경 자체는 모델 함수를 바꾸지 않는다. 이는 base의 기존 일반화 능력을 보존하는 설명이며 base의 새 데이터 정확도를 보증하는 것은 아니다. 캐시/모델 객체 재사용도 전체 TENT 초기 상태/optimizer/RNG 등 의미적 동등성을 따져야 한다. AMP/배치 변경은 수치 근사를 포함해 valid/test 일치가 높더라도 모든 입력에 대한 동등성 증명이 아니다.

## 완료된 실행 최적화의 실제 결과
`base12_resident_packed_20261008` 완료/PNG 재채점/원본 보존 확인. 원본 base12 683,320,900→628,669,975바이트(7.99784% 감소), 모든 checkpoint ZIP entry/tensor 정확히 동일. 고유 가중치 수는12개다. 결합판에는 전체 TENT 원본 복원 + HRNet FP16 autocast/batch8이 포함되어 ZIP만 변경한 판과 구분한다.

각3회 교차 전체 valid20: 기준 [347.63134,347.93431,347.51782]초, 후보 [331.39949,331.27345,331.15790]초. 중앙값 347.63134→331.27345초, 4.70553% 감소. 모든 반복 품질 조건을 통과했으나 사전 시간 10% 조건은 실패했다. raw image mIoU80.1251872%, 미세 수치 차이는 정확도 향상으로 주장하지 않는다. test10 마스크 일치99.9972452%; test IoU/정확도가 아니다. shared P100 측정이며 CPU/다른 환경 재현 한계가 남는다.

후속 singleworker12는 초기 GPU 유휴 검사에서 memory6MiB/utilization5%를 감지해 `GPU7 occupied; no worker started`로 종료했다. 후보 학습/추론/반복 속도 결과가 없다. 외부 작업을 종료하거나 실패 파일을 지우지 않는다. 이는 일반화 실험 실패나 방법의 성능 기각이 아니다.

## 권장 순서
1. **예측 변경 없는 구현/저장 최적화 우선.** ZIP 무손실 저장, 중복 로드/전처리 제거, 동일 계산을 한 worker에서 수행하는 구현. ZIP tensor 동등성과 실행 의미 보존, 필요 상태 복원, 반복 시간 측정을 분리한다. 순수 구현 변경은 valid 점수로 구성 선택을 하는 것보다 선택 편향을 추가할 여지가 작다. AMP/양자화/타일/TTA 축소는 별도 근사 변경이다.
2. **실질적인 모델 수 축소는 부분 증류부터.** 329의 B4 4fold를 같은 입출력의 학생1개로 근사하고 HRNet/269/고해상도는 보존하는 고정 후보를 첫 연구 대상으로 삼는다. 성공 시12→9개지만 성능/속도 유지가 사전에 보장되지 않는다. 학습은 정답 loss+teacher soft-target loss를 함께 사용한다. teacher 오류를 무조건 따라야 하는 것은 아니고 추가 무라벨 데이터가 증류의 필수 조건도 아니다. 다만 70장의 작은/상관된 표본과 학생 용량 차이가 실질적인 제약이다. 77.5~78.5%라는 기존 학생 예상은 학습으로 검증한 성적이나 한계가 아니다.
3. **성능 향상 연구는 학습 단계의 시편 변이 대응.** 물리적으로 가능한 조명/대비/초점·노이즈 변동 범위를 사전에 정하고 위치 변환은 이미지/마스크에 동일 적용한다. Eutectic Si 등 약한 클래스의 GT 오차와 클래스 균형을 우선 점검한다. 증강/GT+teacher 결합/학생 용량 선택을 같은 valid 최고 조합 탐색으로 반복하면 동일 문제가 재발한다.

## 90개 정답으로 가능한 평가 설계
새 데이터 없이 절대 보장 대신 선택 편향을 줄인 *재학습 절차 평가*를 할 수 있다. 시편 단위 outer split을 미리 고정한다. 같은 시편의 이미지와 패치는 한 그룹이며 이미지 단위 무작위 분할은 사용하지 않는다.

- outer 학습 시편만으로 teacher/student/모든 KoMaP upstream checkpoint와 priors/calibration/후처리를 학습·선택한다. 현재 frozen weights가 outer 시편을 학습/선택에 사용했으면 이를 그대로 써서 누수 없는 평가라고 부를 수 없다. KoMaP 데이터를 보지 않은 초기화에서 시작하는 학습 이력을 추적한다.
- inner 시편 분할에서 학습 epoch/loss와 고정된 소수 후보만 선택한다. OOF teacher soft labels를 사용한다면 outer 학습 범위 안에서 각 student 학습 시편을 teacher fitting에서도 제외해 생성한다.
- outer 시편에는 한 번만 추론하고 기준/학생의 paired 이미지 mIoU·클래스/시편별 손실·checkpoint bytes·반복 시간을 기록한다. 보고할 공식 점수는 전체 heldout 이미지별4IoU 평균의 이미지 평균이다. 시편 단위 재표본화 구간은 보조 진단이다.
- 독립 outer 평가를 보려는 동안 그 결과로 후보/비율을 다시 고르지 않는다. 기존90장으로 방법 설계가 이미 영향을 받았으므로 이런 CV도 완전히 새로운 독립 검증으로 포장하지 않는다. 이는 기존 base12 자체의 새 데이터 성적이 아니라 새로운 고정 재학습 절차의 평가다.
- 전체12개 teacher/base를 outer마다 재학습해야 한다면 계산 비용이 크다. 부분 증류의 작은 타당성 실험과 엄밀한 전체 비교를 구분한다. 기존 가중치/릴리스는 유지한다. test10은 정답 평가나 독립 holdout 점수로 사용하지 않는다.

## 근거
- Cawley & Talbot, model selection overfitting/selection bias: https://www.jmlr.org/papers/v11/cawley10a.html
- Hinton et al., distillation (hard/soft targets): https://arxiv.org/abs/1503.02531
- PyTorch numerical accuracy: https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html
- 로컬 데이터/계보 감사: generalization_20261008/report.md 및 data_exposure_audit.json.

이 문서는 후보와 평가 방식 제안이다. 신규 대규모 재학습이나 독립 일반화 개선을 완료했다는 보고가 아니다.
