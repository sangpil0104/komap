# Model349 fusion805 lite20 B4ws — 전체 재학습 결과

대표 결과: **lite20 B4ws R+E+F image mIoU 80.255846%**.

| 구성 | valid20 image mIoU |
|---|---:|
| 새로 학습한 R (lite20) | 80.010135% |
| 1단계 R+E (비교) | 80.227734% |
| 2단계 R+E+F (요청 모델) | 80.255846% |

GitHub lite20 B4ws 고정 설정, S=0으로 총 35개 학습 작업을 완료했습니다.
선행 Model6·62·97·98·109·194·195·239·242·257도 새로 학습했습니다. 기존 학습 가중치를 시작점으로 재사용하지 않았습니다.
공개 ImageNet encoder 초기화에서 시작하며, R과 E의 동일 B4-329 fold 1·3·4는 한 번 학습한 가중치를 공유합니다.
E는 원래 B4 4-fold와 HRNet 2 seeds × 4-fold를 학습하고, B4 4개를 같은 fold train으로 2,000회 추가 학습했습니다.
B4 추가 학습은 seed 819349, lr 3e-5, BN running 통계 고정입니다. 각 fold에 통계 불변 검사 결과가 있습니다.
최종 E는 B4ws 0.5 + HRNet seed1 fold2·4 평균 0.25 + seed2 fold1·4 평균 0.25입니다.
F는 추가 학습 전 E 12개 모델의 train70 group OOF만으로 8 epochs/1,120 updates 학습했습니다.
최종 저장 가중치는 R11 + B4ws4 + HR4 + F1 = 20개입니다. 나머지 중간 가중치는 실행 폴더에 보존됩니다.

지표는 각 이미지의 네 클래스 IoU 평균을 20장에 대해 평균하며 빈 클래스 IoU=0입니다. global mIoU는 계산하지 않습니다.
원본 조합 개발과 R checkpoint 선택에는 valid20이 쓰였습니다. 이 점수는 독립 test 성능이 아닙니다.
프로젝트 지침에 맞춰 과거 일부 R 학습기의 absent-class 제외 선택 지표도 image mIoU(empty=0)로 통일했습니다.
GitHub의 공개 점수 80.5301323%는 이전 고정 가중치의 개발 결과입니다. 위 결과는 이번 새 가중치의 실측값입니다.
test10은 정답이 없어 예측 마스크만 저장했으며 test mIoU를 계산하지 않았습니다.

GPU: 물리 2,3,4,5,6,7 / conda komap.
실행 원본·로그·모든 학습 결과: `/Data3/sppark/komap/experiments/model349_lite20_b4ws_retrain_gpu234567_v1/outputs/lite20_b4ws_train_20261011_130221_265591`

`valid_mask_comparison/`: 요청 모델의 20장. `stage1/valid_mask_comparison/`: F 적용 전 비교 20장.
좌상 원본 / 우상 오류 / 좌하 예측 / 우하 정답. 맞은 픽셀은 흰색, 오류 픽셀은 정답 색.
각 이미지 위에 image mIoU와 100%−image mIoU를 표시했습니다.
`selected_model/`: 20개 새 가중치·추론 코드·SHA-256 manifest.
`test_predictions/`: test10 예측. `source_snapshot/`: 실제 학습·실행 코드 사본.

| 이미지 | 1단계 image mIoU | 2단계 image mIoU |
|---|---:|---:|
| A117_step4_sample2 | 86.839814% | 86.804790% |
| A11_step1_sample4 | 80.307787% | 80.274949% |
| A11_step2_sample4 | 77.110169% | 77.129249% |
| A125_step1_sample4 | 79.238719% | 79.238624% |
| A153_step3_sample3 | 71.932621% | 72.006914% |
| A170_step4_sample3 | 88.951935% | 88.937243% |
| A181_step1_sample1 | 79.239618% | 79.249941% |
| A186_step2_sample2 | 81.029900% | 81.031879% |
| A191_step4_sample3 | 82.514127% | 82.508017% |
| A40_step3_sample3 | 82.547848% | 82.538612% |
| A42_step2_sample4 | 81.636950% | 81.641980% |
| A44_step4_sample3 | 90.147267% | 90.198297% |
| A55_step2_sample1 | 82.055871% | 82.083523% |
| A59_step3_sample1 | 82.949746% | 83.079121% |
| A63_step1_sample4 | 84.466935% | 84.749904% |
| A67_step1_sample1 | 75.667588% | 75.744696% |
| A71_step2_sample1 | 54.205846% | 54.235486% |
| A72_step4_sample4 | 86.809238% | 86.766041% |
| A74_step3_sample4 | 74.470727% | 74.478550% |
| A7_step1_sample4 | 82.431970% | 82.419093% |
