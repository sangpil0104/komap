# Model349 fusion805 — 구성 모델 재학습 결과

대표 결과: **1단계 R+E image mIoU 80.236507%**.

| 구성 | valid20 image mIoU |
|---|---:|
| 새로 학습한 R (exh14) | 80.051924% |
| 1단계 R+E (대표) | 80.236507% |
| 2단계 R+E+F (보존용 비교) | 80.267268% |

R 14개 + E 12개 + F 1개 = 총 27개를 각각 새로 학습했습니다.
R은 보존된 Model98·257에서 시작하는 미세조정 및 새 부모 연결을 이전 실험과 같이 유지했습니다.
E는 ImageNet에서 B4 4-fold, HRNet 2 seeds × 4-fold를 학습했습니다.
F는 새 E 모델의 train70 group OOF만으로 8 epochs/1,120 updates를 학습했습니다. valid/test 정답을 학습에 넣지 않았습니다.

지표는 각 이미지의 네 클래스 IoU 평균을 20장에 대해 평균하며 빈 클래스 IoU=0입니다.
배포 문서에 따라 1단계를 대표로 기록합니다. 이번 점수로 단계를 다시 고르지 않았습니다.
원본 fusion805 조합 개발과 R의 checkpoint 선택에 valid20이 쓰였으므로 독립 test 성능이 아닙니다.
GitHub 공개 가중치·점수의 동일 재현을 주장하지 않습니다. 위 값은 이번 새 가중치의 실제 검증값입니다.

GPU: 물리 2,3,4,5,6,7 / conda komap.
실행 원본·로그·모든 학습 결과: `/Data3/sppark/komap/experiments/model349_fusion805_retrain_gpu234567_v1/outputs/fusion805_train_20261009_164635_301741`

`valid_mask_comparison/`: 1단계 20장. `stage2/valid_mask_comparison/`: 2단계 20장.
좌상 원본 / 우상 오류 / 좌하 예측 / 우하 정답. 정답 위치는 흰색, 오류 위치는 정답 색.
각 이미지 위에 image mIoU와 100%−image mIoU를 표시했습니다.
`selected_model/`: 새 가중치 27개와 실행 소스, 새 SHA-256 manifest.
`source_snapshot/`: 실제 학습·실행 코드 사본.

| 이미지 | 1단계 image mIoU | 2단계 image mIoU |
|---|---:|---:|
| A117_step4_sample2 | 86.905225% | 86.885943% |
| A11_step1_sample4 | 80.382005% | 80.370580% |
| A11_step2_sample4 | 77.356787% | 77.390972% |
| A125_step1_sample4 | 79.379401% | 79.384025% |
| A153_step3_sample3 | 71.376994% | 71.448197% |
| A170_step4_sample3 | 88.766043% | 88.749281% |
| A181_step1_sample1 | 79.278000% | 79.287796% |
| A186_step2_sample2 | 81.059065% | 81.059239% |
| A191_step4_sample3 | 82.552131% | 82.551927% |
| A40_step3_sample3 | 82.500437% | 82.504580% |
| A42_step2_sample4 | 81.548944% | 81.552627% |
| A44_step4_sample3 | 90.129810% | 90.168797% |
| A55_step2_sample1 | 82.034492% | 82.063179% |
| A59_step3_sample1 | 82.975325% | 83.098620% |
| A63_step1_sample4 | 84.531321% | 84.762849% |
| A67_step1_sample1 | 75.641339% | 75.734808% |
| A71_step2_sample1 | 54.171679% | 54.197899% |
| A72_step4_sample4 | 86.847702% | 86.826974% |
| A74_step3_sample4 | 74.820515% | 74.838632% |
| A7_step1_sample4 | 82.472935% | 82.468445% |
