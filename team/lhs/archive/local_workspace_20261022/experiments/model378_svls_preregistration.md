# Model 378 사전등록: 공간 가변 soft target CE (SVLS)

## 상태와 목적

- 상태: **사전등록 초안, 미실행**. A100에서 진행 중인 Model377의 4-fold grouped OOF가 완전히 끝나기 전에는 시작하지 않는다. GPU 학습을 겹쳐 돌리지 않는다.
- 목적: 학습 오류가 GT 경계에 집중되는 현상에 대해, 경계 픽셀의 CE target을 이웃 클래스와 공간적으로 완만하게 섞는 것이 hard target CE보다 나은지 단 한 번의 고정 grouped-OOF 실험으로 검증한다.
- 근거/주의: Model374 OOF에서 오류의 88.6%가 GT 경계 1px band에 있었으나, SVLS는 경계 CE 가중치와 상호작용할 수 있고 과도한 target 완화는 얇은 Eutectic Si의 예측을 줄일 수 있다. train.zip의 70개 GT mask만 대상으로 한 CPU 사전점검에서는 아래 커널 적용 뒤 GT의 strict argmax가 바뀐 픽셀은 0개, 동률 픽셀은 각 클래스별 0.0003–0.0090%였다. 이는 target 보존 점검일 뿐 모델 성능의 증거가 아니다.

## 비교 기준 선택 규칙

Model377의 결과를 확인한 뒤 아래 규칙만 적용한다.

1. Model377이 기존 사전등록 mIoU 채택 gate와 directional mechanism gate를 모두 통과하면 Model377 recipe를 Model378의 parent로 한다.
2. 그렇지 않으면 Model374 recipe를 parent로 한다. 미채택 Model376을 별도 가중 실험 없이 parent로 삼지 않는다.
3. Model378에서 parent 대비 바뀌는 것은 아래의 CE target smoothing 하나뿐이다. 선택 규칙은 실험 전에 고정하며, 공식 valid/test 결과로 parent나 설정을 바꾸지 않는다.

## 고정된 방법

- Model374와 같은 specimen-group 4-fold CV, fold, seed, 70 epochs, optimizer, sampling, augmentation, patch/input scale, boundary head와 평가 코드를 사용한다. Model377이 parent가 되면 그 directional loss 설정도 그대로 유지한다.
- 기하 augmentation 적용 후, CE에만 3×3 Gaussian one-hot spatial smoothing을 적용한다. sigma=1.0 (학습 patch 좌표계)로 고정한다. Dice target, boundary-head target 및 (parent가 Model377일 때) directional-loss mask는 원래의 hard GT를 그대로 쓴다.
- 커널은 raw Gaussian을 합 1로 정규화하고 center weight를 주변 8개 weight의 합으로 바꾼 뒤, 전체 합으로 다시 나눈다. 최종 center weight는 0.5이고 나머지 0.5는 주변에 분배된다. 경계 바깥 padding은 replicate로 고정한다. CE는 soft target과 `log_softmax`의 곱을 클래스 축 합산하고, 기존 parent의 CE pixel weighting이 있으면 같은 pixel weight를 그대로 적용한다.
- smoothing kernel, sigma, weight, epoch, checkpoint 선택을 탐색하지 않는다. 결과는 원본 hard GT 및 empty-class IoU=0의 공식 image-wise mean IoU로 채점한다.

## 사전 고정 판정 기준

모든 비교는 70장 train OOF의 같은 이미지에 대한 paired 비교이고, bootstrap 단위는 59개 specimen group이다.

**성능 채택 gate — parent 대비 모두 충족해야 함**

- image-wise mean mIoU 증가가 `+0.50%p` 이상.
- 4개 fold 중 최소 3개에서 mIoU가 증가하고, 가장 나쁜 fold의 변화가 `−0.30%p`보다 낮지 않음.
- specimen-group paired bootstrap 95% CI의 하한이 0보다 큼.
- 어떤 클래스 평균 IoU도 `−0.50%p`보다 크게 하락하지 않고, Step 4 평균 mIoU도 `−0.30%p`보다 크게 하락하지 않음.

**작동기전 gate — 모두 충족해야 함**

- Model374 감사 코드의 `boundary_band(truth, radius=1)` 정의(4-neighbor GT edge를 Chebyshev radius 1로 팽창)에서 pixel error rate가 parent보다 상대적으로 3% 이상 감소하는 fold가 4개 중 최소 3개.
- GT Eutectic Si 중 다른 클래스와 4-neighbor로 맞닿은 픽셀의 recall이 어느 fold에서도 parent보다 `0.50%p` 초과 하락하지 않음. 이는 얇은 Eutectic phase 과소분할을 막기 위한 guard다.

하나라도 실패하면 Model378은 미채택한다. 실패 후 valid 점수를 보고 kernel/sigma를 조정하거나 같은 CV에 반복 탐색하지 않는다.

## 후속 평가 및 보고

OOF 두 gate를 통과할 때만 고정 설정으로 전체 70장 재학습을 한 번 수행하고 valid 20장은 최종 확인용으로 한 번 평가한다. valid 수치는 OOF 채택 여부나 설정을 재선택하는 데 쓰지 않으며, test 10장과 test label은 학습·선택·개발 평가에 사용하지 않는다. 보고에는 전체/fold/class/Step별 mIoU, group-bootstrap CI, 경계 band 오류율, Eutectic 경계 recall, checkpoint와 source hash를 포함한다.

## 대안 검토: 기존 4개 fold 모델 앙상블은 OOF 점수로 채점하지 않음

Claude Code Opus가 단순 fold-ensemble을 제안했지만, 각 train 이미지마다 4개 fold 모델 중 3개가 그 이미지로 학습되었으므로 4개 예측을 모두 평균하면 OOF 누수가 생긴다. 올바른 2-seed ensemble OOF에는 같은 4개 outer split마다 동일한 52장으로 두 번째 seed 모델을 추가 학습해야 하며, 각 held-out 이미지에서는 그 fold의 두 seed만 평균해야 한다(4회 추가 학습). 이는 SVLS 4-fold 1회와 비슷한 계산량이나 기대 개선폭·seed 안정성은 확인되지 않았다. 우선 사전등록된 SVLS 한 번만 시험하고, 현재 Model377 학습과 겹쳐 어떤 후보도 실행하지 않는다. 기존 fold 예측을 섞어 성능을 주장하지 않는다.

## 검토 출처

- Islam & Glocker, [Spatially Varying Label Smoothing: Capturing Uncertainty from Expert Annotations](https://arxiv.org/abs/2104.05788).
- 저자 공개 구현: [mobarakol/SVLS](https://github.com/mobarakol/SVLS). 이 실험은 위 구현의 공간 Gaussian 아이디어를 2D CE에 한정해 고정 적용하며, 논문 성능이 금속조직 데이터에 재현될 것이라고 가정하지 않는다.
