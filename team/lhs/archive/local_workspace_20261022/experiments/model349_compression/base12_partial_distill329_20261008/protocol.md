# 329 4개→학생 1개: 고정 소규모 타당성 시험

## 범위와 한계
사용자가 추천안 실행을 요청했다. 단일 worker 최적화가 완료한 후 GPU7에서 순차 실행하며 중복/타 GPU 작업을 건드리지 않는다. 고정된 base12의 329 4개를 학생1개로 대체하는 9가중치 판이다. 그 외 326/HR336/269/고해상도/TENT5/2입력/D4/계수/후처리는 원래 base12 그대로 유지한다. 기존 학습 출력과 frozen tensor가 일치하지 않았으므로 교사는 배포 base12의 frozen FP16 4개를 직접 사용한다. 원본은 읽기 전용이다.

기존 교사가 train70 시편을 본 상태이며 valid 선택에도 노출되었다. 이 시험은 **증류 구현 및 성능 유지 가능성**을 보는 시험이고 독립 일반화 평가가 아니다. train-valid 중복 시편과 기존 선택 편향을 유지하며 일반화 개선을 주장하지 않는다. 정답 없는 test10을 학습에 사용하지 않는다.

## 결과 전 고정한 후보와 학습 조건
1. `init9`: 329 네 state_dict의 실수 tensor를 FP32에서 균등0.25 평균, 정수 buffer elementwise max. 동일 계보/구조라는 것만으로 출력 보존을 가정하지 않는다. 학생 초기값과 무학습 기준이다. fold 선택이나 비율 탐색 없음.
2. `hard9`: 같은 초기값에서 train70 정답 CE+Dice만400 optimizer updates.
3. `distill9`: 같은 초기값/400개 crop/optimizer 조건에서 CE+Dice + 0.2 * T² * KL, T=2. 교사의4개 softmax 확률 평균을 p^(1/T)로 temperature soften 후 재정규화한다. 학생 log_softmax(logits/T)에 대한 KL이다. 교사 eval+no_grad, 모든 parameter frozen.

학생/교사: 기존 EfficientNet-B4 U-Net, image_global/zcdf3, 가로4000. 학습 crop512, batch1, uniform image sampling, seed20261008. 전역 mean/std/CDF는 각 이미지의 가로4000 full image에서 계산하고 crop에 그대로 적용한다. crop+동일 D4를 이미지/GT에 함께 적용, 추가 조명 augmentation/클래스 가중치/구조변경 없음. AdamW lr1e-5, wd1e-4, grad clip1, FP16 autocast/GradScaler init1024. hard/distill 학생 train mode, 교사 eval mode. 고정 last checkpoint를 FP16 저장해 실제 배포 로드 조건에서 평가한다. epoch/valid best 선택 없음. 400 random updates는 전체 데이터 epoch와 같지 않다.

각 trial의 런타임 상한30분. 상한에 도달하면 현재 상태/RNG/optimizer/scaler를 보존하고 불완료로 표시하며, 검증 점수에 맞춰 budget/loss/seed를 수정하지 않는다. 고유 GPU7 compute 과정만 사용하며 외부 프로세스가 감지되면 우리 작업을 멈춘다. 처음5 updates loss/gradient/teacher 상태 불변 검사를 수행하고 의미 있는 오류가 있으면 학습을 계속하지 않는다.

## 평가와 후속 조건
각 후보의 329 분기를 원본 valid20 이미지부터 TTA identity로 추론한다. 다른4분기는 frozen base12의 원본 재추론 확률을 재사용해 초기 결합 진단을 한다. 이것은 캐시 재결합 진단이며 전체 모델 원본 재추론/시간 성적으로 부르지 않는다. 공식 image mIoU 정의로 GT 재채점하고 이미지/클래스 손실을 모두 기록한다. 원래 base12의329 확률을 넣은 진단이 기존 마스크/IoU를 정확히 재현하는지 먼저 확인한다.

사전 지정된 평가 대상은 distill9 하나이며 init9/hard9은 원인 분석용 control이다. distill9이 base12·기존lite15 각각 평균손실≤0.01%p, 최악이미지손실≤0.10%p를 모두 통과해야 전체9개 원본 valid20 재추론으로 진행한다. init/hard가 더 높다는 이유로 선택 대상을 바꾸지 않는다. 전체 재추론도 조건을 통과하면 기준12/후보9 각각3회 A-B/B-A/A-B 시간 및 test10 마스크 일치율을 측정한다. full inference startup/fusion 포함 중앙값10% 이상 단축, 모든 반복 이중 품질 통과, GPU7 외부 compute 간섭 없음이 시간 판정이다. 캐시/학습/단회 시간은 추론 개선 근거로 쓰지 않는다. test10에는 IoU/정확도 없음.

수치 실패 시 fold/비율/epoch/loss를 같은valid에 맞춰 추가 탐색하지 않는다. 모든 checkpoint/입력/코드SHA와 실패/부분 결과를 보존한다. 기존 base pointer/릴리스/가중치/미추적 파일은 바꾸지 않는다. 독립 평가를 하려면 모든 KoMaP teacher/student/calibration을 outer 학습 시편만으로 새로 학습하는 별도 절차가 필요하다. 이 pilot이 그 절차를 대체하지 않는다.
