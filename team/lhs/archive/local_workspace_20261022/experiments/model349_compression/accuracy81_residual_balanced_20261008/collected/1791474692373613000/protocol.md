# train70 클래스 균형 잔차 모델 후속 실험

이 프로토콜은 첫 잔차 후보의 raw valid20 독립검증 기각 후 작성한다. 같은 valid 반복 개발평가이며, 새로운 독립 일반화 증거를 주장하지 않는다. 목표81%는 아직 미달이다.

## 고정 비교

첫 잔차의 Al3Ni/PrimarySi/Eutectic 평균 악화와 Al 개선을 관측했지만 클래스 불균형을 원인으로 확정하지 않는다. 후속 후보에서는 같은 parent14, 학습용 raw train70 출력, 같은 CNN/zero-init/delta 상한1/seed81349/샘플러/증강/8epochs/1120updates를 사용하고 CE의 클래스 균형만 변경한다. 체크포인트 중간 epoch나 가중치/seed를 valid 점수로 선택하지 않는다.

train70 GT에서 이미지별 클래스 비율을 구하고 70장 평균 f_c를 계산한다. 고정 w_c=1/sqrt(f_c+1e-6)를 클래스 평균1로 정규화한다. 이미지 픽셀 수에 따른 특정 시편의 지배를 줄이려는 정의이며, valid/test에서 비율을 추정하지 않는다. 배치 CE는 sum(w_target * −logsoftmax_target)/sum(w_target)이다. .5 balancedCE+.5기존Dice+.05mean(delta²), strict CUDA 결정성, 최종epoch8만 평가한다. 클래스 순서[Al3Ni,PrimarySi,Al,EutecticSi]. train70에서 계산한 정확한 빈도/가중치/manifest SHA를 기록한다.

## 재사용과 검증 범위

원본 source/14 weights SHA와 기존 train70/valid20 field manifests 및 NPZ SHA가 일치하는지 확인한 뒤, 완료된 부모 예측을 읽기 전용으로 재사용한다. raw valid20 부모 출력은 이전 실험에서 원본 입력부터 재추론해 PNG/GT 독립채점한 것이지만, 이번 보정 후보 자체의 full fresh pipeline 재현으로 보고하지 않는다. 전체 시간은 재사용으로 추정하지 않고 보정 시간만 측정한다. 원본 입력부터 전체fresh 재현은81%관측시별도 실행한다.

strict deterministic balancedCE를 CPU/CUDA reference weighted CE의 값/gradient와 합성 자료로 비교하고 초기 zero-refiner20부모마스크exact를 확인한다. valid20 이미지별4IoU/emptyunion0/20평균을 PNG/GT로 독립 채점하고 클래스/이미지 차이를 모두 남긴다. 이는 train-only 학습이라도 같은valid를 반복 개발한 탐색이며 일반화 보장이 아니다.

## 실행 안전과 완료 기준

이전 잔차 컨트롤러의 종료를 실제ps로 확인하고 유휴GPU/compute app/기존 lock을 확인한다. 타사용자 점유시대기하고 충돌시우리자식만중단한다. source/weights/releases/dirty tree보존. 이 폴더에만 새코드/모델/산출물을작성하며 원본을덮어쓰지않는다.

이 문서는 아직 실행/학습/검증 성적이 아니다. 81% 달성 완료는 full fresh 재현,독립공식채점,원본불변,test10마스크/일치율(정답없음/test mIoU 없음),재현코드·설정·한국어결과가 모두 확인된 뒤에만 판단한다.
