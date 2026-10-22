# 조건부 HRNet autocast 보완 실험

D4batch4의 첫 base component는 105.98초였고 첫 기준은 115.30초였다. 전체 valid 점수와 반복 시간은 아직 확정 전이다. 15-FP16/OpenCV2/D4batch4가 품질 기준을 통과하되 반복 중앙값 시간 감소 10% 기준을 못 넘을 경우에만 다음 안을 추가 검증한다.

- 전체 15개 checkpoint와 EMA, TENT 5-step, raw/style, B4 모든 fold, 269, 혼합 계수와 후처리를 보존한다.
- HRNet의 기존 FP32 forward를 `torch.autocast('cuda', dtype=torch.float16)` 범위에서 수행한다. checkpoint는 여전히 기존 FP32 module로 로드하며, 기존 FP32 합산/타일 stitching을 유지한다. HRNet 타일 batch_size는 2→8로 고정해 동시에 처리하는 타일을 늘린다. BN eval과 타일 합산 순서는 유지하며, 두 변경을 합친 파이프라인 효과를 평가하고 AMP 단독 효과로 주장하지 않는다. 이는 FP16 저장 효과와 구분되는 실제 연산 정밀도 변경이다.
- view batch=4는 그대로 유지하며 고정한 HRNet batch=8 이외의 배치 크기·혼합 계수·fold를 탐색하지 않는다.
- P100 GPU7에서 새 lite15 기본판과 후보를 A-B-A-B-A-B 각 3회 원본 valid20 재추론한다. 첫 후보 또는 후속 후보의 평균 손실 >0.01%p, 최악 이미지 손실 >0.10%p이면 기각하고 반복을 중단한다. 통과한 3회 중앙값 시간 10% 이상 감소와 bytes 감소가 있어야 탐색 채택 기준 통과다.
- test10은 정답 없이 기존 lite15 마스크 일치율만 확인한다. valid20 재실행은 독립 일반화 자료가 아니다.
- 별도 복사한 신규 코드·설정·출력 경로만 사용한다. 현재 D4batch4 실행을 바꾸지 않고, 기존 릴리스·원본 가중치도 수정하지 않는다. 빈 GPU 확인과 flock을 유지한다.
