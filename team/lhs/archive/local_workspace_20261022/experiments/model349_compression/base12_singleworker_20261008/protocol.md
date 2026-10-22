# 조건부 단일 GPU worker 후속 구현 실험

앞선 resident/HRAMP/무손실 압축 12개 실험의 종료·원본 보존·품질·시간 결과를 먼저 확인한다. 앞선 반복 중앙값 단축 ≥10%, 모든 반복 이중 품질 통과, GPU7 타 compute 간섭 없음이면 이 단계는 **실행하지 않는다**. 앞선 결합판이 원본 valid 품질에 실패하면 이 단계도 실행하지 않는다.

앞선 결합판 품질은 통과했으나 시간 기준을 충족하지 못했을 때만 같은 가중치/압축/연산/예측/TENT/결합/후처리를 유지하고 5개 component를 **하나의 GPU subprocess에서 기존 순서대로** 실행한다. 두 경로를 동시에 실행하지 않는다. 새 GPU/CUDA context와 Python import를 경로별로 반복하는 비용을 줄이는 구현 실험이다. 가중치 수는 여전히12개, ZIP 파일도 앞선 결합판과 정확히 같다.

각 component의 모델은 함수 범위 종료 후 해제한다. base component 종료 때 resident GPU model과 CPU source cache도 지운다. 경계에서 gc 및 CUDA empty_cache를 수행하되 단일 CUDA context를 유지한다. 이미지마다 원본 복원과 새 TENT optimizer는 그대로다. 각 경로 로그/complete와 확률 산출을 별도로 보존한다. 이 구현은 GPU 한 개만 허용한다. 기존 controller/다른 사용자의 작업은 변경하지 않는다.

원본 valid20 재추론 후 base12·기존 lite15 각각 평균 손실 ≤0.01%p, 최악 이미지 손실 ≤0.10%p를 모두 통과해야 각3회 A-B/B-A/A-B 전체 시간 비교 및 test10을 진행한다. 첫 후보는 앞선 packed valid의 component 확률과도 비교하여 고해상도/TENT 수치 변동 및 HRAMP/다른 component 차이를 기록한다. 재추론/PNG 독립 재채점은 일반화 증거가 아니다. 계수/fold/TENT/방향/배치 크기를 추가 조정하지 않는다.

기준은 원래12개 base의 새로운 control12로 고정한다. 전체 시간 중앙값10% 이상 감소·모든 반복 이중 품질 통과·GPU7 타 compute 간섭 없음이면 탐색 조건의 시간 개선으로 판정한다. 앞선 packed판 대비 시간도 참고 표시하며 직접 교차한 인과 비교로 주장하지 않는다. 시간 측정 도중 다른 자체 CPU 분석 작업을 시작하지 않는다. test10은 정답 없는 base 마스크 일치율만 확인한다.

모든 이전 GPU7 lock과 새 lock을 확보하고 유휴 상태 및 기존 inference 종료를 확인한다. 이전 base/릴리스/체크포인트를 덮어쓰지 않는다. 별도 source hash/manifest/부하/실패/완료 기록을 보존한다. 원본 local 파일2024개와 server base 해시 보존을 재확인한다.
