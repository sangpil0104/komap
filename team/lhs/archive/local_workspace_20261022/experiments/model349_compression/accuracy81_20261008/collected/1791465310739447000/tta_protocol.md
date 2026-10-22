# 정확도 탐색 2: 대칭 TTA

2026-10-08 결과 전 고정. 전역 보정120개 최고는 원래base14와 동일해 보정계수를 바꾸지 않는다.

HRNet336/347을 identity에서 flip4로 바꾼 hrflip4_14를 원본valid20부터 재추론한다. 가중치14개, 기존fold/고해상도D4/TENT5/전처리/융합계수/후처리 모두 유지한다. YAML의 tta_mode만이 아니라 enabled 플래그가 실제 tta=True 전달로 연결되도록 별도entry에서 구현한다. 원래entry/가중치 유지. CPU에서checkpoint/소스SHA를 전후 확인한다.

HR 후보 raw score가81% 미만이면 별도 b4d4_14를 실행한다: B4 322/326/329만 identity에서D4 8방향, HRidentity. 이어서 combined_tta14를 실행한다: B4 D4와HRflip4 모두 적용. 후보가81% 이상이면 추가TTA 확대를 중단하고 독립PNG/GT 재채점 및test마스크 생성으로 검증한다. 최고점 선정은 반복사용valid 탐색이며 일반화보장하지 않는다. 세후보는 결과전 순서를 고정한다. 속도개선주장 없으며 실제시간만보고한다.

GPU7만 사용하며 기존 GPU7 실험lock을 모두 획득, memory<=64MiB/util=0 및compute process 부재 3회 확인 후 시작. 매컴포넌트 전 원래idle guard 유지. 타사용자작업에 신호하지 않는다. 다른GPU에 실행 중인 모델410류 작업을 복제하지 않는다. 실패시원인기록/부분산출물보존, 자동중복재실행없음.
