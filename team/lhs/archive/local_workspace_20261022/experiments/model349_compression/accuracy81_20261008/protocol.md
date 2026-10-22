# 81% 목표: base14 전역 보정 탐색

2026-10-08 결과 확인 전 후보 고정. 원본 base14/기존 실험/타 사용자 작업 보존.

목적은 정확도이며 속도·가중치 수 감소를 채택 기준으로 삼지 않는다. 첫 단계는 기존 valid 저장 확률 재생이다. 14개 fold/라우팅은 base14 고정, 모든 영상에 동일한 전역 계수만 적용한다. 이미지 이름/GT로 라우팅하거나 마스크를 수정하지 않는다.

고정 grid: prior_scale 0.75/1/1.25/1.5/1.75, al_delta -0.05/0/0.05, eut_delta -0.05/0/0.05/0.1/0.15/0.2/0.25/0.3 = 120 후보. beta_scale/공간 필터/cleanup/TENT/가중치는 유지한다. base14 원래 설정 포함. baseline 재생이 이전 replay IoU/마스크와 정확히 일치해야 탐색 결과를 인정한다. 필터 결과 재사용은 고정 bias의 선형성에 의한 계산 최적화이며 선정 후보는 원래 predict()로 전부 재검증한다.

공식 측정은 이미지별 4-class IoU 평균, 빈 union=0, 다시20장 평균. 클래스/이미지별 변화 및 최악 손실 공개. 전체 고정 grid에서 specimen 하나 제외하고 나머지로 선택한 평가와 cluster bootstrap을 진단으로 기록한다. base14와 valid 자체가 이미 반복 선택되어 있으므로 독립 일반화 검증으로 주장하지 않는다.

선정 후보는 탐색과 원래 predict 정확 재생이 일치한 경우 별도 YAML로 동결하여 원본 이미지부터 재추론하고 PNG/GT 독립 재채점한다. 캐시 결과로 81% 달성을 선언하지 않는다. test GT 없음, test mIoU 주장 금지. train/valid/test 정답 혼합 학습 없음. 모델 파일/원본 릴리스 덮어쓰기 없음.

공식 확인: https://komap.ai/front/competition2026/process1 (70/20/10, 제공 데이터 augmentation 허용), https://komap.ai/front/competition2026/award_info (mIoU50점), https://komap.ai/front/competition2026/apply_info (재현 코드/환경과 test10 마스크 제출). 공개 페이지로 외부 데이터 허용 여부/valid 학습 허용 여부를 확인하지 못했으므로 해당 방법은 사용하지 않는다.
