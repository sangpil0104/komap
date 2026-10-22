lite15-N3 R326 EMA seed3461 교체 후보
R326 기존 2-fold를 seed3461 EMA 단일 가중치로 교체, R8+E6+F1=15. 기존 lite16-N3 보존.
ANA 후보 경로: /home/sangpil/komap/lite15_N3_R326EMA3461_candidate_20261013
원본부터 생성된 EMA 저장 확률을 사용해 CPU로 valid20 최종 마스크 재생성, 독립 GT 채점 및 기존 screen 20장 모두 일치. 전체 패키지 GPU fresh 추론은 아직 미실행.
per_image_both_baselines.csv에는 N3 이전 원본 lite16 및 현재 lite16-N3 두 기준이 모두 포함된다.
EMA 가중치 SHA256: 074c2bf55886853bcc88d220560ab587770666d5e3986bd8e2067a7f9c49d4d5
현재 모든 valid20 비교는 반복개발·선택에 사용한 검증셋 결과이다.
