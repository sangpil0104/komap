# Model369 후보: Al↔Al3Ni 경계 정보 이전 — 평가 전 보류

validation 점수를 계산한 Model369는 실행하지 않았다. train-OOF에서 보정한 국소 Al→Al3Ni 수정안을 validation label을 열기 전에 기각했다. 보정 데이터와 적용 대상 route가 서로 다른 모델 계열이기 때문이다.

## 판단 근거

- Model368에서 보인 Al/Al3Ni 면적 차이는 Model329 B4 계열 train-OOF 예측에 해당하며, 특정 source 모델의 오류이지 보편적인 금속 경계 편향이라고 볼 수 없다.
- Model360의 Al 결정은 주로 HRNet336/347을 사용한다. 반면 Al3Ni는 B4 및 Model270 조합을 포함한 Model341/339 route를 따른다. 따라서 Model329의 raw softmax에서 맞춘 임계값 `tau`는 Model360의 최종 Al 대 Al3Ni margin을 보정하지 않는다.
- B4/hybrid OOF source를 여러 개 추가해도 이 불일치는 해결되지 않는다. 이 source들은 같은 데이터와 fold를 공유하고 HRNet Al OOF 예측이 없다. 이 단일 이전을 위해 HRNet OOF를 새로 만들면 약 30–36 P100 GPU-hours가 필요할 것으로 Claude Code가 추정했으며, Model270/prior 항의 차이도 남는다. 기대 개선 약 `0.06%p`에 비해 비용이 크다.
- 과거 전역 Al-logit sweep에서 0 bias가 선택된 점도 Model360 Al 쪽이 일관되게 과대예측한다고 가정하기 어렵게 한다. 현재 증거는 Model360에서 국소 보정 방향을 결정해 주지 못한다.

## 결정 및 다음 진단

Model329에서 구한 `tau`를 Model360에 적용하지 않고, 이 보정만을 위해 HRNet OOF를 추가 생성하지 않는다. 대신 validation mask를 열지 않는 [label-free 사전 점검](../experiments/model369_boundary_source_distribution_audit.py)을 실행했다. 이 코드는 사용 가능한 B4/HRNet 확률원에서 Al/Al3Ni 점유율과 pairwise 결정을 비교한다. Model360 쪽 Al3Ni 근사는 Model322/326 평균을 사용하므로 명백히 맞지 않는 transfer는 걸러낼 수 있지만, 성능 개선을 증명할 수는 없다. mIoU도 계산하지 않는다.

20장 사전 점검에서 proxy의 Al-win 비율은 raw Model329보다 17/20장에서 낮았고, paired 차이 중앙값은 `−2.532%p`, pairwise 결정 불일치 중앙값은 `7.056%`였다. 두 pairwise log-probability margin의 이미지별 공간 상관은 높았다(중앙값 `0.9696`). 이는 넓은 순위 패턴은 비슷하나 임계값/source offset이 의미 있게 다를 수 있음을 시사한다. B4에서 맞춘 픽셀 이동이 HRNet/B4 route를 개선한다는 증거는 아니다.

이 문서는 모델 선택 및 연산 배분 결정이며 성능 결과가 아니다. 현 저장소의 최고 raw valid 관측치는 Model364 `79.9429727%`; 보수적 기준 route는 Model360 `79.9416620%`다. 최신 점수·split 감사는 [한국어 최신 현황](current_status_ko.md)을 참고한다.
