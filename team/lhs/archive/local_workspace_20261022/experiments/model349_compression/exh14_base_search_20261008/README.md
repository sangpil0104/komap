# 사용자 지정 exh14 base 후속 탐색

새 기준: Model349_exh14, 원본 FP32 14개, 공식 valid20 image mIoU80.2312313%, 1,755,428,956바이트. source/weight manifest와 원본 PNG/GT 재채점을 감사했다. base12는 비교용으로 유지한다.

- base14_audit.json: 원본 재채점·base12 클래스/이미지 손실·test10 일치율.
- previous_base_pointer.json: 변경 전 포인터 원문. 현재 포인터는 ../model349_base.json.
- protocol.md/run_search.py: GPU7 단일 기준 → FP16+ZIP 저장14 → 조건부 상태복원/HRAMP/단일worker14 → 각3회 교차 시간 → test10.
- finalize.py: 전체 PNG/GT 재채점, 클래스/이미지별손실, 반복 시간, test마스크 비교, checksum 아카이브.
- 신규 후보를 실제 채택하는 단계와 사용자가 지정한 base전환을 구분한다. 독립 일반화 미확인이다.

기존GPU3,4의 valid와GPU6 test결과를 수집했다. valid260.51초는2GPU/단회이므로 단일GPU기준과 비교하지 않는다. test정답 없음. 원본성적이 오프라인80.23098%와 가까워도 같은valid선택편향은 사라지지 않는다.

기존 base12 단일worker raw80.12535%/302.47초는 참고 단회이며 반복측정 중 사용자기준전환으로 우리controller/process group만 중단했다. 완료/부분로그 보존, 타 사용자프로세스는 유지했다. 이전 base12 증류pilot은 학습을 시작하기 전에 취소했다.

후속증류는 새base의329 fold1·3·4 세 모델→학생1개, 14→12개로 다시 정의해야 한다. 이전4개→1개/12→9개의 코드를 그대로 실행하지 않는다. 새로운평가와일반화주의조건은protocol.md 참조. 아직 이 새증류 학습을 실행하지 않았다.
