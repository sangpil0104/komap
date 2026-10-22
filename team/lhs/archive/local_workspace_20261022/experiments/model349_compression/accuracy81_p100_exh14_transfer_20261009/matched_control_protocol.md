# 동일 P100 가중치의 원래 Model349 route 대조군

주 이식 후보와 같은 `Model349_exh14` 실행기와 valid20 원본 입력을 사용하되, P100 Model349의 기존 route를 결과 JSON에서 복원한다. `Model339` base route, `Model341` gamma329, `Model346` gamma336, `Model349` gamma347을 그대로 적용한다. B4 322/326/329는 각각 4-fold 모두 사용하고, 고해상도 EMA 후보도 사용한다. 대응 P100 출력의 총20checkpoint를 SHA/bytes로 고정한다. `b4mean`은 원래 route에서 사용한 322/326 확률의 50:50 평균을 실행기에 한 줄 추가한 것이다.

먼저 14가중치 이식 후보가 정상 종료한 뒤 같은 GPU3에서 순차 실행한다. 모든 sibling gpu3 lock, GPU3 메모리·util·compute를 확인해 세 번 연속 완전 유휴일 때만 시작한다. 외부 compute가 나타나면 우리 자식만 종료하며 자동 재실행하지 않는다. 원본 묶음과 P100 출력은 수정하지 않는다.

과거 P100 Model349 캐시 점수 79.9095188739%는 control 재현성 참고치다. 새 control은 다른 실행기에서 fresh 추론하므로 작은 수치 차이가 있을 수 있다. 차이가 크면 일치 원인을 감사한 뒤 route의 순수 효과로 해석하지 않는다. 이식 후보와 control의 PNG/GT 20장80IoU를 각각 독립 채점하고 동일 이미지별 차이를 비교한다. 사용한 valid20은 여러 번 개발에 쓰였으므로 독립 일반화 근거가 아니다. test 정답은 사용하지 않는다.
