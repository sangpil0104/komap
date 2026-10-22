# 80.2% exh14 방법의 P100 가중치 이식 비교

사용자 지적에 따라 릴리스 묶음의 80.2312313198%를 P100에서 학습한 동일 이름 모델의 성능으로 간주하지 않는다. 서버 `/workspace/model349_compress/Model349_exh14`의 14개 checkpoint와 `/workspace/komap_model1/outputs`의 대응 14개는 전체 텐서 key/shape/dtype이 맞으나 각 checkpoint의 텐서 값이 다르다.

단일 고정 후보: `Model349_exh14`의 추론 코드, 가로4000 전처리, 고해상도 route, B4 fold 선택(322:3·4,326:1·2,329:1·3·4), HR336/347 계수, TENT5, D4, 후처리를 그대로 사용한다. 14개 checkpoint 파일만 P100 outputs의 대응 경로로 교체한다. config `validation_image_miou`만 null로 하고 체크포인트 manifest의 SHA/bytes 및 출처를 갱신한다. 원본 두 묶음의 파일은 수정하지 않는다. valid/test 정답은 학습에 쓰지 않고 새 학습도 없다.

GPU3의 다른 사용자 compute, 메모리, util과 모든 sibling `gpu3.lock`을 확인한다. GPU가 세 번 연속 완전히 유휴일 때에만 한 번 실행한다. 외부 compute가 실행 중 나타나면 우리 추론 자식만 중지하고 실패를 기록한다. 기존 작업을 재시작하거나 타 사용자 프로세스를 종료하지 않는다. 서버의 원본 valid20 영상에서 fresh 추론하고 20 PNG/GT 80 IoU를 독립 채점한다.

주 비교는 **같은 80.2 방법의 릴리스 가중치판 80.2312313198% 대 P100 가중치판**이다. 보관 P100 Model349의 79.9095188739%는 다른 route/과거 캐시 구현의 참고값으로만 별도 기재한다. 그 차이를 가중치 또는 GPU만의 효과로 분해하지 않는다. test10에는 GT가 없어 점수 추정을 하지 않는다. valid20은 반복 개발에 쓰였으므로 독립 일반화 증거가 아니다.
