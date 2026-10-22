# 저장 오류 정정

첫 CPU 선별은 fold1~3 학습과 fold4 17장 평가 계산을 끝냈으나 최종 JSON 직렬화에서 NumPy int64 값 때문에 종료됐다. 결과 JSON이 없으므로 첫 실행의 점수를 주장하지 않는다. 실패 로그는 `attempt1_serialization.log`에 보존한다.

변경은 JSON 인코더가 NumPy scalar를 Python scalar로 바꾸는 한 줄뿐이다. 입력·샘플·LightGBM 설정·혼합 비율·점수 산식·판정 기준은 그대로다. GPU와 valid/test 자료는 사용하지 않았다. 수정 source lock을 새로 확인한 뒤 CPU 선별을 한 번 다시 실행한다.
