# 사용자 지정 exh14 base 전환 및 고정 후속 탐색

사용자의 명시적 요청으로 base12 대신 최고 구성 `(12,3,13,1,1,1,0,1,7)`을 후속 base로 지정한다. 기존 Model349_exh14 원본 FP32 14파일/1,755,428,956바이트를 보존한다. 322 f3·4 / 326 f1·2 / 329 f1·3·4 / HR336·347 / 269 / high4개이다. 이전12개 base 기준 최악 손실 gate를 넘었던 사실은 유지하지만 사용자 지정에 따라 새로운 비교 기준으로 삼는다. 과거 보고서를 소급 변경하지 않는다.

기존 실행 `/workspace/model349_compress/test_compare_20261008/exh14_valid20` 및 `exh14_test10`이 완료되어 중복 실행하지 않았다. 원본 valid20 image mIoU80.2312313%, PNG/GT 독립 재채점 확인. test10에는 정답 점수 없음. valid 시간260.51초는 GPU3,4 두 개/단회이며 과거 GPU7 한 개의 시간과 속도 비교하지 않는다. frozen 데이터/코드/가중치 SHA를 확인하고 자체 단일GPU7 기준 실행을 새로 만든다.

결과 전 고정한 후보:
1. `fp16_storage14`: 모든 floating checkpoint tensor를FP16으로 저장하고 로드 후FP32로 계산. integer/기타 metadata 보존. 저장후 tensor==원본.half()와 nonfloating tensor 정확 일치를 확인한다. ZIP DEFLATE6으로 추가 압축하며 ZIP entry/tensor는 FP16 저장판과 정확히 동일해야 한다. FP32→FP16은 근사, 그 이후ZIP만 무손실이다. 연산/VRAM 개선으로 주장하지 않는다.
2. 위 후보 품질을 통과하면 `fast14`: 같은 압축된14파일에 매 이미지 모든 parameter/buffer/counter 복원 + 새 TENT optimizer, HR336/347 FP16autocast+tilebatch8, 모든7 component를 단일GPU subprocess로 기존순서대로 수행. TENT5/D4/raw·style2입력/계수/후처리/선택fold는 유지한다. GPU 상태변조후복원 probe와 rawvalid20을 확인한다. HRAMP는 모든입력 출력 동등 보증이 아니다.

각 후보는 새14base 대비 평균손실≤0.01%p, 최악이미지손실≤0.10%p가 사전품질조건이다. base12/lite15/original349는 보조 비교로 기록하며, 새base자체가 과거base12gate를 넘으므로 예전 이중gate를 그대로 적용하지 않는다. 이 변경은 사용자 base 전환에 따른 기준 정의이지 후보 결과에 따른 threshold 조정이 아니다.

공식 metric은 이미지마다4개IoU 평균 후20개이미지 평균, empty union=0. GPU7 한 개, torch/OpenCV2, 모든 실제입력/가중치/로드조건/GT/로그 보존. 기존 추론/외부GPU7 compute/기존lock가 있으면 실행하지 않는다. 프로세스가 없고memory≤64MiB/util0인 상태를3회 연속 확인한다. 외부 작업 종료 금지. 메타데이터와 실제 source hash로 다른 작업과 구분한다.

기준control14와 품질통과fast14(실패하면fp16_storage14)를 각각3회 A-B/B-A/A-B 시간측정한다. 전체 추론 로드/fusion 포함 중앙값10% 이상 단축, 모든 반복 품질 통과, GPU7 다른 compute 간섭 없음이 시간판정이다. 시간측정중 자체CPU 재채점/압축/학습은 하지 않는다. 후보 test10은 base14/base12/원본 마스크 일치율만 확인한다. 기각된 후보에 맞춰fold/ratio/TENT/배치/loss를 탐색하지 않는다.

다음 모델수 축소 연구는 새base의329 **3개→학생1개**, 14→12개다. 이전 base12의4→1/12→9 시험은 실행 전에 취소하고 소스/프로토콜을 보존했다. 새 증류 프로토콜에서는 같은3개 균등초기값, GT control, GT+teacher400 updates 고정비교로 다시 정의한다. 기존교사사용pilot은 독립 일반화 검증이 아니며 학습계보/시편 노출 한계를 유지한다.

원본/이전 릴리스/기존 미추적파일/과거base문서를 덮어쓰지 않는다. 포인터변경은 previous_base_pointer.json에 이전내용을 보존한다. 사용자 지정 새base이지만 동일valid 선택편향/누수 있는 global shortlist LOSO 한계/독립 일반화 미확인 판정은 유지한다.
