# 12개 base 추가 탐색: 상태 복원·HRNet 연산·무손실 저장·고해상도 가중치 평균

서버 GPU7과 CPU 점유를 재감사했다. GPU7 6MiB/0%, 기존 추론 없음, 기존 CPU 탐색 종료 뒤 load 0.19/0.54/4.40이었다. 다른 GPU 작업은 유지한다. GPU 매뉴얼 재확인, /workspace 별도 폴더, 기존 모든 GPU7 flock과 유휴 확인 후 단일 작업으로 실행한다.

결과 확인 전에 다음 후보와 조건을 고정한다. 기존 base/가중치/릴리스는 변경하지 않는다.

1. `soup11`: 같은 ASPP 구조의 seed456·pairwise 가중치를 **0.5:0.5**로 float32 평균 후 FP16 저장한다. 두 밝기 경로가 같은 합성 가중치를 사용하고 나머지는 그대로다. 정수 buffer는 두 값의 최대값을 사용한다. 원본 두 모델은 별도로 보존한다. 실수 가중치의 stem/encoder/context/decoder/head cosine이 0.9999 이상인 것을 CPU 확인했지만 출력 보존은 가정하지 않는다. 혼합 비율·다른 조합은 이 결과를 보고 바꾸지 않는다. fold 선택이나 학습은 없다. 합성 checkpoint에는 synthetic provenance를 기록하고 기존 학습 epoch/성능을 합성 모델의 것으로 주장하지 않는다.
2. `resident12`: 고해상도 모델 객체를 재사용하고 checkpoint의 원본 CPU state_dict를 캐시한다. **매 이미지의 TENT 직전에 모든 parameter·BN buffer·counter를 원본으로 strict 복원**, grad 제거, eval 및 requires_grad 상태 초기화, 새 Adam 생성 후 기존 TENT5를 수행한다. 적응 가중치를 다음 이미지로 넘기지 않는다. 각 이미지 끝 강제 gc/empty_cache 호출도 제거한다. 원본·밝기 보정 두 입력, D4 8방향, TENT5/lr, 모든 결합/후처리를 보존한다. GPU 순차 probe로 두 source 전환과 원본 복원, 변조한 BN/parameter가 초기화되는지 확인한 뒤 valid20을 실행한다.
3. `hramp12`: HRNet forward에 FP16 autocast를 적용하고 타일 batch 2→8로 고정한다. 모델은 FP32 로드, 기존 FP32 stitching을 유지하며 실제 convolution 출력 dtype을 기록한다. 앞선 15개판에서 시험한 변경을 현재 12개 base에 별도 검증하는 것으로 독립 일반화 검증이 아니다. 다른 배치 크기는 탐색하지 않는다.
4. resident12와 hramp12가 둘 다 아래 품질 조건을 통과하면 둘을 결합하고 모든 체크포인트 ZIP entry를 **DEFLATE level6**으로 무손실 압축한다. soup11도 품질 통과하면 `packed_fast11`, 아니면 `packed_fast12` 한 개만 실행한다. 모든 entry 해시 및 원본 대비 전체 저장 tensor·shape·dtype를 CPU 대조해 같아야 원본 valid20부터 검증한다. 가중치 평균은 soup11에만 해당하며 ZIP 압축 자체는 값 변화가 없다.

각 후보는 원본 valid20 재추론과 PNG/GT 독립 재채점으로 평가한다. image mIoU는 이미지별 4 IoU 평균 후 20장 평균, absent/empty=0이다. base12와 기존 lite15 각각 평균 손실 ≤0.01%p, 최악 이미지 손실 ≤0.10%p를 **모두** 통과해야 후속 후보/반복 대상으로 올린다. 클래스별·이미지별 변화도 기록한다. 검증 노출 탐색이며 일반화 개선을 주장하지 않는다.

반복 대상은 사전 순서 packed_fast11/packed_fast12 → soup11 → resident12 → hramp12에서 품질 통과한 첫 하나다. 기준/후보를 A-B, B-A, A-B 각3회 원본 valid20부터 실행한다. 모델 로드·fusion 포함 전체 시간 중앙값 10% 이상 감소, 모든 반복 이중 품질 통과, GPU7 타 compute 간섭 없음이 시간 개선 조건이다. 시간 측정 도중 별도 CPU 재채점/압축 작업을 시작하지 않는다. 체크포인트 bytes/count도 분리 비교한다. raw/probe 단회 시간은 개선 근거가 아니다.

반복 후보가 있으면 test10 원본 재추론 후 기존 base와 마스크 일치율만 기록한다. test 정답/IoU가 없다. 실패 결과에 맞춰 ratio/fold/TENT/계수/배치 크기를 재조정하지 않는다. 모든 실행·부하·실패·미실행 조건을 기록하고 원본 SHA 및 로컬 기존 파일 보존을 재확인한다.
