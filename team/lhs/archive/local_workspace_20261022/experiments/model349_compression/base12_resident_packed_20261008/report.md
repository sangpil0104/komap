# 12개 base 추가 탐색 결과

고정 후보와 사전 이중 품질 조건을 적용했다. 같은 valid20 탐색이며 독립 일반화 검증이 아니다. 원본 base를 변경하지 않았다.

| 후보 | 가중치 / 바이트 | valid image mIoU | base 평균 변화 | 최악 이미지 변화 | base / lite15 gate |
|---|---:|---:|---:|---:|---|
| soup11 | 11 / 626,687,153 | 80.1075937% | -0.0179673%p | -0.2243462%p (A71_step2_sample1) | False / False |
| resident12 | 12 / 683,320,900 | 80.1255342% | -0.0000268%p | -0.0015646%p (A74_step3_sample4) | True / True |
| hramp12 | 12 / 683,320,900 | 80.1254847% | -0.0000764%p | -0.0029810%p (A67_step1_sample1) | True / True |
| packed_fast12 | 12 / 628,669,975 | 80.1251872% | -0.0003738%p | -0.0029810%p (A67_step1_sample1) | True / True |

공식 평가는 각 이미지 4 IoU 평균 후 20장 평균, absent/empty=0. 원본 영상 재추론과 PNG/GT 독립 재채점 일치를 확인했다. 모든 클래스·이미지별 손실과 기존 original349 비교는 comparison.json에 포함한다. raw 단회 시간은 속도 개선 근거가 아니다.

## 상태 복원 및 저장 확인

- 실제 GPU 복원 probe: {'restoration_checks': [{'source': 'restart59_aspp768_seed456', 'all_parameters_and_buffers_exact': True, 'state_entries': 375, 'same_model_object': True, 'grad_reset': True, 'eval_reset': True}, {'source': 'restart61_pairwise_margin', 'all_parameters_and_buffers_exact': True, 'state_entries': 375, 'same_model_object': True, 'grad_reset': True, 'eval_reset': True}, {'source': 'restart59_aspp768_seed456', 'all_parameters_and_buffers_exact': True, 'state_entries': 375, 'same_model_object': True, 'grad_reset': True, 'eval_reset': True}], 'cpu_source_cache_count': 2, 'model_constructions': 1, 'scope': 'state restoration after deliberate parameter/BN/counter mutation and source switching; full valid will verify TENT outputs'}.
- full valid 복원 감사: 5개 실행에서 매 이미지 모든 source 복원 로그 확인. 원본 CPU state cache는 이미지 적응 tensor와 분리한다.
- packed_fast12: 12파일의 모든 ZIP entry 해시 및 실제 state_dict tensor의 값/shape/dtype가 압축 전과 정확히 같다. 압축은 tensor 양자화나 연산 정밀도 변경이 아니다.
- 최종 기준은 별도 원본 base 직접 대조: 683,320,900→628,669,975바이트, 원본 12파일의 모든 ZIP entry 및 checkpoint 전체 tensor 정확히 동일. canonical_lossless_storage_verification.json을 확인한다.
- 최초 압축 검사에서 B4/HRNet의 다른 checkpoint 키 구조를 처리하지 못해 중단됐다. generic tensor 재귀 검사로 고치고 완료된 추론을 다시 실행하지 않은 채 재개했다. 초기 코드와 failure 기록을 보관한다. 재개 과정의 두 파일 old_bytes는 이미 부분 압축된 파일 크기이므로 원본 기준 용량 계산에 쓰지 않는다. 원래 base 문서의 683,320,900바이트와 모든 원본 해시는 정확했다. 원본 파일 직접 대조 결과를 최종 기준으로 사용한다.
- soup11은 seed456/pairwise 실수 가중치 0.5:0.5 평균 후 FP16 저장, 정수 buffer 최대값이다. 합성 provenance를 따로 기록했다. 새로운 학습 epoch/독립 일반화 성능은 없다.
- HRNet은 FP16 autocast + tile batch8의 결합 변경이며 AMP 단독 효과로 주장하지 않는다. 실제 convolution dtype은 logs/336.log에 기록했다.

## 반복 시간

선택 후보: packed_fast12 (사전 우선순위).
- 기준 3회: [347.63134274678305, 347.9343117480166, 347.51781618595123]초 / 중앙값 347.63초.
- 후보 3회: [331.3994926298037, 331.273445405066, 331.15790376532823]초 / 중앙값 331.27초.
- 중앙값 단축 +4.706%, 교차 회차별 [4.669271185021628, 4.7885091468118475, 4.707647107182867]%. 모든 반복 품질 True, 10% gate False.
- 저장량 감소 7.998%, 가중치 12개.
- GPU7 복수 compute process 감지 0 frames. CPU/GPU 부하 telemetry/monitor를 보관했다. 공유 서버에서의 측정이며 다른 환경 재현 속도를 보증하지 않는다.

## test10

- packed_fast12: base와 99.9972452% 마스크 일치, 250/9,074,964픽셀 변화. 원본 test10 추론 203.99초 단회. 정답이 없어 test IoU/정확도가 아니다.

## 제한 및 사용 상태

100장뿐이며 동일 valid가 반복 후보 선택에 노출되었다. base의 specimen 중복·fold 선택·학습 이력/OOF 한계는 유지한다. 신규 데이터에 대한 일반화 개선을 주장하지 않는다. 기존 base와 가중치를 자동 교체하지 않고 실험 코드·설정·출력만 별도 보관했다. 이 판의 성능 수치가 높아도 교사의 학습 오류를 해소한 증류/새 학습 결과가 아니다.
