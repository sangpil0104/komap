# 12개·10개 FP16 후보의 제한된 일반화 검증

[한국어 결과](report.md), [사전 고정 범위](protocol.md), [조건부 통계](valid_conditional_stability.json), [데이터/학습 이력 감사](data_exposure_audit.json), [정답 없는 test 출력](test_mask_stability.json).

현재 이미지 100장 중 정답은 90장이고 모든 valid20은 이미 후보 선택에 사용됐다. 이 작업은 독립 일반화 점수를 만들지 않았다. completion.json의 diagnostics_completed는 데이터 감사·조건부 통계·test 마스크 확인의 완료를 뜻하며 independent_generalization_verified는 false다.

데이터 감사에서 train-valid specimen 중복 7개를 확인했다. fold 학습 출력과 동결 배포판의 실제 텐서가 달라 현재 OOF 캐시를 배포 후보의 성적에 사용할 수 없었다. 사전 프로토콜의 학습 노출 가정은 기록된 원래 CV 분할을 적용할 때의 조건부 판단이며, 동결 가중치의 완전한 학습 계보가 인증됐다는 뜻이 아니다. 참조 훈련 코드는 실행 증명이 아니며 파일·텐서 비교 결과는 JSON에 있다.

서버 root: `/workspace/model349_compress/generalization_20261008`.
Python: `/workspace/venv-komap/bin/python`.
후보 실물은 기존 `audit_20261008/bundles/cpu2_fp16_12` 및 `cpu2_fp16_10`에 보존했다. 이 폴더의 frozen_bundle_metadata에는 정확한 추론 코드·설정·calibration·체크포인트 SHA 목록을 수집했다. 가중치와 확률 캐시는 서버에 있고 로컬 아카이브에는 제외했다.

CPU 재현 순서는 audit_data_exposure.py → analyze_stability.py → (두 test 실행 완료 후) compare_test_masks.py → generate_report.py다. 현재 코드의 ROOT/OLD/DATA는 서버 경로로 고정돼 있으므로 다른 머신에서는 새 출력 root와 실제 입력 경로를 지정하도록 조정해야 한다. 원본 source 파일과 가중치는 수정하지 않는다. 참조 reference_hrnet_train.py는 저장소 훈련 코드의 읽기용 복사본이며 실행하지 않았다.

test10 재현은 두 동결 bundle에서 기존 output과 다른 신규 경로를 지정해 infer_lite.py --input /workspace/komap_model1/data/test/images --output NEW_OUTPUT --gpus FREE_GPU --keep-probabilities로 실행한다. --gt는 지정하지 않는다. 시작 전 현재 작업·GPU 점유·공유 GPU 매뉴얼을 확인한다. 이번 run_test10.py와 finalize.py는 이미 완료된 이번 실행의 기록이며 같은 출력 경로로 다시 실행하지 않는다.

두 후보는 test10 원본 추론을 각각 한 번 수행했다. 시간이 429.04/332.79초였으나 공유 부하와 동시기 반복 기준이 없어 속도 개선을 확정하지 않는다. GPU7은 종료 뒤 6MiB/0%로 확인했고 다른 사용자 작업은 조작하지 않았다. 가중치·설정·추론 코드의 실행 전후 해시는 일치했다.

bootstrap은 specimen 단위 20,000회(seed20261008), 각 뽑힌 specimen의 모든 이미지를 포함해 공식 image-weighted 평균을 계산했다. 선택에 이미 사용한 valid를 재표본화한 조건부 진단이며 선택 편향 교정이나 신규 holdout 검증이 아니다. 한 specimen 제외 분석도 재학습 cross-validation이 아니다.
