# 준비 및 검증 결과

2026-10-11, conda `komap`에서 준비했습니다. **실제 학습·GPU 추론을 시작하지 않았으며 `outputs/`와 학습 로그 폴더는 비어 있습니다.** 사용자가 `./start_nohup.sh`를 실행해야 학습이 시작됩니다.

## 고정 범위

요청 모델은 `Model349_fusion805_lite23_b4ws`입니다. 고정 커밋 `fdeeccf246ee8dfcecca0c9d5ebfd85f31cd18db`의 요청 ZIP과 통합 학습 ZIP을 모두 내려받아 배포 SHA-256 및 ZIP CRC를 확인했습니다. lite21이나 full70 후보로 대체하지 않았습니다.

원본 `S=0`의 전체 재학습을 작업별 의존성 그래프로 옮겼습니다. R 선행 모델 9개도 새로 학습하고 B4-329는 R/E에서 공유합니다. 총 37개 학습 작업 / 53단계, 물리 GPU 2~7에 최대 6개 GPU 작업입니다. 최종 패키지는 정확히 23개 가중치를 사용합니다.

## 통과한 검사

| 검사 | 결과 | 근거 |
|---|---|---|
| 모든 학습·OOF·특징 생성·valid 명령 인자 | 53개 실제 entrypoint의 argparse 통과, 학습 본문 실행 전 중단 | `checks/design.json` |
| GPU 스케줄러 | 53단계 모의 작업 완료, GPU 2~7만 배정, 최대 동시 6개, 부모 완료 후 자식 시작 | `checks/scheduler.json` |
| 중단 및 오류 처리 | 실패 자식·후손 정리, 누락 산출물 거부, 점유 GPU 대기 | 같은 파일 |
| 데이터와 코드 사전 점검 | train70 / valid20 / test10, 190개 입력 해시, GT 팔레트·크기, R/E fold 일치, 시편 분리, CUDA GPU 6개 확인 | `checks/preflight_*.json` |
| 공개 초기 가중치 | ResNet34, EfficientNet-B4, HRNet-W48의 캐시된 ImageNet 초기화 CPU 로드 | `checks/models_cpu.json` |
| image mIoU | empty union=0, 이미지 크기와 무관한 이미지별 동일 가중치 검증 | 같은 파일 |
| F 입력 | 합성 OOF70 특징, 중복·잘못된 OOF 거부, gray 정규화, 1,120 crops, 11,876 parameters, 영 초기화 F의 항등 출력 | `checks/features_cpu.json` |
| 최종 결합·결과 저장 | 실제 stage1/stage2·채점 코드에 합성 입력, tie 순서, 독립 IoU 재계산, 비교 그림 40장과 test 이미지 10장 내보내기 | `checks/export_cpu.json` |
| 실행 스냅샷·23개 패키징 | 소스 해시 동일, 23개 가중치 manifest, strict metadata 해시, E/F 9개 CPU strict load, HRNet 전처리 보존 | `checks/runtime_cpu.json` |
| 기존 산출물 보존 | 기존 소스·계획·결과 관련 929개 파일 해시 불변 | `checks/preservation_check.json` |

패키징 검사에서는 기존 가중치를 **삭제되는 임시 fixture**로 읽어 직렬화와 호환성만 검증했습니다. 새 실험의 학습 결과나 시작 가중치로 등록하지 않았습니다. F 입력 검사의 작은 합성 CPU forward 외에는 모델 forward를 실행하지 않았고 optimizer update는 0입니다. 임시 fixture는 모두 제거했습니다.

## 로컬 조정

- 기존 실험에서 검증한 로컬 학습 소스를 별도 복사했습니다. 프로젝트 지침대로 image mIoU(empty=0), terminal tqdm, B4의 명시적 worker/Albumentations 시드를 적용했습니다. 과거 R의 absent-class 제외 선택 대신 image mIoU를 사용합니다.
- 원본의 GPU별 긴 shell 묶음을 의존성 스케줄러로 옮겨 GPU 여섯 개에 작업을 배정합니다. `CUDA_VISIBLE_DEVICES`에는 물리 GPU의 UUID 하나만 넣습니다. 새 실행 폴더, 중복 실행 lock, 자식 process group 정리를 사용합니다.
- B4ws 원본 구조·손실·추가 학습 횟수는 유지합니다. BN running mean/variance와 batch counter가 학습 전후 정확히 같은지 실행 중 검사하여 `bn_freeze_check.json`에 기록합니다.
- F는 추가 학습 전 E 12개 branch의 OOF만 사용합니다. valid E에는 B4ws 4개와 지정 HR fold 4개만 사용합니다. 가족 평균을 FP32로 계산한 뒤 FP16으로 한 번 반올림하는 순서를 유지합니다.
- 원본 공개 가중치 SHA를 새 가중치에 적용하지 않고, 새 가중치·메타데이터의 해시를 생성하고 검증합니다. `--allow-retrained`로 검사를 우회하지 않습니다.
- 요청 모델에 맞춰 R+E+F를 대표 결과로 저장하고, R+E는 비교용으로 저장합니다. test10은 예측만 합니다.

세부 코드 차이는 `checks/source_changes.diff`, 전체 계획은 `plan.json`, 다운로드 출처는 `provenance.json`에 있습니다. 이번 서버에서 37개 전체 학습을 끝까지 실행한 실측이나 성능 결과는 아직 없습니다. 검증은 준비 단계의 정적·CPU·모의 검사 범위입니다.
