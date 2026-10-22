# lite20 준비 및 검증 결과

2026-10-11, conda `komap`에서 빌드했습니다. 실제 학습과 GPU 모델 추론은 시작하지 않았습니다. `outputs/`와 학습용 `logs/`는 비어 있으며 `./start_nohup.sh`를 사용자가 실행해야 시작됩니다.

## 다운로드와 실행 범위

- 고정 커밋: `f47ade8ebd8842944ba86f076eac79f5a4e51ae5`
- ZIP: `Model349_lite20_b4ws_bundle_code.zip`, 2,481,090 bytes.
- 배포 SHA-256과 ZIP CRC 확인: `60cd2b9d208fb39c68cc645786a7712c96a3b18169ed483e8ee64743322b8bb8`.
- 물리 GPU 2~7, UUID로 작업별 1개 GPU 노출, 최대 동시 GPU 작업 6개, `nohup`, conda `komap`.
- S=0, 선행 단계 포함 35개 학습 / 전체 51단계. 최종 R11 + E8 + F1 = 20개 가중치.

## 원본 패키지에서 연결한 부분

ZIP의 원래 재학습 shell은 lite23 설정으로 마무리합니다. 로컬에서는 사용하지 않는 restart59와 B4-322 fold4 학습을 제거하고, 후속 학습에 필요한 Model109는 선행 단계로 유지했습니다. 검증·test 추론·패키징·보고서를 모두 lite20 설정으로 연결했습니다.

R의 `infer_lite.py`는 이번 ZIP과 바이트 단위로 같습니다. `selected_R_lite20.yaml`도 원문 그대로 사용하며 기본 R 설정으로 복사했습니다. model109/seed456 구간에서 restart51을 선택하고, B4-322는 fold3만 사용합니다. E/F와 나머지 학습 레시피는 이전 검증된 로컬 조정을 유지합니다.

프로젝트 지침대로 image mIoU(empty union=0), tqdm, 명시적 B4 worker/Albumentations 난수 시드를 적용했습니다. 새 가중치·메타데이터의 SHA-256 manifest를 생성하고 strict 검사를 유지합니다. `--allow-retrained`로 검사를 생략하지 않습니다.

## 통과한 검사

| 검사 | 확인 내용 | 결과 파일 |
|---|---|---|
| CLI / 의존성 | 실제 entrypoint 51개의 argparse 검증, 학습 본문 전에 종료; 35개 학습과 Model109 선행 관계 | `checks/design.json` |
| lite20 변경점 | 실제 밝기 선택 코드의 6개 합성 입력, 제외된 구간의 restart51 대체, 필요한 R 가중치 11개 | `checks/lite20_routing.json` |
| R GPU 전달 | 실제 R launcher를 모의 subprocess로 검사; 구성요소 7개가 GPU2~7 UUID에 전달됨 | 같은 파일 |
| 스케줄러 | 51개 모의 작업, 최대 동시 6개, 의존성 순서, 점유 GPU 대기, 실패·누락 산출물·자식 종료 처리 | `checks/scheduler.json` |
| 공개 초기화와 지표 | ImageNet ResNet34/B4/HRNet을 CPU에서 로드; 빈 클래스 IoU=0과 이미지별 동일 가중치 | `checks/models_cpu.json` |
| F OOF | 합성 train70 OOF 특징, 잘못된 heldout inventory 거부, gray 정규화, 1,120 crops, 11,876 parameters, F 초기 항등 출력 | `checks/features_cpu.json` |
| 보고서 / 마스크 | 실제 결합·채점 스크립트에 합성 입력, 독립 IoU 계산, stage2 대표 결과, 비교 그림 40장과 test 마스크 10장 내보내기 | `checks/export_cpu.json` |
| 20개 패키징 | 실행 소스 스냅샷, R11+E/F9의 정확한 개수와 strict hash, 보조 모델 9개 CPU strict load, HR 전처리 유지 | `checks/runtime_cpu.json` |
| 사전 점검 | 190개 입력 해시, train70/valid20 GT 색·크기, 시편 fold, R/E B4 fold 일치, lite20 설정 일치, 요청 GPU 6개 확인 | `checks/preflight_*.json` |
| 기존 파일 보존 | 기존 lite23 소스·계획·결과 관련 1,373개 파일 해시 불변 | `checks/preservation_check.json` |

패키징 검사의 기존 가중치는 임시 fixture로만 읽었고 검사 후 삭제했습니다. 실제 학습 시작 가중치나 새 학습 결과로 등록하지 않았습니다. 합성 F 입력의 작은 CPU forward 외에는 모델 forward를 실행하지 않았습니다. 밝기 배정 검사에서는 모델 예측과 GPU subprocess를 대체했으며 optimizer update는 0입니다.

초기 사전 점검에서 로컬 검사기의 팔레트 숫자 오기를 발견해 원래 팔레트로 수정했습니다. 데이터 파일은 변경하지 않았고 재점검을 수행했습니다. 초기 로그는 `checks/preflight_initial.log`, 수정 후 로그는 `checks/preflight.log`에 남겼습니다.

전체 실제 학습과 GPU 추론을 끝까지 수행한 검증은 아직 없습니다. 재학습 점수와 실제 소요 시간은 학습 완료 후에 알 수 있습니다. 코드 차이는 `checks/source_changes.diff`, 원본은 `upstream/`, 실행 계획은 `plan.json`에 보존했습니다.
