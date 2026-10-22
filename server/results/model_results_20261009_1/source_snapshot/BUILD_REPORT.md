# exh14 재학습 실험 준비 기록

준비일: 2026-10-09 (Asia/Seoul).

**실제 학습, GPU forward/backward, 최종 추론, TENT를 실행하지 않았습니다.**
사용자가 `./start_nohup.sh`를 실행해야 14개 구성 가중치 재학습을 시작합니다.

## 준비된 내용

- GitHub `CNUKoMaP/lhs`, commit `1704fff772a713101a2d63467b4d9321420ada3f`의
  exh14 추론 ZIP과 Model349 학습 ZIP 다운로드 및 해시 보존.
- 학습 소스는 기존 image-mIoU 수정본을 새 경로에 복사.
  GitHub 원본 대비 차이는 `checks/training_source_patches.diff`에 보존.
- 새로운 실험 입력: train70/valid20, 원래 specimen group 4-fold, Model98·257 부모 사본.
- 원래 epoch/iteration·학습률·손실·checkpoint 선택을 유지한 14개 학습 명령.
- 새 부모 연결: Model109→restart51·59, restart51→restart61, HRNet336→347.
- 물리 GPU 2~7을 UUID로 배정하는 최대 6개 병렬 작업과 부모 의존성 처리.
- 학습 후 새 가중치 14개를 FP32로 묶고 원본 exh14 추론·고정 결합으로 valid20 평가.
- image mIoU 보고서·20개 비교 이미지·새 모델·실행 소스 자동 결과 저장.

## 통과한 검사

| 검사 | 확인 내용 | 기록 |
|---|---|---|
| 실제 학습 CLI | 14개 명령을 실제 parser까지 실행하고 학습 진입 전에 중단; 원래 하이퍼파라미터 유지 | `checks/design.json` |
| GPU 스케줄러 | 합성 subprocess로 14개 작업, GPU 2~7 전체 사용, 최대 동시 6개, 중복 GPU 배정 없음, 부모 의존성 | `checks/scheduler.json` |
| 실패/대기 | 실패한 부모의 후속 작업 금지, 자식 프로세스 정리, 누락 checkpoint 거부, 사용 중 GPU 대기 | `checks/scheduler.json` |
| 부모/초기 가중치 | Model98·257과 offline ImageNet B4·HRNet encoder CPU 로딩 | `checks/models_cpu.json` |
| 실행 스냅샷·추론 모델 저장 | 실제 보존 체크포인트를 임시 테스트 입력으로 사용하여 FP32 14개 포장, 해시 검증, 원본 추론 로더의 엄격 CPU 로딩 | `checks/runtime_cpu.json` |
| 지표 | 고해상도/restart, B4 gray·zcdf, HRNet, 최종 exporter에서 빈 클래스 0, 예제 image mIoU 60% | `checks/metrics.json` |
| 결과 저장 | 크기가 다른 합성 valid20의 이미지 평균 67.5%, 20개 2×2 패널, 오류/정답 색상, 새 폴더 번호 | `checks/export_cpu.json` |
| 데이터·환경 | train70/valid20 크기·팔레트·해시, 6개 GPU 가용성, conda komap | `checks/preflight_*.json` |

Fold별 학습/heldout 이미지 수: **52/18, 52/18, 53/17, 53/17**.
각 fold 내 시편 그룹이 겹치지 않으며 70장 각각 정확히 한 번 heldout에 포함됩니다.
원래 이름인 `oof_fold1..4`를 유지하고 새 실험의 train 데이터만 가리키도록 연결했습니다.

Model98에서 ASPP Model109를 구성할 때 새 `context.*` 파라미터 31개가 없는 것은
원래 아키텍처 확장 동작입니다. 예상하지 않은 키는 없었고 나머지 초기 파라미터가 로딩되었습니다.
HRNet ImageNet 로딩의 classifier/head unused-key 안내는 feature encoder 구성에 따른 출력이며 로딩이 완료되었습니다.

`checks/layout_preview_synthetic.png` 및 runtime 검사에 사용한 가중치는 검사 입력입니다.
실제 재학습 점수나 새 학습 모델로 저장하지 않았습니다. 임시 검사 출력은 임시 경로에서 정리했습니다.

실제 GPU 학습의 소요 시간·VRAM 사용량·최종 image mIoU는 사용자가 실행한 뒤 측정됩니다.
기존 모델, 이전 실험 소스·결과, 기존 고정 추론 세팅, conda 패키지를 변경하지 않았습니다.
