# GitHub lhs Model349 + Model397 — GPU 4,5,6,7

준비일: 2026-10-06 (Asia/Seoul). **학습은 시작하지 않았습니다.**

GitHub CLI로 https://github.com/CNUKoMaP/lhs 의 commit `851086d2b95c0c5f01f7a67853f5364ab496c121`에서 두 재현 ZIP을 다운로드했습니다. 파일 해시는 `provenance.json`에 기록했고 `downloads/`, `upstream349/`, `upstream397/`에 원본을 보존했습니다. ZIP은 학습된 가중치가 아닌 재학습용 소스입니다. 기존 모델, 기존 소스, 기존 결과는 수정하지 않았습니다.

## 실행

```bash
cd /Data3/sppark/komap/experiments/lhs_349_397_gpu4567_v1
./start_nohup.sh
```

사용자가 실행하면 conda `komap`에서 nohup 학습이 시작됩니다. 별도 환경 활성화가 필요 없습니다. **Model349 전체 완료 후 Model397 전체 실행** 순서입니다. GPU는 물리 인덱스 4,5,6,7만 배정하며, 자식 작업에는 해당 GPU UUID를 전달합니다. 모델/fold당 GPU 하나를 사용하고 최대 4개 독립 작업이 동시에 실행됩니다. DDP가 아닙니다. 의존 단계/CPU 계산 단계에서는 GPU 사용량이 줄 수 있습니다.

실행 후 PID, 로그 경로 및 `tail -f` 명령이 출력됩니다. 단계 진행률, 경과 시간, ETA와 현재 작업 진행 정보가 로그에 표시됩니다. 상세 로그는 각 실행의 `logs/`와 학습 폴더에 남습니다. localhost 서버나 대시보드는 사용하지 않습니다.

각각만 실행하려면 아래 중 하나를 실행합니다. 두 명령을 동시에 실행하지 마세요.

```bash
./start_nohup.sh --model 349
./start_nohup.sh --model 397
```

학습 없이 다시 검사하려면 `./run.sh check`를 실행합니다. 중단은 출력된 메인 PID에 `kill PID`를 보내면 됩니다. 실행기는 활성 작업과 그 자식 프로세스를 종료하며 로그·체크포인트를 보존합니다. 349가 실패하면 397를 시작하지 않습니다. 이미 성공한 349를 다시 실행하지 않고 397부터 새로 실행하려면 `--model 397`을 사용합니다. 새 실행마다 별도 폴더를 만들고 실패 실행을 자동 재개하지 않습니다.

## 모델과 결과

- **Model349**: 이전에 검증한 로컬 image-mIoU 실행기를 별도 사본으로 준비했습니다. 이번 다운로드 ZIP과 보존 upstream의 전체 파일이 바이트 단위로 일치합니다. 59단계(50 GPU / 9 CPU), ResNet 계보·HRNet·B4 fold 학습과 최종 조합입니다. 학습 설정을 유지하고 GPU 목록을 4,5,6,7로 변경했습니다. 원본 대비 로컬 지표 및 결과 저장 수정은 `model349/reports/local_patches.diff`에 있습니다. valid를 모델·조합 선택에 사용하므로 독립 test 점수가 아닙니다.
- **Model397**: B4 EfficientNet-B4 U-Net 4-fold × 16,000 iterations, HRNet-W48 U-Net++ 2 seeds × 4-fold × 70 epochs를 원본 설정으로 학습합니다. 각 fold OOF와 provenance를 생성하고 원본 OOF scorer를 실행합니다. 고정 조합은 B4 0.50 + HR seed1 0.25 + HR seed2 0.25입니다. 원본 397는 OOF 전용이고 이전 gate 실패로 공식 valid 결과가 없습니다. 사용자의 검증 결과 요구에 따라 추가한 **로컬 valid20 평가**는 각 family의 4-fold 확률 평균을 위 비율로 결합하며 공식 원본 valid 결과로 주장하지 않습니다. valid로 추가 탐색하거나 체크포인트를 선택하지 않습니다.

397 로컬 사본에는 tqdm, 빈 클래스 IoU=0, global 지표 제거만 적용했습니다. 원본과 차이는 `reports/model397_local_patches.diff`에 보존했습니다. 새 검증 평가기는 `model397/export_valid.py`이며, 클래스 순서를 명시적으로 처리합니다.

결과는 모델별로 `/Data3/sppark/komap/results/model_results_MMDD_다음번호/`에 자동 저장됩니다. 날짜는 한국 시간이며 기존 폴더를 덮어쓰지 않습니다. 같은 이름의 Markdown 보고서, `metrics.json`, 예측 마스크 및 `valid_mask_comparison/` 20장이 생성됩니다. 비교 배치는 좌상 Original / 우상 오류 / 좌하 Prediction / 우하 Ground truth이며, 오류는 정답 색이고 맞은 픽셀은 흰색입니다. image mIoU와 `100%−image mIoU`가 표시됩니다.

지표는 이미지별 4-class IoU 평균 후 valid 20장 평균이며, union=0은 IoU=0입니다. global mIoU는 계산하지 않습니다. Model397 OOF 점수는 별도로 train 70장 기준이며 `model397/outputs/run_.../results/model397_oof.json`에 저장합니다.

새 체크포인트·확률·소스 및 실행기 스냅샷은 `model349/outputs/runs/`와 `model397/outputs/run_.../`에 남습니다. 397의 valid 확률은 float16 archive를 float32로 복원·정규화한 뒤 결합합니다. 원본 P100 환경과 현재 RTX4090/conda 환경이 다르므로 원본 기록 점수와 동일함을 보장하지 않습니다.

## 완료한 검사

- conda komap, 데이터 train70/valid20, 마스크 색상·크기와 Model349 원본/로컬 소스 무결성 검사.
- Model397 원본 smoke 9개, 실제 70장/59 specimen group의 4-fold split 및 26-task 실행 계획 검사.
- GPU 4,5,6,7 각각에서 ImageNet 가중치로 B4 입력 2×3×1536×1536, HRNet 입력 4×3×512×512 forward 통과. 학습/backward는 실행하지 않았습니다.
- 59단계 합성 subprocess 실행으로 GPU 4개 독점 배정·의존 관계·실패 시 자식 정리·필수 출력 검사 통과.
- 합성 valid20 결과 저장 통과: 클래스 순서, 빈 클래스 IoU=0, 이미지 평균, 20개 비교 이미지 및 배치/흰색 정상 픽셀 확인.

검사 로그와 JSON은 `reports/`에 있습니다. 전체 실제 학습의 완료 여부는 사용자가 실행한 뒤 확인할 수 있습니다.
