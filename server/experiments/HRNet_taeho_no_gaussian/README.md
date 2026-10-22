# HRNet_taeho_no_gaussian — 0928

기존 `HRNet_taeho`와 같은 실험에서 **Gaussian noise와 Gaussian blur 두 증강만 제거**한 비교 실험입니다. 기존 실험의 소스·설정·모델·결과는 수정하지 않았습니다. 학습은 사용자가 시작합니다.

| 항목 | HRNet_taeho | HRNet_taeho_no_gaussian |
|---|---|---|
| GaussNoise | std_range=(0.02, 0.08), p=0.2 | 제거 |
| GaussianBlur | blur_limit=(3, 5), p=0.15 | 제거 |
| 기본 물리 GPU | 0 | 1 |
| 나머지 학습·평가 설정 | 동일 | 동일 |

모델은 HRNet-W48 + U-Net++ + 경계 head이며 seed 0, batch 8, 512×512 입력, 70 epoch, AdamW, 학습률 3e-4입니다. 확대 배율, crop 샘플링, flip·회전·affine·명암·감마·elastic/grid 증강, ImageNet 정규화, 손실, 마지막 epoch 선택을 유지합니다. **추론 타일을 합칠 때의 Gaussian 가중치와 4종 flip TTA도 유지합니다.**

원본 커밋은 `97adcdf337ab78fb7ec1469fe48943ff9a6363ea`입니다. 복사한 `source/komap_seg/dataset.py`에서 두 증강 호출만 삭제했습니다. 정확한 차이는 [source_changes.patch](source_changes.patch), 변경 후 해시는 [source_manifest.json](source_manifest.json), 변경 전 해시는 [baseline_source_manifest.json](baseline_source_manifest.json)에 있습니다. `source/README.md`는 가져온 원본 설명을 그대로 보관한 문서입니다.

## 두 실험을 동시에 nohup 실행

```bash
cd /Data3/sppark/komap/experiments/HRNet_taeho_no_gaussian
./start_pair_nohup.sh
```

기존 실험은 GPU 0, 새 실험은 GPU 1에서 각각 독립된 nohup 프로세스로 실행됩니다. 두 프로세스의 PID와 로그 확인 명령이 출력됩니다. GPU를 바꾸려면:

```bash
BASELINE_GPU=2 ABLATION_GPU=3 ./start_pair_nohup.sh
```

GPU는 0~5 중 서로 다른 번호를 지정합니다. 시작 없이 배정만 확인하려면 `./start_pair_nohup.sh --dry-run`을 사용합니다.

**기존 실험을 이미 실행 중이면 새 실험만 시작하세요.**

```bash
cd /Data3/sppark/komap/experiments/HRNet_taeho_no_gaussian
HRNET_GPU=1 ./start_nohup.sh
```

각 실행은 자체 `outputs/runs/`와 `outputs/logs/`에 저장되며, 학습률·epoch·batch 등의 값은 [configs/default.json](configs/default.json)에 있습니다. 진행률·경과 시간·예상 남은 시간은 tqdm 로그에서 확인합니다.

## 실행 환경과 결과

현재는 기존 실험의 `../HRNet_taeho/.venv/bin/python`을 공유합니다. 별도 프로세스에서 이 폴더의 소스만 읽으므로 기존 코드나 환경 패키지를 변경하지 않습니다. 독립 환경이 필요하면 이 폴더에서 `bash setup.sh`를 실행하면 됩니다. 새 `.venv`가 있으면 그 환경을 우선 사용합니다. `HRNET_PYTHON`으로 실행할 Python 경로를 직접 지정할 수도 있습니다.

```bash
./run.sh preflight   # 데이터 및 이 실험 소스 해시 검사
./run.sh train       # 설정 확인만 수행, 학습하지 않음
```

학습 종료 후 valid 20장을 평가하고 `results/model_results_0928_<빈 번호>/`에 MD·CSV·JSON과 `valid_mask_comparison/`의 2×2 비교 그림을 생성합니다. 두 실험이 동시에 끝나도 폴더 번호를 원자적으로 예약하므로 덮어쓰지 않습니다. MD 제목 및 JSON의 모델명으로 두 실험을 구분합니다.

지표는 기존과 같은 image mIoU입니다. 이미지마다 4개 클래스 IoU를 평균하고, 20장의 평균을 구합니다. 정답·예측 양쪽에 없는 클래스의 IoU는 1입니다. 그림은 원본/오류/예측/정답을 2×2로 배치하고 image mIoU와 `100% − image mIoU`를 표시합니다.

학습 전 점검 결과는 [VALIDATION_REPORT.md](VALIDATION_REPORT.md)에 기록했습니다. 아직 이 변형 모델의 학습 결과나 성능 점수는 없습니다.
