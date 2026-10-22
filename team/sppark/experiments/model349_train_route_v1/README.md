# Model349 학습·조합 실험

요청한 [Model349_train_route_code.zip](https://github.com/CNUKoMaP/lhs/blob/fd7ced108b6577fa8049fee2ae9ebd32caa17cc4/Model349_train_route_code.zip)을 내려받아 별도 실험으로 구성했습니다. **학습은 시작하지 않았습니다.**

## 실행

```bash
cd /Data3/sppark/komap/experiments/model349_train_route_v1
./start_nohup.sh
```

위 명령을 사용자가 실행하면 실제 학습이 시작됩니다. PID와 로그 경로, `tail -f` 명령을 출력합니다. 기본은 **물리 GPU 0 한 개**에서 모든 단계를 순차 실행하고, CPU 조합 탐색은 4개 worker를 사용합니다. GPU·CPU worker를 지정하려면:

```bash
MODEL349_GPU=2 KOMAP_WORKERS=4 ./start_nohup.sh
```

터미널에서 직접 실행하거나 학습 없이 점검할 수도 있습니다.

```bash
./run.sh check                 # 데이터·환경·원본 무결성·fold 점검만
./run.sh train                 # 학습 설정 점검만, 학습하지 않음
./run.sh train --execute       # 사용자가 실행하는 실제 학습
```

`run.sh`는 `/home/sppark/miniconda3`의 **conda komap**을 활성화합니다. 별도 venv를 만들거나 기존 패키지를 변경하지 않았습니다. 현재 PyTorch 2.7.1+cu128 / torchvision 0.22.1+cu128 / SMP 0.5.0 / timm 1.0.29입니다. 전체 버전은 `reports/environment.freeze.txt`에 있습니다. 원본 P100용 패키지 버전과 다르므로 비트 단위 재현을 보장하지 않습니다.

## 실행 범위

Model349는 단일 네트워크가 아니라 다음 모델들의 학습과 확률 조합을 포함합니다.

1. ResNet34 U-Net++ 및 ASPP: Model6 → 62 → 97 → 98 → 109, restart51/59/61, Model123, Model181.
2. Model183 조합 입력 재생성, Model194 → 195, Model220 TENT, Model239 → 242 → 257 → 269.
3. EfficientNet-B4 Model322/326/329, 각 4-fold × 16,000 step: **총 12개 fold, 192,000 step**.
4. HRNet-W48 U-Net++ Model336 70 epoch 및 Model347 12 epoch.
5. CPU 확률 조합 탐색 Model335 → 338 → 339 → 341 → 346 → 349, 최종 검증 및 결과 저장.

원본의 학습 순서를 유지합니다. 여러 GPU로 fold를 동시에 학습하는 구성은 아닙니다. 장시간 실험이며 전체 소요 시간은 아직 측정하지 않았습니다. 학습·추론·탐색의 tqdm에 진행률, 경과 시간, ETA가 표시됩니다. B4 상세 진행률도 메인 로그와 각 단계 로그에 함께 기록합니다.

공개 ImageNet 초기 가중치 3종(ResNet34, EfficientNet-B4, HRNet-W48)은 기존 캐시에서 오프라인 로딩과 CPU forward를 확인했습니다. 기존 경쟁용 학습 체크포인트를 가져다 쓰지 않으며, 필요한 중간 모델은 이 실험에서 새로 학습합니다.

## 데이터와 보존

- `downloads/Model349_train_route_code.zip`: 다운로드 ZIP 원본.
- `upstream/`: 압축을 그대로 푼 원본. 변경하지 않았습니다.
- `source/`: 프로젝트 실행용 사본. 변경 내역은 `reports/local_patches.diff`.
- `adapters/`, `build_source.py`: 실행용 사본 생성 규칙과 지표·결과 저장 코드.
- `datasets/train.zip`, `valid.zip`: 기존 `dataset/`의 학습 70장, 검증 20장을 원본 바이트 그대로 패키징. 기존 데이터는 변경하지 않았습니다.
- `provenance.json`, `source_manifest.json`, `datasets/manifest.json`: Git commit, ZIP·소스·데이터 SHA-256.
- `outputs/runs/model349_train_UTC시간/`: 매 실행의 새 체크포인트·확률·로그·소스 스냅샷.

원본 commit은 `fd7ced108b6577fa8049fee2ae9ebd32caa17cc4`이며 ZIP SHA-256은 `89686ffb53bfc09bc01aafb63b81aabd3905143107ed4f9996683ac0427a01b2`입니다.

4개 specimen-group fold는 원본 manifest를 사용합니다. 학습/OOF 이미지 수는 52/18, 52/18, 53/17, 53/17이며, 모든 학습 시편은 정확히 한 번 OOF에 포함됩니다. test 데이터는 학습·조합에 연결하지 않습니다.

새 실행은 항상 새 디렉터리를 만들며, 같은 실행기의 중복 실행은 파일 잠금으로 차단합니다. 실패하면 생성된 로그와 체크포인트를 보존하고 종료합니다. 자동 중간 재개 기능은 추가하지 않았습니다. 실행용 소스를 나중에 바꾸려면 이 버전을 보존하고 새 실험 버전을 만드세요.

## 프로젝트 지표 적용

최종 점수는 **이미지마다 4개 클래스 IoU를 평균한 뒤 검증 20장에 대해 평균한 image mIoU**입니다. 정답·예측에 모두 없는 클래스는 IoU=0입니다. `(0.8 + 0.8 + 0.8 + 0) / 4 = 0.6`을 검사했습니다.

중간 학습 모델의 체크포인트 선택과 B4 평가도 이 기준으로 맞췄습니다. 원본의 global mIoU 계산은 실행 경로에서 제거했습니다. 기존 호출 호환을 위한 `image_miou_compat`와 학습 코드의 `image_mean_miou_exclude` 필드는 모두 같은 empty=0 점수입니다. 조합 탐색의 skip 참고 통계는 원본대로 남아 있지만 최종 선택은 `best_image_empty_zero`를 사용합니다.

따라서 **원본과 중간 체크포인트 선택 정책이 다를 수 있습니다**. 원본의 보관 점수를 그대로 재현한다고 주장하지 않습니다. valid 데이터는 중간 체크포인트와 조합 선택에 사용되므로 최종 점수 역시 검증 점수입니다.

재학습으로 선택되는 조합이 달라질 때의 원본 예외도 보완했습니다. Al expert가 없는 부모 조합에서는 Al gamma를 0으로 유지하고, Model341이 Model338 기준 조합을 선택한 경우 이후 단계도 그 기준을 사용합니다. 원본 파일은 `upstream/`에 남아 있습니다.

## 학습 완료 후 결과

완료 시 프로젝트 루트에 `results/model_results_MMDD_다음번호/`를 새로 생성합니다. 이미 존재하는 결과는 건드리지 않습니다.

- 같은 이름의 Markdown 결과 보고서와 `metrics.json`.
- `valid_mask_comparison/`: 검증 20장 각각의 2×2 비교 이미지.
- `predicted_masks/`: 클래스 색상 마스크.
- `predicted_labels/`: 0–3 label PNG. 클래스 순서는 Al3Ni, Primary Si, Al, Eutectic Si.
- 사용한 ZIP 출처, 환경 목록, 실행용 패치.

비교 이미지는 **좌상 Original / 우상 Error / 좌하 Prediction / 우하 Ground truth**입니다. Error는 맞은 픽셀을 흰색, 틀린 픽셀을 해당 위치의 정답 색으로 표시합니다. 위쪽에 image mIoU와 `100% − image mIoU`를 표기합니다. 후자는 픽셀 오류율과 다릅니다.

최종 마스크를 실제 조합 함수로 다시 계산하고 선택 점수와 일치하는지 확인한 뒤 결과를 저장합니다. 실행 폴더의 `result_path.txt`에도 결과 경로가 기록됩니다.

## 준비 검증

검증 로그와 JSON은 `reports/`에 있습니다. 실행 인자는 실제 parser까지만 실행하고 즉시 중단해 검사했으며, 학습·optimizer update는 하지 않았습니다. 합성 결과는 `outputs/checks/`에만 저장하여 실제 `results/`와 분리했습니다.

- 원본 ZIP CRC와 Git blob SHA, 소스 SHA-256 확인.
- 90개 image/mask 쌍의 크기·색상·투명도·fold 확인.
- 파이프라인 62개 Python 호출을 추적하고 33개 고유 인자 구성 검사.
- 세 모델 계열의 공개 초기 가중치 오프라인 로딩 및 CPU forward.
- 빈 클래스=0, 이미지별 동일 가중치, 실제 지표 adapter, 20개 결과 패널과 오류 색상 검사.
- 작은 합성 입력으로 6단계 조합 연결 및 최종 점수·결과 저장 검사. 전체 후보 탐색 또는 실제 학습 성능 검증은 아닙니다.
