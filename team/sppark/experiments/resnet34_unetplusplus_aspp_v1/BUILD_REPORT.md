# BUILD_REPORT — ResNet34 U-Net++ + Residual ASPP 검증 보고서

## 1. 단위 테스트 결과
- **테스트 환경**: Python 3.11.16, PyTorch 2.7.1+cu128, CUDA 12.8, RTX 4090 8-GPU
- **결과**: **8 passed in 6.00s**

| 테스트 항목 | 대상 함수 / 클래스 | 결과 | 검증 내용 |
|---|---|:---:|---|
| `test_compute_3channels_and_padding` | `dataset.compute_3channels`, `pad_to_multiple` | PASSED | 3채널 통계(raw, z-score, cdf) 및 32배수 패딩 검증 |
| `test_dataset_splits_count` | `workspace.get_dataset_samples` | PASSED | Train 70장, Valid 20장, Test 10장 수량 검증 |
| `test_train_dataset_sampling_and_shapes` | `dataset.KomapTrainDataset` | PASSED | 패치 크기(384×384) 및 D4 증강 검증 |
| `test_eval_dataset_shapes` | `dataset.KomapEvalDataset` | PASSED | 검증셋 해상도 및 32배수 패딩 검증 |
| `test_residual_aspp_shape_and_gradient` | `models.ResidualASPP` | PASSED | 다중 Atrous Conv 및 잔차 연결 shape/gradient 검증 |
| `test_resnet34_unetplusplus_aspp_forward_and_backward` | `models.ResNet34UNetPlusPlusASPP` | PASSED | U-Net++ Dense Skip 및 순전파/역전파 gradient 검증 |
| `test_resnet34_parameter_count_and_groups` | `models.ResNet34UNetPlusPlusASPP` | PASSED | 총 파라미터 수(27,481,396) 및 차등 LR 파라미터 그룹 분리 검증 |
| `test_full_resolution_padded_forward` | `models.ResNet34UNetPlusPlusASPP` | PASSED | 원해상도(909×1278 $\to$ 928×1280) 크롭 복원 검증 |

---

## 2. 통합 Smoke 검증 결과
- **실행 명령**: `python run.py train --smoke --gpus 1`
  - Patch 256 2 updates $\to$ Full-frame 1 update
  - Validation mIoU 평가 및 EMA 가중치 선택 정상 동작 확인
- **평가 및 예측 명령**:
  - `python run.py evaluate`: Plain 검증 mIoU 45.02% (초기 업데이트 기준)
  - `python run.py evaluate --d4`: D4 TTA 검증 mIoU 45.03%
  - `python run.py predict`: 테스트 10장 예측 PNG 생성 완료 (`outputs/predictions/*_mask.png`, 8-bit RGBA)
  - `python run.py report`: [results/model_results_0921_1.md](../../results/model_results_0921_1.md) 생성 및 자동 갱신 확인

---

## 3. 리소스 및 메모리 프로파일
- **총 파라미터 수**: 27,481,396개 (27.48M)
- **BF16 Full-Frame(909×1278) 역전파 피크 GPU 메모리**: 약 10.8 GB (RTX 4090 24GB 기준 여유 13.2GB 확보)
- **BF16 Patch(512×512) 역전파 피크 GPU 메모리**: 약 2.4 GB
- **추론 피크 GPU 메모리**: 약 1.2 GB

