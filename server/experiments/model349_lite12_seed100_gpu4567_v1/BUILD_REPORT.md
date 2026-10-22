# Model349 lite12 seed100 빌드 검증

검증일: 2026-10-08 (한국 시간). **학습·GPU 추론·TENT를 시작하지 않았습니다.**

## 구성

- GitHub `CNUKoMaP/lhs`, commit `f6456f54e5fc6517a89a3fdb679645a0520bbf61`, `Model349_lite12_base`.
- GitHub의 추론 코드·설정 31개 파일은 `frozen/`과 바이트 단위 일치합니다.
- 실행 환경: `/home/sppark/miniconda3/envs/komap`, PyTorch `2.7.1+cu128`.
- 학습 100개, 시드 `3407–3506`, GPU 물리 `4,5,6,7`, 완료 후보 84개.
- 기본 `finish-all`: 100개 완료를 우선하며 시간 제한으로 조기 종료하지 않습니다.
- GitHub 원본은 `upstream/`, 로컬 복사본 변경은 `source_changes.diff`, 출처는 `provenance.json`에 기록했습니다.

## 통과한 검사

| 검사 | 결과 |
|---|---|
| CPU 동작 테스트 | 8개 통과 (`checks/unit_tests.log`) |
| 100개 학습 명령 | 시드·출력 경로·부모 가중치·fold 검증 |
| 학습 entrypoint 5개 | 전체 인자를 포함한 `--help` 실행 성공, 학습 미실행 |
| 지표 | 빈 클래스 0, 예시 60%, 이미지 크기와 무관한 이미지별 평균 |
| GPU 스케줄링 | 실제 subprocess JSON 모형으로 UUID 배정, 점유 GPU 제외, 완료 작업 재실행 방지 |
| 결과 비교 이미지 | 20장 생성 및 2×2 위치·흰색 정답·정답색 오류 픽셀 검사 |
| 체크포인트 | 12개 모두 CPU strict load 성공, 저장 FP16 / 로드 FP32, 모든 tensor 유한 |
| ImageNet 초기 가중치 | B4/HRNet offline cache CPU 로드 성공 |
| 배포 저장 | 12개 가중치 수·해시·교체 매핑·고정 설정·추론 `--help` 확인 |
| 실행 소스 스냅샷 | 별도 사본에서 preflight 통과 |
| 원본 입력 보존 | 데이터·부모 가중치·기존 기준 가중치·pretrained cache 761개 파일 SHA256 확인 |
| Shell | `bash -n run.sh start_nohup.sh` 통과 |
| 실행기 사전 점검 | `./run.sh check` 통과 |

`checks/cpu_model_load.json`, `checks/runtime_cpu.json`, `checks/preflight_*.json`에
검증 결과가 있습니다. 체크포인트 로드 검사에서 forward/backward 호출은 0회였습니다.
실제 학습 실행 폴더 `outputs/runs/`는 빌드 중 생성하지 않았습니다.

## 점수와 재현 범위

GitHub에는 공식 FP16 가중치가 없습니다. 로컬에 보존된 기존 23개판의 필요한 12개 가중치에서
별도 FP16 추론 사본을 생성했습니다(683,701,846 bytes). 공식 가중치 해시와 다르며
GitHub의 80.1255610% 점수를 재현한 것으로 취급하지 않습니다.
사용자가 시작하면 먼저 로컬 기준 점수를 실제 valid20에서 측정하고, 그 점수와 후보를 비교합니다.
현재 빌드 검증으로 학습 완료, GPU 메모리 실측 또는 최종 정확도를 보장하지 않습니다.

학습 레시피는 이전 실행기의 사본을 사용합니다. B4와 HRNet의 worker RNG 보정도 이어받았습니다.
새 소스 변경은 restart61의 불필요한 test 추론 생략뿐입니다. CUDA kernel의 비트 단위 결정성은
강제하지 않습니다. 마지막 조합은 고정 계수와 12개 가중치를 유지하며 validation 선택 결과로 기록합니다.

## 사용자 실행

```bash
cd /Data3/sppark/komap/experiments/model349_lite12_seed100_gpu4567_v1
./start_nohup.sh
```

결과는 새 `results/model_results_YYYYMMDD_번호/`에 보고서, 20장 `valid_mask_comparison/`,
84개 완료 후보의 비교 자료, 최종 12개 가중치와 소스 스냅샷으로 저장됩니다.
