# Fusion805 재학습 준비 검증

준비 완료. **실제 학습과 실제 데이터 GPU 추론을 시작하지 않았습니다.**
시작은 사용자가 `./start_nohup.sh`를 실행해야 합니다.

## 통과한 검사

| 검사 | 결과 기록 | 확인 내용 |
|---|---|---|
| GitHub 원본·기존 파일 보존 | `checks/preservation.json`, `provenance.json` | 공개 ZIP SHA256, 원본 소스 manifest, 직전 실험 실행기/학습 소스 불변 |
| 전체 CLI·의존성 | `checks/design.json` | 실제 명령 43개 인자 파싱, 학습 27개, R 14 + E 12 + F 1, 설정·부모 연결 |
| 작업 배정·종료 | `checks/scheduler.json` | 합성 작업 43개, GPU 2~7 최대 6개, 의존 순서, 실패 시 자식 종료, 필수 출력 누락 차단, 사용 중 GPU 대기 |
| 부모·ImageNet 초기화 | `checks/models_cpu.json` | CPU에서 Model98/257 및 B4/HRNet ImageNet cache 로딩, forward/backward 없음 |
| OOF·F 입력 | `checks/features_cpu.json` | 합성 70장 OOF·정답 클래스·gray normalization, crop 1,120개, F 11,876 parameters, zero-refiner identity, 잘못된 OOF 목록 거부 |
| 두 단계 추론·결과 저장 | `checks/export_cpu.json` | 합성 데이터에서 실제 stage1/stage2/scorer 실행, 동률 우선순위, 독립 image IoU 대조, 보고서·비교 이미지 총 40장 |
| 소스 스냅샷·모델 패키지 | `checks/runtime_cpu.json` | 임시 fixture로 27개 패키지 생성·해시 검증, E/F 13개 strict CPU load, HRNet 전처리 metadata 보존, 추론 CLI |
| 실제 입력·환경 사전 점검 | `checks/preflight_*.json`, `checks/preflight.log` | conda komap, GPU 2~7 가시성, train70/valid20 크기·색상·해시, R/E 시편 분할·링크·소스 해시 |

실행 검증에는 임시 폴더의 합성 데이터와 기존 checkpoint fixture만 사용했습니다.
fixture checkpoint는 새 학습 결과가 아니며 실행 후 임시 폴더를 정리했습니다.
합성 F 검사에서 CPU forward를 수행했으나 optimizer update는 0회입니다.
실제 학습의 GPU 메모리 사용량·소요 시간·성능은 학습 후 확인할 수 있습니다.

## 실행 환경

- Python: `/home/sppark/miniconda3/envs/komap/bin/python`
- torch 2.7.1+cu128, torchvision 0.22.1+cu128
- segmentation-models-pytorch 0.5.0, timm 1.0.29
- numpy 2.4.6, albumentations 2.0.8, tqdm 4.70.1
- 패키지 설치/업그레이드 없이 현재 `komap` 환경 사용
- E fold별 train/heldout: 52/18, 52/18, 53/17, 53/17 (70 images / 59 specimen groups)

## 보존과 결과 해석

소스·설정은 `code_manifest.json`, 데이터·선행 부모는 `input_manifest.json`,
두 학습 경로의 fold는 `fold_manifest.json`과 `expert_fold_manifest.json`으로 고정했습니다.
학습 시작 시 소스 사본과 패키지 환경을 실행 폴더에 저장합니다.
R의 초기 Model98·257을 제외한 완성된 이전 모델을 학습 출력으로 재사용하지 않습니다.
실제 학습 27개와 모든 준비/추론 단계가 끝나야 최종 결과를 저장합니다.

기존 실험처럼 결과 경로는 `results/model_results_YYYYMMDD_다음번호/`입니다.
GitHub 현재 설명의 대표 구성 stage1을 루트 보고서와 비교 이미지에 저장하고,
요청한 fusion805 stage2도 `stage2/`에 따로 저장합니다. 새 점수로 대표 구성을 다시 고르지 않습니다.
새 가중치의 점수는 원본 공개 점수와 다를 수 있습니다.
