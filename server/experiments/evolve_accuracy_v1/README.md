# evolve_accuracy_v1 — accuracy_v1 baseline + evolve 기법 누적 적용 (8단계)

## 목적

`experiments/accuracy_v1/baseline`에 보존된 프로젝트의 기존 공식 모델(0919.md 기준
`accuracy_v1` baseline, plain **78.4729%** / D4 **78.6865%**, `experiments/accuracy_v1/BUILD_REPORT.md`
참고)을 부모 체크포인트로 삼아, `experiments/evolve_v1`이 정의한 8가지 evolve 기법을
**기법 1개만 적용한 모델부터 8개를 전부 적용한 모델까지 누적으로 하나씩 늘려가며** 총 8개
모델을 만든다.

`evolve_v1`은 hundred_v1의 top-10 parent 각각에 기법을 하나씩만 켜서 비교하거나(v1~v8),
경계/텍스처 목적의 heavy 조합 2개(v9/v10)를 만드는 실험이다. 이 실험은 그와 달리 **부모 1개
고정, 기법을 정해진 순서(1→8)로 하나씩 누적 추가**해서 "기법을 더할수록 얼마나 나아지는가"를
직접 보기 위한 것이다. 부모가 다르므로(accuracy_v1 baseline vs hundred_v1 top-10) evolve_v1과는
완전히 별도 실험이며, evolve_v1의 코드·설정·진행 중인 100개 스위트에는 전혀 손대지 않았다.

## 기법 누적 순서와 이유

evolve_v1의 기법 번호(`experiments/evolve_v1/evolvelab/catalog.py` 주석 기준) 순서를 그대로 따랐다.

| 단계 | 실험 이름 | 이번에 추가된 기법 | 학습 예산 |
|---|---|---|---:|
| 1 | `e001_c1_hires_decoder` | 1 hires_decoder | full_frame 900 |
| 2 | `e002_c2_add_crop_mix` | +2 crop_mix (sampling만, patch 단계 필요) | patch 400 + full_frame 500 |
| 3 | `e003_c3_add_hires_cnn_branch` | +3 hires_cnn_branch | patch 400 + full_frame 500 |
| 4 | `e004_c4_add_boundary_aux_plus` | +4 boundary_aux_plus (loss.boundary=.08) | patch 400 + full_frame 500 |
| 5 | `e005_c5_add_fullres_refine` | +5 fullres_refine | patch 400 + full_frame 500 |
| 6 | `e006_c6_add_composition_head` | +6 composition_head (loss.composition=.08) | patch 400 + full_frame 500 |
| 7 | `e007_c7_add_texture_branch` | +7 texture_branch | patch 400 + full_frame 500 |
| 8 | `e008_c8_all_techniques` | +8 complex_expert → **8개 전부 적용** | patch 400 + full_frame 500 |

기법 2(crop_mix)는 patch 단계에서만 효과가 있어(`evolve_v1/evolvelab/dataset.py` 참고), 1단계는
evolve_v1의 v1과 동일하게 full_frame만, 2단계부터는 patch→full_frame 2단계 일정을 쓴다. 모든
단계를 동일하게 총 900 updates로 맞춰서, 예산 차이가 아니라 기법 자체의 효과를 비교할 수 있게
했다(evolve_v1의 heavy 조합 v9/v10처럼 예산을 늘리지 않음). light 레벨만 사용했다(heavy는 이번
비교 범위 밖).

각 단계는 직전 단계가 아니라 **매번 accuracy_v1 baseline에서 새로 warm-start**한다(기법이
"단독으로 누적됐을 때"의 효과를 보기 위함이며, evolve_v1의 각 v1~v8도 같은 방식으로 항상 원래
parent에서 시작한다). 새로 추가되는 서브모듈(`additions.*`)은 0으로 초기화되므로, 학습 시작
시점의 예측은 항상 accuracy_v1 baseline과 동일하다(`evoacclab/models.py`의 zero-init 관례).

## 보존 및 독립성 (AGENTS.md 준수)

- 기존 `accuracy_v1/baseline/run/best.pt`, `data/manifest.csv`, `data/folds.json`은 읽기만
  하며, 해시를 `preservation.json`에 기록했다. `run.py verify` 또는 `run.py preflight`가 매번
  이 파일들이 바뀌지 않았는지 확인한다.
- `evolve_v1`의 코드·설정·진행 중인 100개 스위트는 전혀 수정하지 않았다. 이 실험은 자체
  `vendor/alphase`(엔진 사본)와 `evoacclab/`(기법 구현 사본, `evolve_v1/evolvelab`을 부모
  checkpoint/카탈로그만 바꿔 복제) 위에서 완전히 독립적으로 동작한다.
- 결과는 전부 `experiments/evolve_accuracy_v1/outputs/` 아래에만 쓰인다.
- 진행률은 터미널 `tqdm`(학습 루프·suite 진행 모두)로만 표시한다. `evoacclab/progress.py`는
  의도적으로 localhost 서버/웹 대시보드를 만들지 않는다(AGENTS.md 지침).

## 사전 검증 완료

`build_model()`로 8개 설정 전부를 accuracy_v1 baseline 위에서 실제로 빌드하고 CPU 더미
forward pass까지 확인했다(파라미터 로드, split/palette/class 일치, technique 조합별 출력
shape). 실제 학습(GPU)은 아직 수행하지 않았다 — 아래 명령으로 직접 실행하면 된다.

## 실행 방법

```bash
cd /Data3/sppark/komap
conda activate komap

# (선택, 권장) 기법 2의 hard_example crop 항목을 위한 부모 오류맵 생성 — 없어도 자동으로
# boundary_rich로 대체되어 학습은 되지만, 생성해두면 기법 2 효과를 더 정확히 볼 수 있다.
python -B experiments/evolve_accuracy_v1/run.py build-hard-maps

# 설정 검증 (보존 파일 확인 + configs/resolved/*.yaml 생성 + 중복 설정 검사)
python -B experiments/evolve_accuracy_v1/run.py preflight

# 개별 학습 (원하는 GPU를 CUDA_VISIBLE_DEVICES로 고르세요; 현재 GPU0은 evolve_v1이,
# GPU1/3/4는 다른 사용자가 쓰는 중입니다 — nvidia-smi로 먼저 확인)
CUDA_VISIBLE_DEVICES=<idx> python -B experiments/evolve_accuracy_v1/run.py train e001_c1_hires_decoder
CUDA_VISIBLE_DEVICES=<idx> python -B experiments/evolve_accuracy_v1/run.py train e002_c2_add_crop_mix
# ... e003 ~ e008 동일 패턴

# 중단된 학습 재개
CUDA_VISIBLE_DEVICES=<idx> python -B experiments/evolve_accuracy_v1/run.py train e004_c4_add_boundary_aux_plus --resume

# 8개를 순서대로 한 번에 (완료된 건 재사용, 실패해도 기록 후 다음으로 진행)
CUDA_VISIBLE_DEVICES=<idx> python -B experiments/evolve_accuracy_v1/run.py suite

# 개별/전체 평가 및 리포트 (suite가 자동으로도 실행하지만 필요시 수동 실행)
python -B experiments/evolve_accuracy_v1/run.py evaluate e001_c1_hires_decoder
python -B experiments/evolve_accuracy_v1/run.py evaluate e001_c1_hires_decoder --d4
python -B experiments/evolve_accuracy_v1/run.py report
```

동작 확인용 짧은 실행은 아무 명령에나 `--smoke`를 붙이면 된다(`outputs/smoke_runs`,
`outputs/smoke_evaluations`에 본 실험과 분리되어 저장됨).

## 산출물

- 체크포인트: `outputs/runs/<experiment>/best.pt`, `best_trained.pt`, `last.pt`
- 검증 지표: `outputs/evaluations/<experiment>/{plain,d4}/metrics_summary.json`
- 전체 완료 후 리포트: `outputs/reports/results.md`, `results.csv` (8개 누적 단계 mIoU 순위표)
- 학습 로그: `outputs/runs/<experiment>/resolved_config.yaml`, `metrics.jsonl`, `tensorboard/`

8개 학습이 모두 끝나면 알려주세요 — `results/model_results_0923_2/` 아래에 결과 정리 md와
`results/valid_mask_comparison/`에 이번 8개 모델의 검증셋 예측 비교 이미지를 생성하겠습니다.
