# lite23 + B4 BN 고정 추가 학습 (`Model349_fusion805_lite23_b4ws`)

Codex의 lite23 구성(80.5172705%, 가중치 23개)에서 **E 안의 B4-329 4 fold만 BN 고정 추가 학습판으로 바꾼** 구성입니다. 가중치 수(23개)와 용량은 같습니다.

| 구성 | valid20 | lite23 대비 |
|---|---:|---:|
| lite23 | 80.5172705% | 기준 |
| **lite23 + B4 교체** | **80.5464316%** | **+0.0292%p** (시편 bootstrap 95% [+0.003, +0.067], 14/20장 개선, 최악 −0.064%p) |

- **바뀐 것:** E의 B4-329 fold k를 같은 fold의 train70 학습 이미지로 2,000회 더 학습한 가중치(`model329_ws/foldK`)입니다.
  - 학습 조건은 lr 3e-5, seed 819349이고, BatchNorm running 통계는 고정했습니다.
  - R 14개, HR seed1 fold2·4, HR seed2 fold1·4, 비율 0.5/0.25/0.25, 덮어쓰기 규칙, F는 lite23과 같습니다.
- **근거:** 학습에 쓰지 않은 시편 기준(OOF70)으로 B4 단독이 +0.376%p(51/70장)입니다. BN을 고정하지 않으면 −0.104%p였습니다. valid·test 정답은 학습에 쓰지 않았습니다.
- **한계:** 고정 후보 하나의 결과라 후보 선택 편향은 없습니다. 다만 valid20을 반복해서 써 왔으므로 독립 일반화 점수는 아닙니다. test에는 정답이 없습니다.

## 준비

1. 부모 패키지: lhs의 `Model349_fusion805_reproduction_code.zip`(커밋 `19d56d4` 이후)을 풉니다. `config/b4ws_assets.json`과 `b4ws/`가 들어 있어야 합니다.
2. 가중치: `selected_assets_b4ws.json`의 경로와 SHA대로 둡니다.
   - R 14개는 `부모/Model349_exh14/` 아래에 둡니다.
   - E·F는 `assets/` 아래(`model329_ws/foldK/last.pt`, `model392/...`, `refiner/epoch8.pt`)에 둡니다.
   - `model329_ws`는 부모의 `b4ws/train_b4ws.sh`로 직접 만들 수 있습니다(fold당 P100 약 34분). 직접 학습한 가중치는 SHA가 다르므로 `selected_assets_b4ws.json`의 해당 4줄을 새 SHA로 바꿔야 합니다.

## 실행

```sh
python infer_compact.py --config selected_config_b4ws.json \
  --parent-package /path/Model349_fusion805 --assets /path/assets \
  --input /path/images --output /path/new_output --gpus 0
# 이미 추론한 R 결과를 다시 쓰려면 --release /path/release (입력·설정 해시 검사)
```

- `--config`를 주지 않으면 원래 lite23(`selected_config.json`)으로 동작합니다.
- 바뀐 코드는 `infer_expert.py`뿐입니다. 설정의 `expert.b4_dir`(`model329` 또는 `model329_ws`)에 따라 B4 폴더와 SHA 기준을 고릅니다. 기본값은 원래와 같습니다.

## 검증과 결과 (`results_b4ws/`)

- `stored_probability_candidate.json`: lite23의 저장 E 확률에서 B4만 교체한 결과입니다. 대조군은 lite23 80.5172705%를 0픽셀 차이로 재현했습니다.
- `fresh_expert_valid20_metrics.json`: E를 원본 영상에서 새로 추론해 얻은 80.5464316%입니다. 저장 확률 결과와 0픽셀 차이입니다.
- 이 ZIP 자체를 새 폴더에 풀어 끝까지 실행하는 패키지 검증은 하지 않았습니다(사용자 결정, 팀원 실행). 위 80.5464316%는 같은 가중치와 같은 E 추론 코드(B4 SHA만 교체)로 원본 영상부터 새로 추론해 확인한 값입니다.
- `combo_with_codex_seed2warm.json`: Codex HR seed2 추가 학습판까지 함께 넣은 조합(80.5496%)입니다. B4 교체 단독보다 +0.003%p뿐이어서 채택하지 않았습니다.

## 시드별 전체 재학습

- `retrain/`: 시드 하나로 23개 가중치를 모두 다시 학습하고, lite23 고정 설정으로 valid 채점과 test 예측까지 합니다(`retrain/retrain_all.sh`). 자세한 내용은 `retrain/README.md`에 있습니다.
- `refiner_retrain/`: F(보정기) 재학습 스크립트입니다. 원래 F와 SHA까지 같게 재현되는 것을 확인했습니다.
- 재학습한 가중치로 추론할 때는 `infer_compact.py --allow-retrained`를 씁니다. SHA 대신 파일 존재만 확인합니다.

## 팀원 실행용 실험 (미검증)

- `full70_experiment/`: E의 B4를 train70 전체로 BN 고정 추가 학습하는 실험입니다. 학습 스크립트, 설정(`selected_config_b4full70.json`), 판단 기준이 들어 있습니다. 아직 실행하지 않아 점수가 없습니다.

이 아래의 원래 lite23 설명은 `report.md`에 그대로 두었습니다.
