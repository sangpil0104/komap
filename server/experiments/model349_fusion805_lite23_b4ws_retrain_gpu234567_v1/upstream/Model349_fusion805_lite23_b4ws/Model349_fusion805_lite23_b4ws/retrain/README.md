# 시드별 전체 재학습 (lite23 + B4 교체, 가중치 23개)

시드 하나를 정하면 23개 가중치를 모두 다시 학습하고, **lite23 설정을 그대로 고정한 채** valid20 채점과 test 예측까지 합니다. 설정·규칙·부분집합은 다시 고르지 않습니다(재선택하면 R이 80.02% → 79.29%로 떨어진 것을 확인했습니다).

## 필요한 것
- `Model349_fusion805_reproduction_code.zip`: 부모 패키지. 이 ZIP과 같은 커밋이거나 그 이후 판이어야 합니다. `b4ws/`, `config/b4ws_assets.json`, `infer_fusion.py --allow-retrained-refiner`가 들어 있어야 합니다.
- `Model349_train_route_code.zip`: R 학습 코드.
- 데이터: `DATA/train/{images,masks}`(70장), `DATA/valid/{images,masks}`(20장), `DATA/test/images`(선택).
- Python 3.10, torch 2.6 + cu118 (부모의 `requirements.txt`).

## 실행

```sh
bash retrain/retrain_all.sh SEED /path/Model349_fusion805 /path/train_route_root /path/DATA /path/runs/seedSEED 0,1,2,3
# 마지막 인자 = 쓸 GPU 목록. 서로 독립인 학습 묶음을 GPU에 나눠 동시에 돌림
```

- 끝난 단계는 건너뛰므로, 중간에 끊겨도 같은 명령으로 이어서 돌릴 수 있습니다.
- 결과는 `runs/seedS/pred_valid/metrics.json`(valid20 점수)와 `runs/seedS/pred_test/fusion/predicted_masks`(test 마스크)에 생깁니다.

## 단계와 시드 규칙

| 단계 | 스크립트 | 가중치 | 학습 데이터 | 시드 |
|---|---|---|---|---|
| R | `retrain_r.sh` | 14개 (B4 7 — 이 중 329 3개는 E와 공유, HRNet336/347, 269, 고해상도 ASPP 4) + 중간 단계 | train70. ResNet/ASPP 계열의 best.pt는 원래 파이프라인처럼 valid로 고름. B4는 원래 시편 fold(322 f3·4, 326 f1·2, 329 f1·3·4) | 기록된 시드 + 1000×S |
| E | `retrain_e.sh` | B4-329 4 fold + HRNet392 seed1·2 × 4 fold, 그리고 각 fold의 OOF 확률 | train70 시편 fold | B4 1000×S, HR s+1000×S |
| B4 추가 학습 | `retrain_b4ws.sh` | E의 B4 4개에서 출발, BN 고정, 2,000회 | 같은 fold 학습 이미지 | 819349 + 1000×S |
| F | `../refiner_retrain/` | 보정기 1개 | 재학습한 E의 train70 OOF 확률 | 81349 (고정) |
| 추론 | `infer_compact.py --allow-retrained` | lite23 고정 설정 | – | – |

- S=0이면 원래 기록된 시드입니다. 다만 GPU·환경 차이 때문에 원래 가중치와 비트 단위로 같지는 않습니다.
- lite23은 HR fold 중 seed1 2·4, seed2 1·4만 추론에 씁니다. 그래도 F 학습에 train70 전체의 OOF가 필요하므로 HR 8개를 모두 학습합니다.
- `--allow-retrained`는 가중치 SHA 검사를 건너뜁니다. 파일이 있는지와 추론 설정만 확인합니다.

## 비용과 시간 단축 (P100 기준 추정)
- **B4-329 공유:** R의 B4-329(fold 1·3·4)와 E의 B4-329(fold 4개)는 레시피와 시편 fold가 같습니다(held-out 시편 목록 4개 모두 일치, 학습 코드 동등). 그래서 4 fold를 한 번만 학습해 함께 씁니다. B4 3개(약 14시간)가 줄어 시드당 약 **76 GPU시간**입니다. 따로 학습하려면 `SHARE_329=0`을 줍니다.
- **GPU 병렬:** 학습 묶음 6개를 예상 시간 순으로 GPU에 나눠 배치합니다.

| 묶음 | 내용 | 예상 시간 |
|---|---|---|
| e_b4329_then_b4ws | B4-329 4 fold(+OOF) → B4 추가 학습 4 | 21h |
| r_b4 | R B4 322 ×2, 326 ×2 | 18h |
| r_resnet | R ResNet/ASPP 계열(중간 단계 포함) | 12h |
| e_hr1 / e_hr2 | E HRNet seed1 / seed2 × 4 fold(+OOF) | 각 9h |
| r_hrnet | R HRNet336 → 347 | 4h |

- 실제 걸리는 시간은 GPU 1장이면 약 76시간, 2장이면 약 39시간, **4장이면 약 21시간**, 6장이면 약 21시간입니다(가장 긴 묶음이 상한).
- 묶음 로그는 `OUT/logs/lane_*.log`에 남습니다. 끝나지 않은 묶음이 있으면 F 학습·추론으로 넘어가지 않습니다.

## 검증한 범위
- **F 재학습 (`refiner_retrain/`)**: 원래 E OOF로 학습 입력을 다시 만들었습니다. 원래 입력 70장과 배열 단위로 같았고, E OOF70 79.98586%도 재현됐습니다. 다시 학습한 F는 **원래 F와 SHA까지 같았습니다**(`932606f8…`, P100).
- **R·E 학습 명령**: 원래 파이프라인(`run_model349_pipeline.sh`, `run_model397_oof.sh`)의 명령을 시드만 바꿔 그대로 옮겼습니다. 같은 명령의 재학습 결과는 이전에 확인했습니다(R 80.05%, E 정상). 다만 **이 스크립트로 시드 전체를 처음부터 끝까지 돌린 적은 아직 없습니다.**
- **재학습 결과 예상**: 재학습 R이 80.02~80.05%였으므로 전체는 약 80.2~80.3%로 봅니다. 시드별 편차는 아직 모릅니다.
