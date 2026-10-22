# Model338에 필요한 외부 파일

원본 Git 저장소는 연구 코드와 기록을 포함하며, 학습된 체크포인트·예측 확률·캐시를 포함하지 않습니다.
Model338의 **기록된 image mIoU 최적 경로**는 다음 4개 파일로 실행합니다.
현재 이 서버의 새 실험 폴더에는 4개 모두 없습니다. 파일 없이 Model338 실제 예측이나 점수 재현은 할 수 없습니다.

| 구성 | 원래 연구 환경 경로 (`/workspace/komap_model1/` 기준) | 이 실험의 기본 경로 |
|---|---|---|
| Model183 캐시 | `outputs/model183_metric_calibration/entries.pkl` | `artifacts/model338/entries.pkl` |
| Model269 확률 | `outputs/model269_fullstats_1x_extended/valid_probs.npz` | `artifacts/model338/model269_valid_probs.npz` |
| Model322 4-fold 확률 | `outputs/model322_team_b4_group_oof/valid_ensemble/probabilities.npz` | `artifacts/model338/model322_probabilities.npz` |
| Model326 4-fold 확률 | `outputs/model326_hybrid_fullstats_group_oof/valid_ensemble/probabilities.npz` | `artifacts/model338/model326_probabilities.npz` |

원본 연구 환경의 파일을 복사하거나, `configs/model338.json`의 `artifacts`에 실제 절대 경로를 지정합니다.
`.pkl`은 원본 연구 환경에서 받은 신뢰할 수 있는 캐시를 사용합니다.
Model269 채널은 `[Al3Ni, Primary Si, Al, Eutectic Si]`, Model322/326 채널은 `[Al, Al3Ni, Primary Si, Eutectic Si]`입니다.
각 NPZ에는 현재 검증 이미지 20장의 `(4,H,W)` 확률 배열이 필요합니다. 키는 이미지 stem이며 `_image` 접미사는 있어도 됩니다.

`entries.pkl`에는 이미지 ID가 없으므로 정답 마스크와 영상 밝기 guide를 대조하여 이미지에 연결합니다.
순서만 믿고 연결하지 않으며, 데이터가 다르거나 대응이 모호하면 실행을 중단합니다.
정답은 파일 대응 확인과 평가에만 사용하며, 최종 예측 함수에는 정답을 전달하지 않습니다.

```bash
cd /Data3/sppark/komap/experiments/lhs
./run.sh check338 --deep
./run.sh evaluate338
```

검사 시 누락 파일이 있으면 종료 코드 2와 목록을 반환합니다. 예측도 필수 입력이 없으면 결과 폴더를 만들기 전에 중단합니다.
기본 설정을 보존하려면 설정 사본을 만들어 두 명령에 `--config /절대/경로/model338.json`을 지정할 수 있습니다.

선택된 최종 경로에는 Model195/220 확률 파일이 필요하지 않습니다. 원본 전체 경로 탐색은 다른 후보 때문에 이 파일들도 읽지만, 이 실행기는 선택된 Model338 설정만 실행합니다.
원본 기록의 image mIoU는 **79.33996345357988%**입니다. 이 서버에서 재현한 점수가 아니며, 검증 세트에서 경로를 선택한 기록입니다.
새 Model326 학습의 `outputs/runs/.../evaluations/.../probabilities.npz`를 사용할 수도 있지만 원래 Model326 확률과 동일함을 보장하지 않습니다. 따라서 원본 Model338 점수의 동일 재현을 주장할 수 없습니다.
Model326을 학습하더라도 Model183 캐시, Model269, Model322 파일은 별도로 필요합니다.
