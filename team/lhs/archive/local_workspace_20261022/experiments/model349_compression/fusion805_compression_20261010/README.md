# fusion805 lite25 경량화 탐색 실행 코드

선택한 후보는 HRNet seed2 fold2·3을 제거한 25개 구성입니다. valid20 80.515892%(원래80.500689%, +0.015203%p), 최악 영상−0.071187%p로 사전 기준을 통과했습니다. 가중치 용량4.554→3.936GB(−13.57%)입니다. 같은 E의 F 없는1단계는24개/80.488844%입니다. 기존 릴리스는 교체하지 않습니다.

가중치와 데이터는 포함하지 않습니다. 기존 `Model349_fusion805_reproduction_code.zip`을 풀어 부모 패키지를 준비하고, 해당 패키지의 manifest와 동일한 외부 가중치를 준비해야 합니다. 원래 릴리스와 ZIP은 변경하지 않았습니다.

`report.md`와 `result.json`, `fold_prune/result.json`에서 성능 및 채택 기준을 확인하세요. 모든 후보는 반복 사용된 valid20의 탐색 결과이며 독립 일반화나 test 성능을 보장하지 않습니다. CPU 캐시 재생 및 HR seed2 분기의 원본 영상 검증을 구분합니다. 전체 추론 속도는 측정하지 않았습니다.

## 원본 영상 추론

부모 패키지 검증 환경/requirements를 사용합니다. `--layout`을 명시해야 하며 자동으로 특정 탐색 후보를 기본 선택하지 않습니다.

```sh
python infer_lite.py \
  --parent-package /path/Model349_fusion805 \
  --assets /path/external_assets \
  --input /path/images --output /path/new_run \
  --layout hr2_subset --hr2-folds 1 4 --stage 2 --gpus 0
```

`--stage 1`은 F 없는 R+E이며, `--stage 2`는 원래 fusion805 두 단계입니다. 기존 R14 출력이 있으면 `--release /path/release_run`으로 재사용할 수 있습니다. 각 후보에서 필요한 checkpoint 경로·SHA·용량은 `asset_layouts.json`에 기록했습니다. R의 `weights/` 경로는 부모 패키지 `Model349_exh14/` 아래이며 E/F 경로는 `--assets` 아래입니다. 제외한 E 체크포인트는 실제로 검증/로드/추론하지 않습니다.

- `hr2_subset --hr2-folds 1 4`: 선택한 lite25 후보(1단계24개).
- `seed1`: seed2 전체 제거, B4 0.5 + HR seed1 0.5.
- `seed2`: seed1 전체 제거, B4 0.5 + HR seed2 0.5.
- `drop_seed2_fold4`: seed2 유지 fold1·2·3.
- `drop_seed2_folds34`: seed2 유지 fold1·2.
- `drop_seed2_folds234`: seed2 유지 fold1.

부분 fold 제거는 B4 0.5 + HR seed1 0.25 + HR seed2 0.25를 유지합니다. seed2 내부 평균만 남은 fold 수로 나누고, 가족별 FP16 반올림 및 동률 클래스 우선순위를 보존합니다. F는 변경된 E 확률에서 다시 계산합니다. 추론 실행기는 GT를 읽지 않습니다.

`replay.py`, `infer_hr_folds.py`, `score.py`는 기존 서버 경로를 기록한 실험 감사 코드입니다. 새로운 영상 추론은 위의 `infer_lite.py`를 사용하세요.
