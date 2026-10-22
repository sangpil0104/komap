#!/usr/bin/env bash
# 예측: 가중치 13개로 이미지 폴더의 마스크를 만든다(정답 불필요).
#   bash run_predict.sh WEIGHTS IMAGES OUT [GPU]
#   WEIGHTS : R/ E/ F/ 를 가진 가중치 폴더 (배치는 README 참고; run_retrain.sh 결과는 runs/retrain/model/weights)
#   IMAGES  : 예) data/test/images      OUT : 새 결과 폴더(없어야 함)      GPU : GPU 번호(기본 0)
#   ALLOW_RETRAINED=1 : 공개 가중치의 SHA256 확인을 건너뛴다(새로 학습한 가중치를 쓸 때)
# 결과: OUT/fusion/predicted_masks (색 마스크), OUT/fusion/predicted_labels (0~3 라벨)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd); PY=${PYTHON:-python}
WEIGHTS=$(cd "$1" && pwd); IMAGES=$(cd "$2" && pwd); OUT=$3; GPU=${4:-0}
[[ -e $OUT ]] && { echo "OUT already exists: $OUT"; exit 1; }
mkdir -p "$(dirname "$OUT")"; OUT=$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
# 코드 사본 + 가중치 심볼릭 링크 트리(원본 가중치 폴더는 바꾸지 않음)
cp -r "$ROOT/inference" "$TMP/inference"
cp -rs "$WEIGHTS/R" "$TMP/inference/r/weights"
cp -rs "$WEIGHTS/E" "$TMP/assets"
cp -rs "$WEIGHTS/F/F_refiner" "$TMP/assets/F_refiner"
extra=(); [[ ${ALLOW_RETRAINED:-0} == 1 ]] && extra=(--allow-retrained)
"$PY" "$TMP/inference/predict.py" --parent-package "$TMP/inference" --assets "$TMP/assets" --input "$IMAGES" --output "$OUT" --gpus "$GPU" "${extra[@]}"
echo "masks -> $OUT/fusion/predicted_masks"
