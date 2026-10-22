#!/usr/bin/env bash
# Predict with the published 23 weights (lite23 + B4ws, valid20 80.5464%).
#   bash run_infer.sh WEIGHTS INPUT_IMAGES OUTPUT [GPU] [--allow-retrained]
# WEIGHTS: folder laid out as in weights_manifest.json (WEIGHTS/R/... and WEIGHTS/assets/...)
# Output masks: OUTPUT/fusion/predicted_masks (palette PNG) and OUTPUT/fusion/predicted_labels (0..3)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); W=$(cd "$1" && pwd); IN=$2; OUT=$3; GPU=${4:-0}; EXTRA=${5:-}
PY=${PYTHON:-python}; RW=$HERE/Model349_fusion805/Model349_exh14/weights
if [[ -L $RW ]]; then [[ $(readlink "$RW") == "$W/R" ]] || { echo "$RW already points to $(readlink "$RW")"; exit 2; }
elif [[ -e $RW ]]; then echo "$RW exists and is not a link; remove it first"; exit 2
else ln -s "$W/R" "$RW"; fi
CUDA_VISIBLE_DEVICES=$GPU "$PY" "$HERE/Model349_fusion805_lite23_b4ws/infer_compact.py" --config "$HERE/Model349_fusion805_lite23_b4ws/selected_config_b4ws.json" \
  --parent-package "$HERE/Model349_fusion805" --assets "$W/assets" --input "$IN" --output "$OUT" --gpus 0 $EXTRA
echo "masks: $OUT/fusion/predicted_masks"
