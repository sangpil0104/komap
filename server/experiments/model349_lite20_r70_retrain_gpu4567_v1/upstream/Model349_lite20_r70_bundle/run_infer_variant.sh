#!/usr/bin/env bash
# Predict with a pruned variant of lite23 + B4ws.
#   bash run_infer_variant.sh VARIANT WEIGHTS INPUT_IMAGES OUTPUT [GPU] [--allow-retrained]
# VARIANT: lite21 | lite20 | lite19   (lite23 itself: run_infer.sh)
# WEIGHTS: same layout as run_infer.sh (WEIGHTS/R/..., WEIGHTS/assets/...). Unused checkpoints may stay in place.
set -euo pipefail
# GPU is a physical index; infer_lite/infer_expert set CUDA_VISIBLE_DEVICES themselves and require the GPU to be idle.
V=$1; shift
case $V in lite21|lite20|lite19) ;; *) echo "unknown variant $V"; exit 2;; esac
HERE=$(cd "$(dirname "$0")" && pwd); W=$(cd "$1" && pwd); IN=$2; OUT=$3; GPU=${4:-0}; EXTRA=${5:-}
PY=${PYTHON:-python}; RW=$HERE/Model349_fusion805/Model349_exh14/weights
if [[ -L $RW ]]; then [[ $(readlink "$RW") == "$W/R" ]] || { echo "$RW already points to $(readlink "$RW")"; exit 2; }
elif [[ -e $RW ]]; then echo "$RW exists and is not a link; remove it first"; exit 2
else ln -s "$W/R" "$RW" 2>/dev/null || [[ $(readlink "$RW") == "$W/R" ]]; fi
"$PY" "$HERE/Model349_fusion805_lite23_b4ws/infer_compact.py" --config "$HERE/Model349_fusion805_lite23_b4ws/selected_config_$V.json" \
  --parent-package "$HERE/Model349_fusion805" --assets "$W/assets" --input "$IN" --output "$OUT" --gpus "$GPU" $EXTRA
echo "masks: $OUT/fusion/predicted_masks"
