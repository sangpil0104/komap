#!/bin/bash
# Train the released model, evaluate it on valid, and predict the test split.
# Usage: scripts/reproduce.sh <data-root> [output-dir]
set -euo pipefail
DATA_ROOT=${1:?usage: scripts/reproduce.sh <data-root> [output-dir]}
OUT=${2:-runs/best}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p "$OUT"
uv run python train.py --data-root "$DATA_ROOT" --output-dir "$OUT" | tee "$OUT/train.log"
uv run python evaluate.py --checkpoint "$OUT/last.pt" --data-root "$DATA_ROOT" --split valid | tee "$OUT/eval_valid.log"
uv run python predict.py --checkpoint "$OUT/last.pt" --images-dir "$DATA_ROOT/test/images" --output-dir "$OUT/test_predictions"
