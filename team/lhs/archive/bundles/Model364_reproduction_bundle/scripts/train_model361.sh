#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${KOMAP_ROOT:-$PWD}"
PYTHON="${KOMAP_PYTHON:-python3}"
OUT="${MODEL361_OUTPUT_DIR:-$PROJECT_ROOT/outputs/model361_hrnet_scale3_finetune}"
PARENT="$PROJECT_ROOT/outputs/model336_hrnet_best/last.pt"

[[ -s "$PARENT" ]] || { echo "Missing Model336 parent checkpoint: $PARENT" >&2; exit 2; }
[[ -d "$PROJECT_ROOT/data/train/images" && -d "$PROJECT_ROOT/data/train/masks" ]] || {
  echo "Expected train images/masks under $PROJECT_ROOT/data" >&2; exit 2;
}
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite existing output: $OUT" >&2; exit 2; }
mkdir -p "$OUT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
if [[ -n "${KOMAP_VISIBLE_GPU:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$KOMAP_VISIBLE_GPU"
fi

"$PYTHON" -u "$BUNDLE_ROOT/experiments/model336_hrnet_best/train.py" \
  --data-root "$PROJECT_ROOT/data" --output-dir "$OUT" \
  --device cuda:0 --architecture UnetPlusPlus --encoder-name tu-hrnet_w48 \
  --encoder-weights imagenet --init-checkpoint "$PARENT" \
  --patch-size 512 --input-scale 3.0 --samples-per-image 12 \
  --batch-size 4 --accumulation-steps 2 --num-workers 4 \
  --epochs 12 --lr 1e-4 --encoder-lr-mult 0.3 \
  --weight-decay 1e-4 --boundary-head-weight 1.0 \
  --seed 361 --val-every 0 \
  2>&1 | tee "$OUT/train.log"

"$PYTHON" -u "$BUNDLE_ROOT/experiments/model336_hrnet_best/evaluate_image_mean.py" \
  --checkpoint "$OUT/last.pt" --data-root "$PROJECT_ROOT/data" \
  --split valid --output-dir "$OUT/valid_eval" --device cuda:0 \
  --patch-size 512 --overlap 170 --input-scale 3.0 --batch-size 2 \
  2>&1 | tee "$OUT/valid_eval.log"
