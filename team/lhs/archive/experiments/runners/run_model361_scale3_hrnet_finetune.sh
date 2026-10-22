#!/usr/bin/env bash
# One-P100, short supervised scale-3 fine-tune of Model336; no valid-mask
# checkpoint selection during training. Run inside /workspace/komap_model1.
set -euo pipefail

ROOT="${KOMAP_ROOT:-/workspace/komap_model1}"
PY="${KOMAP_PYTHON:-/workspace/venv-komap/bin/python}"
GPU="${KOMAP_GPU:-4}"
OUT="$ROOT/outputs/model361_hrnet_scale3_finetune"
SOURCE="$ROOT/outputs/model336_hrnet_best/last.pt"

[[ -s "$SOURCE" ]] || { echo "Missing Model336 checkpoint: $SOURCE" >&2; exit 2; }
[[ -d "$ROOT/data/train/images" && -d "$ROOT/data/valid/images" ]] || {
  echo "Missing labeled data under $ROOT/data" >&2; exit 2;
}
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 2; }
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PY" -u "$ROOT/model336_hrnet_best/train.py" \
  --data-root "$ROOT/data" --output-dir "$OUT" \
  --device cuda:0 --architecture UnetPlusPlus --encoder-name tu-hrnet_w48 \
  --encoder-weights imagenet --init-checkpoint "$SOURCE" \
  --patch-size 512 --input-scale 3.0 --samples-per-image 12 \
  --batch-size 4 --accumulation-steps 2 --num-workers 4 \
  --epochs 12 --lr 1e-4 --encoder-lr-mult 0.3 \
  --weight-decay 1e-4 --boundary-head-weight 1.0 \
  --seed 361 --val-every 0 \
  2>&1 | tee "$OUT/train.log"

"$PY" -u "$ROOT/model336_hrnet_best/evaluate_image_mean.py" \
  --checkpoint "$OUT/last.pt" --data-root "$ROOT/data" --split valid \
  --output-dir "$OUT/valid_eval" --device cuda:0 \
  --patch-size 512 --overlap 85 --input-scale 3.0 --batch-size 2 \
  2>&1 | tee "$OUT/eval.log"
