#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${KOMAP_ROOT:-$PWD}"
PYTHON="${KOMAP_PYTHON:-python3}"
CHECKPOINT="$PROJECT_ROOT/outputs/model336_hrnet_best/last.pt"
OUT="${MODEL351_OUTPUT_DIR:-$PROJECT_ROOT/outputs/model351_model336_d4_tta/valid_eval_d4}"

[[ -s "$CHECKPOINT" ]] || { echo "Missing Model336 checkpoint: $CHECKPOINT" >&2; exit 2; }
[[ -d "$PROJECT_ROOT/data/valid/images" && -d "$PROJECT_ROOT/data/valid/masks" ]] || {
  echo "Expected valid images/masks under $PROJECT_ROOT/data" >&2; exit 2;
}
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite existing output: $OUT" >&2; exit 2; }
mkdir -p "$OUT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
if [[ -n "${KOMAP_VISIBLE_GPU:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$KOMAP_VISIBLE_GPU"
fi

"$PYTHON" -u "$BUNDLE_ROOT/experiments/model336_hrnet_best/evaluate_image_mean.py" \
  --checkpoint "$CHECKPOINT" --data-root "$PROJECT_ROOT/data" \
  --split valid --output-dir "$OUT" --device cuda:0 --batch-size 2 \
  --tta-mode d4 2>&1 | tee "$OUT/valid_eval_d4.log"
