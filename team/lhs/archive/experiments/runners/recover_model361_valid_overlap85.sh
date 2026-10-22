#!/usr/bin/env bash
# Resume only Model361 validation inference after its scale-3 stride-1 issue.
# Training is already complete and its checkpoint is preserved.
set -euo pipefail

ROOT="${KOMAP_ROOT:-/workspace/komap_model1}"
PY="${KOMAP_PYTHON:-/workspace/venv-komap/bin/python}"
GPU="${KOMAP_GPU:-4}"
OUT="$ROOT/outputs/model361_hrnet_scale3_finetune"
EVAL="$OUT/valid_eval"
[[ -s "$OUT/last.pt" ]] || { echo "Missing Model361 checkpoint" >&2; exit 2; }
[[ ! -e "$EVAL/valid_summary.json" && ! -e "$EVAL/valid_probabilities.npz" ]] || {
  echo "Refusing to overwrite existing Model361 evaluation artifacts" >&2; exit 2;
}

export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PY" -u "$ROOT/model336_hrnet_best/evaluate_image_mean.py" \
  --checkpoint "$OUT/last.pt" --data-root "$ROOT/data" --split valid \
  --output-dir "$EVAL" --device cuda:0 \
  --patch-size 512 --overlap 85 --input-scale 3.0 --batch-size 2 \
  > "$OUT/eval.log" 2>&1
