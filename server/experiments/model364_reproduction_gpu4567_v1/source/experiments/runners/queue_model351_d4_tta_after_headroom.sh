#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/komap_model1
PY=/workspace/venv-komap/bin/python
GPU_ID=6
OUT="$ROOT/outputs/model351_model336_d4_tta"
EVAL_OUT="$OUT/valid_eval_d4"

mkdir -p "$OUT"
if [[ -e "$EVAL_OUT/valid_summary.json" || -e "$EVAL_OUT/valid_probabilities.npz" ]]; then
  echo "Refusing to overwrite existing Model351 valid artifacts under: $EVAL_OUT" >&2
  exit 1
fi

minimum_free=$((14 * 1024 * 1024 * 1024 + 512 * 1024 * 1024))
stable=0
echo "Waiting for stable P100 memory headroom before valid-only D4 inference." \
  | tee -a "$OUT/queue.log"
while (( stable < 3 )); do
  free_bytes=$(env -u CUDA_VISIBLE_DEVICES GPU_ID="$GPU_ID" "$PY" -c \
    'import os,torch; print(torch.cuda.mem_get_info(int(os.environ["GPU_ID"]))[0])')
  if (( free_bytes >= minimum_free )); then
    stable=$((stable + 1))
    echo "Stable-memory check $stable/3 passed" | tee -a "$OUT/queue.log"
  else
    stable=0
    echo "Waiting for shared-GPU headroom" | tee -a "$OUT/queue.log"
  fi
  if (( stable < 3 )); then sleep 60; fi
done

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY" -u model336_hrnet_best/evaluate_image_mean.py \
  --checkpoint "$ROOT/outputs/model336_hrnet_best/last.pt" \
  --data-root "$ROOT/data" \
  --split valid \
  --output-dir "$EVAL_OUT" \
  --device cuda:0 \
  --batch-size 2 \
  --tta-mode d4 \
  2>&1 | tee "$OUT/valid_eval_d4.log"

echo "DONE: Model351 8-way dihedral TTA valid inference." \
  | tee -a "$OUT/queue.log"
