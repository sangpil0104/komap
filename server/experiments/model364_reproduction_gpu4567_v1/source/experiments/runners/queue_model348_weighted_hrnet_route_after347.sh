#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/komap_model1
PY=/workspace/venv-komap/bin/python
OUT="$ROOT/outputs/model348_model347_into_341_route"
MODEL347_QUEUE="$ROOT/outputs/model347_hrnet_inverse_ce_finetune/queue.log"
MODEL347_PROBS="$ROOT/outputs/model347_hrnet_inverse_ce_finetune/valid_eval/valid_probabilities.npz"

mkdir -p "$OUT"
if [[ -e "$OUT/results.json" || -e "$OUT/run.log" ]]; then
  echo "Refusing to reuse existing Model348 artifacts under: $OUT" >&2
  exit 1
fi

echo "Waiting for Model347 training and official validation inference." | tee -a "$OUT/queue.log"
while [[ ! -s "$MODEL347_QUEUE" ]] \
   || ! grep -q "DONE: Model347 weighted-CE HRNet fine-tune and official valid evaluation complete." "$MODEL347_QUEUE"; do
  sleep 60
done

cd "$ROOT"
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 nice -n 19 "$PY" -u \
    "$ROOT/model346_model336_into_341_route.py" \
    --root "$ROOT" --expert-id 347 --expert-probabilities "$MODEL347_PROBS" \
    --workers 4 --bootstrap 20000 --output "$OUT/results.json" \
  2>&1 | tee "$OUT/run.log"
echo "DONE: Model348 Model347 expert into Model341 route." | tee -a "$OUT/queue.log"
