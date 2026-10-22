#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/komap_model1
PY=/workspace/venv-komap/bin/python
MODEL350_QUEUE="$ROOT/outputs/model350_model346_fine_gamma/queue.log"
MODEL351_QUEUE="$ROOT/outputs/model351_model336_d4_tta/queue.log"
MODEL351_PROBS="$ROOT/outputs/model351_model336_d4_tta/valid_eval_d4/valid_probabilities.npz"
OUT="$ROOT/outputs/model352_d4_hrnet_into_341_route"

mkdir -p "$OUT"
if [[ -e "$OUT/results.json" || -e "$OUT/run.log" ]]; then
  echo "Refusing to reuse existing Model352 artifacts under: $OUT" >&2
  exit 1
fi

echo "Waiting for Model350 and Model351 prerequisites." | tee -a "$OUT/queue.log"
while [[ ! -s "$MODEL350_QUEUE" ]] \
   || ! grep -Fq "DONE: Model350 fine gamma search around Model346 route." "$MODEL350_QUEUE" \
   || [[ ! -s "$MODEL351_QUEUE" ]] \
   || ! grep -Fq "DONE: Model351 8-way dihedral TTA valid inference." "$MODEL351_QUEUE"; do
  sleep 60
done

cd "$ROOT"
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 nice -n 19 "$PY" -u \
    "$ROOT/model346_model336_into_341_route.py" \
    --root "$ROOT" --expert-id 336 --experiment-id 352 \
    --expert-probabilities "$MODEL351_PROBS" \
    --workers 4 --bootstrap 20000 \
    --output "$OUT/results.json" \
  2>&1 | tee "$OUT/run.log"
echo "DONE: Model352 D4-HRNet expert into Model341 route." \
  | tee -a "$OUT/queue.log"
