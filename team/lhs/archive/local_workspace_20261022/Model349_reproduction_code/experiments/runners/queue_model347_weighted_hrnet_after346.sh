#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/komap_model1
PY=/workspace/venv-komap/bin/python
GPU_ID=6
PREVIOUS_QUEUE="$ROOT/outputs/model346_model336_into_341_route/queue.log"
OUT="$ROOT/outputs/model347_hrnet_inverse_ce_finetune"

mkdir -p "$OUT"
if [[ -e "$OUT/last.pt" || -e "$OUT/valid_eval/valid_probabilities.npz" ]]; then
  echo "Refusing to overwrite existing Model347 artifacts under: $OUT" >&2
  exit 1
fi

echo "Waiting for Model346 route evaluation to finish." | tee -a "$OUT/queue.log"
while [[ ! -s "$PREVIOUS_QUEUE" ]] \
   || ! grep -q "DONE: Model346 Model336 expert into Model341 route." "$PREVIOUS_QUEUE"; do
  sleep 60
done

minimum_free=$((14 * 1024 * 1024 * 1024 + 512 * 1024 * 1024))
stable=0
while (( stable < 3 )); do
  free_bytes=$(env -u CUDA_VISIBLE_DEVICES GPU_ID="$GPU_ID" "$PY" -c \
    'import os,torch; print(torch.cuda.mem_get_info(int(os.environ["GPU_ID"]))[0])')
  if (( free_bytes >= minimum_free )); then
    stable=$((stable + 1))
    echo "P100 headroom check $stable/3 passed" | tee -a "$OUT/queue.log"
  else
    stable=0
    echo "Waiting for P100 headroom before Model347" | tee -a "$OUT/queue.log"
  fi
  if (( stable < 3 )); then sleep 60; fi
done

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
"$PY" -u model336_hrnet_best/train.py \
  --data-root "$ROOT/data" \
  --output-dir "$OUT" \
  --batch-size 4 \
  --accumulation-steps 2 \
  --epochs 12 \
  --samples-per-image 12 \
  --lr 1e-4 \
  --encoder-lr-mult 0.3 \
  --weight-decay 1e-4 \
  --boundary-head-weight 1.0 \
  --seed 347 \
  --val-every 0 \
  --init-checkpoint "$ROOT/outputs/model336_hrnet_best/last.pt" \
  --ce-class-weights 0.22906544 0.49862612 1.39166570 1.88064275 \
  2>&1 | tee "$OUT/train.log"

"$PY" -u model336_hrnet_best/evaluate_image_mean.py \
  --checkpoint "$OUT/last.pt" \
  --data-root "$ROOT/data" \
  --split valid \
  --output-dir "$OUT/valid_eval" \
  --device cuda:0 \
  --batch-size 2 2>&1 | tee "$OUT/valid_eval.log"

echo "DONE: Model347 weighted-CE HRNet fine-tune and official valid evaluation complete." \
  | tee -a "$OUT/queue.log"
