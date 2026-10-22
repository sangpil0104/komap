#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/komap_model1
PY=/workspace/venv-komap/bin/python
OUT="$ROOT/outputs/model336_hrnet_best"
WAIT_PID=750884
GPU_ID=6
QUEUE_LOG="$OUT/queue.log"
mkdir -p "$OUT"

echo "Waiting for the existing Model326->328->329->333 queue (PID $WAIT_PID)." \
  | tee -a "$QUEUE_LOG"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done

echo "Existing queue ended; waiting for exclusive P100 headroom." | tee -a "$QUEUE_LOG"
minimum_free=$((14 * 1024 * 1024 * 1024 + 512 * 1024 * 1024))
stable=0
while (( stable < 3 )); do
  free_bytes=$(env -u CUDA_VISIBLE_DEVICES GPU_ID="$GPU_ID" "$PY" -c \
    'import os,torch; print(torch.cuda.mem_get_info(int(os.environ["GPU_ID"]))[0])')
  if (( free_bytes >= minimum_free )); then
    stable=$((stable + 1))
    echo "P100 GPU $GPU_ID headroom check $stable/3 passed" | tee -a "$QUEUE_LOG"
  else
    stable=0
    echo "P100 GPU $GPU_ID is still occupied; retrying in 60s" | tee -a "$QUEUE_LOG"
  fi
  if (( stable < 3 )); then sleep 60; fi
done

export CUDA_VISIBLE_DEVICES="$GPU_ID"
cd "$ROOT"
"$PY" -u model336_hrnet_best/train.py \
  --data-root "$ROOT/data" \
  --output-dir "$OUT" \
  --batch-size 4 \
  --accumulation-steps 2 \
  --epochs 70 \
  --resume \
  --samples-per-image 12 \
  --lr 3e-4 \
  --encoder-lr-mult 1.0 \
  --weight-decay 1e-4 \
  --boundary-head-weight 1.0 \
  --seed 0 \
  --val-every 5 2>&1 | tee "$OUT/train.log"

"$PY" -u model336_hrnet_best/evaluate_image_mean.py \
  --checkpoint "$OUT/last.pt" \
  --data-root "$ROOT/data" \
  --split valid \
  --output-dir "$OUT/valid_eval" \
  --device cuda:0 \
  --batch-size 2 2>&1 | tee "$OUT/valid_eval.log"

"$PY" -u model336_hrnet_best/predict.py \
  --checkpoint "$OUT/last.pt" \
  --images-dir "$ROOT/data/test/images" \
  --output-dir "$OUT/test_predictions" \
  --device cuda:0 \
  --batch-size 2 2>&1 | tee "$OUT/test_predict.log"

mkdir -p "$ROOT/outputs/model337_hrnet_model270_routes"
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 nice -n 19 "$PY" -u model337_hrnet_model270_routes.py \
    --entries "$ROOT/outputs/model183_metric_calibration/entries.pkl" \
    --model195 "$ROOT/outputs/model195_native_resume/valid_probs.npz" \
    --tent20 "$ROOT/outputs/model220_native_tent_extended/valid_probs_06.npz" \
    --model269 "$ROOT/outputs/model269_fullstats_1x_extended/valid_probs.npz" \
    --model336 "$OUT/valid_eval/valid_probabilities.npz" \
    --valid-masks "$ROOT/data/valid/masks" \
    --workers 8 \
    --output "$ROOT/outputs/model337_hrnet_model270_routes/route_probe.json" \
  2>&1 | tee "$ROOT/outputs/model337_hrnet_model270_routes/route_probe.log"

echo "DONE: Model336 training/validation/test and Model337 HRNet route search complete." \
  | tee -a "$QUEUE_LOG"
