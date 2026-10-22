#!/usr/bin/env bash
set -u

BASE=/workspace/komap_model1
PY=/workspace/venv-komap/bin/python
SCRIPT=$BASE/train_stronger_models.py
DATA=$BASE/data
OUT=$BASE/outputs

# Wait for the currently running EfficientNet-B4 U-Net reproduction.
while [ -d /proc/435828 ]; do
  sleep 10
done

run_model() {
  local model="$1"
  local output="$2"
  local batch="$3"
  mkdir -p "$OUT/$output"
  echo "START $model $(date -u)" | tee "$OUT/$output/runner.log"
  CUDA_VISIBLE_DEVICES=3 "$PY" "$SCRIPT" \
    --model "$model" \
    --data-dir "$DATA" \
    --output-dir "$OUT/$output" \
    --epochs 30 \
    --batch-size "$batch" \
    --num-workers 4 \
    > "$OUT/$output/train.log" 2>&1
  local train_status=$?
  if [ "$train_status" -eq 0 ] && [ -f "$OUT/$output/best.pt" ]; then
    CUDA_VISIBLE_DEVICES=3 "$PY" "$SCRIPT" \
      --model "$model" \
      --data-dir "$DATA" \
      --output-dir "$OUT/$output" \
      --batch-size "$batch" \
      --num-workers 2 \
      --checkpoint "$OUT/$output/best.pt" \
      --predict-test \
      >> "$OUT/$output/train.log" 2>&1
  fi
  echo "DONE $model status=$train_status $(date -u)" | tee -a "$OUT/$output/runner.log"
}

run_model effb4_unetpp model82_effb4_unetpp 2
run_model effb5_unetpp model83_effb5_unetpp 1
run_model effv2s_unetpp model84_effv2s_unetpp 2
run_model hrnet_w32 model85_hrnet_w32 2
run_model hrnet_w48 model86_hrnet_w48 1
run_model segformer_b2 model87_segformer_b2 2
run_model segformer_b3 model88_segformer_b3 1
run_model segnext_style_t model89_segnext_style_t 2
run_model segnext_style_s model90_segnext_style_s 1

touch "$BASE/stronger_sequence.done"
echo "ALL DONE $(date -u)"
