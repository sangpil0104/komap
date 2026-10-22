#!/usr/bin/env bash
set -u

BASE=/workspace/komap_model1
PY=/workspace/venv-komap/bin/python
SCRIPT=$BASE/train_stronger_models.py

while [ -d /proc/436112 ]; do
  sleep 10
done

OUT=$BASE/outputs/model84b_effv2s_unetpp
mkdir -p "$OUT"
echo "START effv2s_unetpp $(date -u)" | tee "$OUT/runner.log"
CUDA_VISIBLE_DEVICES=3 "$PY" "$SCRIPT" \
  --model effv2s_unetpp \
  --data-dir "$BASE/data" \
  --output-dir "$OUT" \
  --epochs 30 \
  --batch-size 2 \
  --num-workers 4 \
  > "$OUT/train.log" 2>&1
status=$?
if [ "$status" -eq 0 ] && [ -f "$OUT/best.pt" ]; then
  CUDA_VISIBLE_DEVICES=3 "$PY" "$SCRIPT" \
    --model effv2s_unetpp \
    --data-dir "$BASE/data" \
    --output-dir "$OUT" \
    --batch-size 2 \
    --num-workers 2 \
    --checkpoint "$OUT/best.pt" \
    --predict-test \
    >> "$OUT/train.log" 2>&1
fi
echo "DONE effv2s_unetpp status=$status $(date -u)" | tee -a "$OUT/runner.log"
touch "$BASE/effv2_recovery.done"
