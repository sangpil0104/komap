#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PYTHON="${PYTHON:-/workspace/venv-komap/bin/python}"
MODEL403_LOCK="$ROOT/outputs/model403_local_supcon/.model403_oof.lock"
OUT_ROOT="$ROOT/outputs/model404_component_recall_v6"
WAIT_MINUTES=0

mkdir -p "$OUT_ROOT"
echo "Model404 v6 queued behind Model403; calibration and training have not started."
while true; do
  exec 8>>"$MODEL403_LOCK"
  if flock -n 8; then
    flock -u 8
    exec 8>&-
    break
  fi
  exec 8>&-
  sleep 60
  WAIT_MINUTES=$((WAIT_MINUTES + 1))
  if (( WAIT_MINUTES % 10 == 0 )); then
    date -u
    echo "Model404 v6 queued behind Model403; no calibration or training has started."
  fi
done

date -u
echo "Model403 released P100; starting locked train-only v4 calibration for v6 scorer."
export CUDA_VISIBLE_DEVICES=7
export DEVICE=cuda:0
"$PYTHON" "$ROOT/experiments/model404_component_recall_calibrate_v4.py" \
  --split-root "$ROOT/outputs/model403_local_supcon/splits/model392_cv" \
  --baseline-root "$ROOT/outputs/model392_widthnorm/seed1" \
  --out "$OUT_ROOT/calibration.json" --device "$DEVICE" \
  --probe-batches 8 --probe-seed 1
bash "$ROOT/experiments/run_model404_component_recall_oof_v6.sh" "$ROOT/data"
