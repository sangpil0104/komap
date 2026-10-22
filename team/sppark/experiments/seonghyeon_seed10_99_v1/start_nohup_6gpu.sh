#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$EXPERIMENT_DIR/../.." && pwd)"
# Reuse exactly the same interpreter/packages as seeds 0–9.
PY="$PROJECT_DIR/experiments/seonghyeon_seed10_v1/.venv/bin/python"
[[ -x "$PY" ]] || { echo "Missing original seed0–9 Python environment: $PY" >&2; exit 1; }
mkdir -p "$EXPERIMENT_DIR/outputs/logs"
flock -n "$PROJECT_DIR/results/model_results_0926_2/.seed_extension.lock" true || {
  echo "A seed extension is already running; check outputs/latest.log" >&2; exit 1;
}
LOG="$(mktemp "$EXPERIMENT_DIR/outputs/logs/seeds10_99_$(date -u +%Y%m%dT%H%M%S)_XXXXXX.log")"
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 NO_ALBUMENTATIONS_UPDATE=1 \
  HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PY" -B -u "$EXPERIMENT_DIR/run.py" > "$LOG" 2>&1 < /dev/null &
SUITE_PID=$!
printf '%s\n' "$SUITE_PID" > "$LOG.pid"
ln -sfn -- "$LOG" "$EXPERIMENT_DIR/outputs/latest.log"
printf 'PID: %s\nLOG: %s\nMonitor: tail -n 80 -F "%s"\n' "$SUITE_PID" "$LOG" "$LOG"
printf 'Stop: kill -TERM %s\n' "$SUITE_PID"
