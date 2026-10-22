#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$EXPERIMENT_DIR/logs"
# Keep this descriptor in the child so a second launch cannot interleave a run.
exec 9>"$EXPERIMENT_DIR/logs/run.lock"
if ! flock -n 9; then
  echo "An adaptive-threshold run is already active; see logs/latest.log." >&2
  exit 1
fi
RUN_LOG="$(mktemp "$EXPERIMENT_DIR/logs/run_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
nohup "$EXPERIMENT_DIR/run_cpu.sh" "$@" >"$RUN_LOG" 2>&1 < /dev/null 9>&9 &
RUN_PID=$!
echo "$RUN_PID" >"${RUN_LOG}.pid"
ln -sfn -- "$(basename -- "$RUN_LOG")" "$EXPERIMENT_DIR/logs/latest.log"
printf 'PID=%s\nLOG=%s\nFollow: tail -f "%s"\n' "$RUN_PID" "$RUN_LOG" "$RUN_LOG"
