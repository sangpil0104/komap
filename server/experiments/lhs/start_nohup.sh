#!/usr/bin/env bash
set -euo pipefail
LHS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$LHS_DIR/outputs/logs"
LHS_LOG="$LHS_DIR/outputs/logs/model326_$(date -u +%Y%m%d_%H%M%S_%N).log"
nohup "$LHS_DIR/run.sh" train326 --execute --gpus "${LHS_GPUS:-0,1,2,3,4,5}" "$@" > "$LHS_LOG" 2>&1 < /dev/null &
LHS_PID=$!
printf 'PID: %s\nLOG: %s\nProgress: tail -f %q\n' "$LHS_PID" "$LHS_LOG" "$LHS_LOG"
