#!/usr/bin/env bash
set -euo pipefail
model349_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$model349_root/outputs/logs"
model349_log="$(mktemp "$model349_root/outputs/logs/train_$(date -u +%Y%m%dT%H%M%S)_XXXXXX.log")"
nohup "$model349_root/run.sh" train --execute "$@" > "$model349_log" 2>&1 < /dev/null &
model349_pid=$!
printf '%s\n' "$model349_pid" > "$model349_log.pid"
printf 'PID=%s\nLOG=%s\n진행 확인: tail -f %q\n' "$model349_pid" "$model349_log" "$model349_log"
