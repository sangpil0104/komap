#!/usr/bin/env bash
set -euo pipefail
model364_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$model364_root/outputs/logs"
model364_log="$(mktemp "$model364_root/outputs/logs/resume_$(date -u +%Y%m%dT%H%M%S)_XXXXXX.log")"
nohup "$model364_root/run.sh" train --execute "$@" > "$model364_log" 2>&1 < /dev/null &
model364_pid=$!
printf '%s\n' "$model364_pid" > "$model364_log.pid"
printf 'PID=%s\nLOG=%s\n진행 확인: tail -f %q\n' "$model364_pid" "$model364_log" "$model364_log"
