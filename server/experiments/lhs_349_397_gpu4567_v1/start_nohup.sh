#!/usr/bin/env bash
set -euo pipefail
lhs_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$lhs_root/outputs/logs"
lhs_log="$(mktemp "$lhs_root/outputs/logs/train_$(date -u +%Y%m%dT%H%M%S)_XXXXXX.log")"
nohup "$lhs_root/run.sh" train --execute "$@" > "$lhs_log" 2>&1 < /dev/null &
lhs_pid=$!
printf '%s\n' "$lhs_pid" > "$lhs_log.pid"
printf 'PID=%s\nLOG=%s\n진행 확인: tail -f %q\n' "$lhs_pid" "$lhs_log" "$lhs_log"
