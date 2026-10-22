#!/usr/bin/env bash
set -euo pipefail
seed100_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$seed100_root/outputs/logs"
seed100_log="$(mktemp "$seed100_root/outputs/logs/search_$(date -u +%Y%m%dT%H%M%S)_XXXXXX.log")"
nohup "$seed100_root/run.sh" train --execute "$@" > "$seed100_log" 2>&1 < /dev/null &
seed100_pid=$!
printf '%s\n' "$seed100_pid" > "$seed100_log.pid"
printf 'PID=%s\nLOG=%s\n진행 확인: tail -f %q\n' "$seed100_pid" "$seed100_log" "$seed100_log"
