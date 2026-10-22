#!/usr/bin/env bash
set -euo pipefail
exh14_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$exh14_root/logs"
exh14_log="$(mktemp "$exh14_root/logs/evaluate_$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)_XXXXXX.log")"
nohup "$exh14_root/run.sh" evaluate --execute "$@" > "$exh14_log" 2>&1 < /dev/null &
exh14_pid=$!
printf 'PID: %s\nLog: %s\n' "$exh14_pid" "$exh14_log"
printf 'Progress: tail -f %q\nStop: kill -TERM %s\n' "$exh14_log" "$exh14_pid"
