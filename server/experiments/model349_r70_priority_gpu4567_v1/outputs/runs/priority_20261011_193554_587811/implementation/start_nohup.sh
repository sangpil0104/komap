#!/usr/bin/env bash
set -euo pipefail
priority_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$priority_root/logs"
priority_log="$(mktemp "$priority_root/logs/priority_$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)_XXXXXX.log")"
nohup "$priority_root/run.sh" run --execute "$@" > "$priority_log" 2>&1 < /dev/null &
priority_pid=$!
ln -sfn "$priority_log" "$priority_root/logs/nohup_latest.log"
printf '%s\n' "$priority_pid" > "$priority_log.pid"
printf 'PID: %s\nLog: %s\n' "$priority_pid" "$priority_log"
printf 'Progress: tail -f %q\nStop workflow: kill -TERM %s\n' "$priority_log" "$priority_pid"
