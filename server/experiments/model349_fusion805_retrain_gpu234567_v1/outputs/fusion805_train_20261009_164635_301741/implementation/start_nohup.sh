#!/usr/bin/env bash
set -euo pipefail
fusion805_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$fusion805_root/logs"
fusion805_log="$(mktemp "$fusion805_root/logs/train_$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)_XXXXXX.log")"
nohup "$fusion805_root/run.sh" train --execute "$@" > "$fusion805_log" 2>&1 < /dev/null &
fusion805_pid=$!
printf 'PID: %s\nLog: %s\n' "$fusion805_pid" "$fusion805_log"
printf 'Progress: tail -f %q\nStop: kill -TERM %s\n' "$fusion805_log" "$fusion805_pid"
