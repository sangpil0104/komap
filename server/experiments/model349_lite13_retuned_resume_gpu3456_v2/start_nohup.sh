#!/usr/bin/env bash
set -euo pipefail
lite13_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$lite13_root/logs"
exec 9>"$lite13_root/.run.lock"
flock -n 9 || { echo 'Resume is already running.'; exit 1; }
lite13_log=$(mktemp "$lite13_root/logs/resume_$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)_XXXXXX.log")
nohup "$lite13_root/run.sh" resume --execute --lock-held >"$lite13_log" 2>&1 < /dev/null &
lite13_pid=$!
printf '%s\n' "$lite13_pid" > "$lite13_root/logs/latest.pid"
ln -sfn "$(basename "$lite13_log")" "$lite13_root/logs/latest.log"
printf 'PID: %s\nLog: %s\n' "$lite13_pid" "$lite13_log"
printf 'tail -f %q\n' "$lite13_root/logs/latest.log"
