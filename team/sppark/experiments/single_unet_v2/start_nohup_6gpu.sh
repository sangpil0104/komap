#!/usr/bin/env bash
set -euo pipefail
experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_python="${KOMAP_PYTHON:-/home/sppark/miniconda3/envs/komap/bin/python}"
log_dir="$experiment_dir/outputs/logs"
mkdir -p "$log_dir"
# Inherited by the Python launcher; held until its entire suite exits.
exec 9>"$log_dir/.nohup.lock"
if ! flock -n 9; then
    echo "이미 nohup 실행이 진행 중입니다. $log_dir/latest.log 를 확인하세요." >&2
    exit 1
fi
log_file="$(mktemp "$log_dir/suite_$(date -u +%Y%m%dT%H%M%S)_XXXXXX.log")"
export KOMAP_LOG_PATH="$log_file"
nohup "$experiment_python" -u -B "$experiment_dir/nohup_runner.py" "$@" >"$log_file" 2>&1 < /dev/null &
launcher_pid=$!
ln -sfn "$(basename -- "$log_file")" "$log_dir/latest.log"
printf '%s\n' "$launcher_pid" >"$log_dir/latest.pid"
echo "nohup 실행 PID: $launcher_pid"
echo "통합 로그: $log_file"
echo "확인: tail -F \"$log_dir/latest.log\""
