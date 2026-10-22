#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p outputs/logs
hrnet_log=$(mktemp "$PWD/outputs/logs/HRNet_taeho_0928_$(date -u +%Y%m%dT%H%M%S)_XXXXXX.log")
nohup ./run.sh train --execute "$@" > "$hrnet_log" 2>&1 < /dev/null &
hrnet_pid=$!
echo "PID=$hrnet_pid"
echo "LOG=$hrnet_log"
echo "진행 확인: tail -n 80 -f '$hrnet_log'"
