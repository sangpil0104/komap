#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ $# -ne 0 ]]; then
  echo 'Set HRNET_GPUS to choose devices; no positional arguments.' >&2; exit 2
fi
mkdir -p outputs/logs
hrnet_log=$(mktemp "$PWD/outputs/logs/HRNet_taeho_0928_multigpu_$(date -u +%Y%m%dT%H%M%S)_XXXXXX.log")
nohup ./run_sequence.sh > "$hrnet_log" 2>&1 < /dev/null &
echo "PID=$!"
echo "LOG=$hrnet_log"
echo "진행 확인: tail -n 80 -f '$hrnet_log'"
