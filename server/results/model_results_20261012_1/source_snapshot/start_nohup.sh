#!/usr/bin/env bash
set -euo pipefail
fusion805_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -ne 0 ]; then
  printf '%s\n' 'This launcher resumes the prepared run only. Usage: ./start_nohup.sh' >&2
  exit 2
fi
fusion805_resume="$(/home/sppark/miniconda3/envs/komap/bin/python -B -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); r=Path(json.loads((p/"resume_target.json").read_text())["run"]); assert r.is_relative_to(p/"outputs/runs") and (r/"resume_provenance.json").is_file(); print(r)' "$fusion805_root")"
mkdir -p "$fusion805_root/logs"
fusion805_log="$(mktemp "$fusion805_root/logs/train_$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)_XXXXXX.log")"
nohup "$fusion805_root/run.sh" train --execute --resume "$fusion805_resume" > "$fusion805_log" 2>&1 < /dev/null &
fusion805_pid=$!
ln -sfn "$fusion805_log" "$fusion805_root/logs/nohup_latest.log"
printf '%s\n' "$fusion805_pid" > "$fusion805_log.pid"
printf 'PID: %s\nLog: %s\n' "$fusion805_pid" "$fusion805_log"
printf 'Resume run: %s\nGPU: 4,5,6,7\n' "$fusion805_resume"
printf 'Progress: tail -f %q\nStop: kill -TERM %s\n' "$fusion805_log" "$fusion805_pid"
