#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
RESULT="$ROOT/experiments/results/model410_component_sampler_oof.json"
LOCK="$ROOT/outputs/model409_instance_metric_v2/.model409_instance_metric_oof.lock"
RUNNER="$ROOT/experiments/run_model411_bicc_oof.sh"

# Wait for the predeclared Model410 train-OOF screen. If it passes, preserve
# compute and stop; if it fails, Model411 may run only after the shared P100
# lock is free. This queue never reads runner logs or accesses official valid/test.
while [[ ! -s "$RESULT" ]]; do
  if ! pgrep -f 'run_model409_instance_metric_oof.sh|queue_model410_component_sampler_after409.sh|run_model410_component_sampler_oof.sh' >/dev/null; then
    echo "Model410 ended without a one-shot result; Model411 will not start." >&2
    exit 13
  fi
  sleep 30
done

if ! /workspace/venv-komap/bin/python -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get("valid_read") is False and r.get("test_read") is False and r.get("preregistered_gate",{}).get("pass") is False else 1)' "$RESULT"; then
  echo "Model410 passed its fixed screen or result scope is invalid; Model411 remains unstarted."
  exit 0
fi

mkdir -p "$(dirname "$LOCK")"
while true; do
  exec 9>>"$LOCK"
  if flock -n 9; then
    flock -u 9
    exec 9>&-
    exec bash "$RUNNER"
  fi
  exec 9>&-
  sleep 30
done
