#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
GPU_LOCK="$ROOT/outputs/model409_instance_metric_v2/.model409_instance_metric_oof.lock"
RUNNER="$ROOT/experiments/run_model410_component_sampler_oof.sh"

mkdir -p "$(dirname "$GPU_LOCK")"
while true; do
  exec 9>>"$GPU_LOCK"
  if flock -n 9; then
    flock -u 9
    exec 9>&-
    exec bash "$RUNNER"
  fi
  exec 9>&-
  sleep 30
done
