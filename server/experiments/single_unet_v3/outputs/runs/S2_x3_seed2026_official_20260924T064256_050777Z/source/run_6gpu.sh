#!/usr/bin/env bash
set -euo pipefail
experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_python="${KOMAP_PYTHON:-/home/sppark/miniconda3/envs/komap/bin/python}"
export CUDA_VISIBLE_DEVICES=0,1,2,5,6,7
exec "$experiment_python" -B "$experiment_dir/run.py" continue-suite --gpus 6 "$@"
