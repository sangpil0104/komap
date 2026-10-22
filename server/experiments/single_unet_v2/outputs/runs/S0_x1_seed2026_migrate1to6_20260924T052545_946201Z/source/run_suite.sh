#!/usr/bin/env bash
set -euo pipefail
experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_python="${KOMAP_PYTHON:-/home/sppark/miniconda3/envs/komap/bin/python}"
exec "$experiment_python" -B "$experiment_dir/run.py" suite "$@"
