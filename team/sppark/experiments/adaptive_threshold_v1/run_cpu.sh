#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
KOMAP_PYTHON="${KOMAP_PYTHON:-/home/sppark/miniconda3/envs/komap/bin/python}"
if [[ ! -x "$KOMAP_PYTHON" ]]; then
  echo "Python not found: $KOMAP_PYTHON. Set KOMAP_PYTHON to a Python environment with requirements.txt installed." >&2
  exit 1
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
if [[ $# -eq 0 ]]; then
  set -- suite
fi
exec "$KOMAP_PYTHON" -u "$EXPERIMENT_DIR/run.py" "$@"
