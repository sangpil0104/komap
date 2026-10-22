#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${HRNET_GPU:-0}"
case "$CUDA_VISIBLE_DEVICES" in
  0|1|2|3|4|5) ;;
  *) echo 'HRNET_GPU must be a single GPU index from 0 to 5.' >&2; exit 2 ;;
esac
if [[ ! -x .venv/bin/python ]]; then
  echo 'Python environment missing. Run bash setup.sh first.' >&2
  exit 1
fi
exec .venv/bin/python -B -u run.py "$@"
