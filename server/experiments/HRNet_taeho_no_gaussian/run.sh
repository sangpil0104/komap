#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${HRNET_GPU:-1}"
case "$CUDA_VISIBLE_DEVICES" in
  0|1|2|3|4|5) ;;
  *) echo 'HRNET_GPU must be a single GPU index from 0 to 5.' >&2; exit 2 ;;
esac
hrnet_python="${HRNET_PYTHON:-$PWD/.venv/bin/python}"
if [[ ! -x "$hrnet_python" && -z "${HRNET_PYTHON:-}" ]]; then
  hrnet_python="$PWD/../HRNet_taeho/.venv/bin/python"
fi
if [[ ! -x "$hrnet_python" ]]; then
  echo 'Python environment missing. Run bash setup.sh first.' >&2
  exit 1
fi
exec "$hrnet_python" -B -u run.py "$@"
