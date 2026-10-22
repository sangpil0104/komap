#!/usr/bin/env bash
set -euo pipefail
model364_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
model364_conda="/home/sppark/miniconda3"
source "$model364_conda/etc/profile.d/conda.sh"
conda activate komap
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
exec "$CONDA_PREFIX/bin/python" -B -u "$model364_root/run.py" "$@"
