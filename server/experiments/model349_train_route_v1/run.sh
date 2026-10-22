#!/usr/bin/env bash
set -euo pipefail
model349_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
model349_conda="/home/sppark/miniconda3"
source "$model349_conda/etc/profile.d/conda.sh"
conda activate komap
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
exec "$CONDA_PREFIX/bin/python" -B -u "$model349_root/run.py" "$@"
