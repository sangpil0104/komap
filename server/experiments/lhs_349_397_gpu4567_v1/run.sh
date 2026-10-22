#!/usr/bin/env bash
set -euo pipefail
lhs_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /home/sppark/miniconda3/etc/profile.d/conda.sh
conda activate komap
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MODEL349_GPUS=4,5,6,7
exec "$CONDA_PREFIX/bin/python" -B -u "$lhs_root/run_all.py" "$@"
