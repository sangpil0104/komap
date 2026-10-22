#!/usr/bin/env bash
set -euo pipefail
exh14_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /home/sppark/miniconda3/etc/profile.d/conda.sh
conda activate komap
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1
unset PYTHONPATH
exec "$CONDA_PREFIX/bin/python" -B -u "$exh14_root/run.py" "$@"
