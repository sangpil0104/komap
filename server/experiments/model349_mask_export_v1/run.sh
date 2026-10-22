#!/usr/bin/env bash
set -euo pipefail
source /home/sppark/miniconda3/etc/profile.d/conda.sh
conda activate komap
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
model349_export_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$CONDA_PREFIX/bin/python" -B -u "$model349_export_root/export_masks.py" "$@"
