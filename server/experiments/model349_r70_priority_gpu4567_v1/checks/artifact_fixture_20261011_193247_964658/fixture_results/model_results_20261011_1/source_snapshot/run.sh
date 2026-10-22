#!/usr/bin/env bash
set -euo pipefail
priority_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /home/sppark/miniconda3/etc/profile.d/conda.sh
conda activate komap
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=4,5,6,7 CUDA_DEVICE_ORDER=PCI_BUS_ID
unset PYTHONPATH
exec "$CONDA_PREFIX/bin/python" -B -u "$priority_root/workflow.py" "$@"
