#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export ALBUMENTATIONS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
exec /Data3/sppark/komap/experiments/seonghyeon_seed10_v1/.venv/bin/python -m alphase.cli "$@"
