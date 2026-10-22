#!/usr/bin/env bash
set -euo pipefail
LHS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NO_ALBUMENTATIONS_UPDATE=1
exec "$LHS_DIR/.venv/bin/python" -u "$LHS_DIR/run.py" "$@"
