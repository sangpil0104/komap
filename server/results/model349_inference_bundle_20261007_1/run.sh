#!/usr/bin/env bash
set -euo pipefail
model349_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
if [[ "${CONDA_DEFAULT_ENV:-}" == "komap" ]]; then
    exec "${CONDA_PREFIX}/bin/python" -B -u "${model349_root}/infer.py" "$@"
elif command -v conda >/dev/null 2>&1; then
    exec conda run --no-capture-output -n komap python -B -u "${model349_root}/infer.py" "$@"
else
    echo 'Conda komap 환경을 활성화하세요: conda activate komap' >&2
    exit 1
fi
